# Gesture-Controlled Robotic Arm with CUDA-Accelerated Perception

A hand-gesture recognition pipeline whose image preprocessing runs on **CUDA kernels written from scratch**, built as the perception layer for a simulated **6-DOF robotic arm with spherical wrist** (ROS2 + Gazebo). Recognised gestures will drive simple Cartesian motions of the end effector (right / left / up / down), with the **inverse kinematics computed live** rather than replayed from stored poses.

The project ties together three areas that are usually studied separately: systems programming in C++/CUDA, machine learning, and robotics (ROS2, kinematics).

## Status

| Part | State |
|---|---|
| CUDA preprocessing (resize, normalization, HWC→CHW) | Done, verified against torchvision, benchmarked |
| Python ↔ CUDA binding (`ctypes`) | Done |
| Gesture classifier (5 classes, MobileNetV2 fine-tuned) | Done — 97.3 % validation accuracy |
| Hand detection / cropping before the classifier | Not started (see [Limitations](#limitations)) |
| ROS2 nodes, Docker, Gazebo simulation | Not started |
| 6-DOF arm model (URDF, DH parameters), analytic IK for the spherical wrist | Not started |
| Gesture → Cartesian delta mapping, full integration | Not started |

Detailed phases are in [`docs/scaletta-progetto-ros2-cuda-ml.md`](docs/scaletta-progetto-ros2-cuda-ml.md) (in Italian).

## How it works

```
camera frame (H×W×3, uint8, RGB)
      │
      ▼
CUDA pipeline (cuda/cuda_wrappers.cu)
  1. resize to 224×224 by area averaging  (one thread per output pixel)
  2. ImageNet mean/std normalization       (per-channel, float32)
  3. HWC → CHW layout conversion
      │
      ▼
MobileNetV2 classifier (PyTorch) ──► fist | like | palm | peace | rock
      │
      ▼
[planned] gesture → Cartesian delta → live inverse kinematics → arm (ROS2)
```

The CUDA library is compiled to a shared object and called from Python through `ctypes`; `python/src/cuda_preprocess.py` validates the input (dtype, shape, contiguity) before handing raw pointers to the GPU.

## Results

### Classifier

- **Dataset:** the 5 classes `fist`, `like`, `palm`, `peace`, `rock` from the public HaGRID classification set (512 px, hand crops), 80/20 train/validation split with a fixed seed.
- **Model:** MobileNetV2 pretrained on ImageNet (transfer learning), new 5-class output layer, last feature blocks (`features[15:]`) fine-tuned with a lower learning rate than the head.
- **Validation accuracy:** 74 % with only the new output layer trainable (underfitting) → **97.3 %** after partial fine-tuning. Train and validation loss are close, so no sign of overfitting.
- **Main confusion:** `peace` ↔ `rock`.
- Informal check on a small set of my own photos (not published): 14/15 correct. Too small to be a measurement.

### CUDA preprocessing vs. the torchvision training pipeline

The classifier was trained with torchvision (`Resize` bilinear with antialias, `ToTensor`, `Normalize`). Replacing it with the CUDA pipeline, on 500 random validation images:

| | torchvision | CUDA |
|---|---|---|
| Validation accuracy (500 images) | 97.6 % | 97.4 % |
| Same predicted class | 495 / 500 (99 %) | |

The difference is one image out of 500, within noise. The remaining disagreements come from the resize: the CUDA kernel averages the source block of each output pixel (box filter), while torchvision uses a triangular filter. A first version of the kernel used nearest-neighbour sampling; its mean pixel error against the training pipeline was about nine times larger on a test image, which is why it was replaced.

Reproduce with `python3 compare_batch.py` (from `python/src`).

### Benchmark

Preprocessing time per frame, starting from a NumPy RGB array. CPU = PIL + torchvision (the training transform). CUDA time includes the per-call allocation, host↔device copies and kernels. Median of 200 runs after warm-up, three separate runs; synthetic random images; **NVIDIA GeForce RTX 5050 Laptop GPU**.

| Input size | CPU (ms) | CUDA (ms) | Speedup |
|---|---|---|---|
| 512×512 | 1.1 – 1.3 | 0.9 – 1.0 | 1.0× – 1.3× |
| 640×480 | 1.4 – 1.6 | 0.8 – 0.9 | 1.5× – 1.9× |
| 1920×1080 | 5.1 – 5.5 | 1.3 – 1.4 | 3.8× – 4.2× |

A MobileNetV2 forward pass on one image takes about **2.7 ms** on the same GPU. At 640×480 the preprocessing therefore saves roughly 0.6 ms per frame out of ~4 ms; at 1920×1080 the CPU preprocessing alone would be the bottleneck. CUDA time barely grows with resolution, which suggests it is dominated by fixed per-call costs (not profiled yet).

Reproduce with `python3 benchmark.py`.

## Limitations

- **Domain shift.** HaGRID classification images are tight hand crops. A full camera frame, where the hand is a small part of the image, is misclassified (for example, a photo of a "rock" gesture predicted as `like` with 96 % confidence, identically by both pipelines). A hand detection / ROI step is needed before the classifier, and temporal voting across frames would help with unstable predictions.
- **Resize is not bit-identical** to torchvision's antialiased bilinear (see above); accuracy impact on the validation set is within noise, but not zero.
- **Fixed per-call overhead.** `process_image` allocates and frees GPU buffers on every call. Persistent buffers, pinned host memory and fewer synchronizations are obvious next optimizations; the gain would show mostly at low resolutions.
- Input must be RGB. OpenCV frames (BGR) have to be converted before calling the pipeline.

## Repository layout

```
cuda/                   CUDA kernels and C wrapper (cuda_wrappers.cu), standalone test (resize_kernel.cu)
python/src/
  config.py             single source of truth: paths, classes, image size, ImageNet statistics
  dataset.py            ImageFolder, augmentation, train/validation split
  model.py              build_model(): MobileNetV2 with 5-class head
  train.py              fine-tuning loop with best-checkpoint saving
  evaluate.py           confusion matrix, per-class accuracy
  inference.py          load model, predict from a preprocessed tensor
  cuda_preprocess.py    Python binding to the CUDA pipeline
  compare_preprocessing.py   single-image comparison CUDA vs torchvision
  compare_batch.py      same comparison over many validation images
  benchmark.py          CPU vs CUDA timing
python/scripts/         dataset download
docs/                   roadmap
```

## Running it

Requirements: Python 3, PyTorch + torchvision, NumPy, Pillow, matplotlib, `huggingface_hub` (dataset download only), an NVIDIA GPU with the CUDA toolkit (`nvcc`).

```bash
# 1. Build the CUDA library
cd cuda
nvcc -shared -Xcompiler -fPIC cuda_wrappers.cu -o cuda_wrappers.so
cd ..

# 2. Dataset: downloads the HaGRID classification archive from Hugging Face (several GB),
#    extracts it and keeps only the 5 classes under
#    data/hagrid/hagrid-classification-512p-no-gesture-150k/{fist,like,palm,peace,rock}
pip install huggingface_hub
python3 python/scripts/download_dataset.py

# 3. Train / evaluate (from python/src)
cd python/src
python3 train.py          # note: overwrites the checkpoint in models/
python3 evaluate.py

# 4. Predict on one image, compare pipelines, benchmark
python3 inference.py path/to/image.jpg
python3 compare_batch.py
python3 benchmark.py
```

## Planned work

1. Docker image with ROS2 Jazzy and Gazebo; ROS2 nodes for camera, inference and arm control.
2. 6-DOF arm with spherical wrist: URDF, DH parameters, analytic inverse kinematics (position/orientation decoupling) computed live.
3. Gesture → Cartesian delta mapping and end-to-end integration in simulation.
4. Hand detection before the classifier; temporal voting.
5. Version 2: a Reinforcement Learning component for the arm (to be defined), planned to coincide with a Reinforcement Learning course. Possible hardware actuation layer with an Arduino (micro-ROS or serial bridge).

## Scope note

The CUDA kernels, the Python binding and the training/inference code are written by me. The classifier backbone is a pretrained MobileNetV2 (transfer learning) and the data is the public HaGRID dataset.

"""
Benchmark del preprocessing: CPU (PIL + torchvision) contro CUDA (preprocess).

Le immagini sono sintetiche (rumore casuale uint8) di tre dimensioni realistiche, così non
c'è lettura da disco nel tempo misurato: entrambe le pipeline partono da un array NumPy
(H, W, 3) uint8 in RGB, come un frame di telecamera.

Per ogni dimensione: alcune chiamate di riscaldamento scartate (la prima chiamata CUDA
include l'inizializzazione), poi N ripetizioni; si riporta la MEDIANA dei tempi, meno
sensibile ai picchi occasionali rispetto alla media.

In coda, il tempo di un forward del modello, per mettere il preprocessing in proporzione.

Uso (da python/src):
    python3 benchmark.py
"""

import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from cuda_preprocess import preprocess
from inference import load_model

SIZES = {                       # nome: (larghezza, altezza)
    "512x512 (dataset)": (512, 512),
    "640x480 (webcam)": (640, 480),
    "1920x1080 (HD)": (1920, 1080),
}
WARMUP = 10
REPEATS = 200

# Stessa pipeline del training (val_transform): è il riferimento su CPU.
cpu_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def cpu_preprocess(frame):
    """frame: (H, W, 3) uint8 RGB. Converte in immagine PIL e applica la pipeline torchvision."""
    return cpu_transform(Image.fromarray(frame))


def median_ms(function, frame):
    """Esegue function(frame) WARMUP volte (scartate) e poi REPEATS volte; restituisce la mediana in ms."""
    for _ in range(WARMUP):
        function(frame)
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        function(frame)
        times.append(time.perf_counter() - start)
    return float(np.median(times)) * 1000


def main():
    rng = np.random.default_rng(0)
    print(f"{'dimensione':<20} {'CPU (ms)':>10} {'CUDA (ms)':>10} {'speedup':>9}")

    for name, (width, height) in SIZES.items():
        frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        cpu_ms = median_ms(cpu_preprocess, frame)
        cuda_ms = median_ms(preprocess, frame)
        print(f"{name:<20} {cpu_ms:>10.3f} {cuda_ms:>10.3f} {cpu_ms / cuda_ms:>8.2f}x")

    # --- Per proporzione: quanto costa il modello su un'immagine? ---
    model, device = load_model()
    tensor = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    def forward(_):
        with torch.no_grad():
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()   # le chiamate GPU sono asincrone: aspetta la fine prima di fermare il timer

    print(f"\nForward del modello su 1 immagine ({device}): {median_ms(forward, None):.3f} ms")


if __name__ == "__main__":
    main()
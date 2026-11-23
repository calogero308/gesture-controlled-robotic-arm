"""
Confronto su molte immagini del validation set: la pipeline CUDA e quella di torchvision
predicono la stessa classe? Quanto differiscono le probabilità?

Usa gli stessi indici di validation del training (seed fisso in dataset.py), campionandone
N a caso con un seed proprio per risultati ripetibili.

Uso (da python/src):
    python3 compare_batch.py          # 500 immagini
    python3 compare_batch.py 200      # 200 immagini
"""

import sys

import numpy as np
from PIL import Image

from compare_preprocessing import cuda_pipeline, probabilities, torchvision_preprocess
from dataset import full_val, val_indices
from inference import load_model

# Seed di generazione per ripetibilità
SAMPLE_SEED = 0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    # Sottoinsieme casuale (ma ripetibile) degli indici di validation
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = rng.choice(val_indices.indices, size=min(n, len(val_indices)), replace=False)

    model, device = load_model()

    same_class = 0       # quante volte le due pipeline predicono la stessa classe
    correct_tv = 0       # accuratezza della pipeline torchvision sul campione
    correct_cuda = 0     # accuratezza della pipeline CUDA sul campione
    prob_diffs = []      # differenza massima tra probabilità, un valore per immagine

    for i in chosen:
        path, label = full_val.samples[i]            # (percorso, indice classe vera)
        image = Image.open(path).convert("RGB")

        # Chiamata alle due catene di preprocessing
        p_tv = probabilities(model, device, torchvision_preprocess(image))
        p_cuda = probabilities(model, device, cuda_pipeline(image))

        same_class += int(p_tv.argmax() == p_cuda.argmax())
        correct_tv += int(p_tv.argmax() == label)
        correct_cuda += int(p_cuda.argmax() == label)
        prob_diffs.append(np.abs(p_tv - p_cuda).max())

    total = len(chosen)
    prob_diffs = np.array(prob_diffs)
    print(f"Immagini testate: {total}")
    print(f"Stessa classe predetta: {same_class}/{total}")
    print(f"Accuratezza torchvision: {correct_tv / total:.4f}")
    print(f"Accuratezza CUDA:        {correct_cuda / total:.4f}")
    print(f"Differenza max tra probabilità - media: {prob_diffs.mean():.4f}, massima: {prob_diffs.max():.4f}")


if __name__ == "__main__":
    main()
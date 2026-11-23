"""
Confronta il preprocessing CUDA con quello di torchvision (usato in training).

Entrambe le pipeline restituiscono ora un tensore (3, 224, 224) float32 GIÀ NORMALIZZATO
con mean/std di ImageNet, pronto per il modello. Due livelli di confronto:
  1. sui pixel: quanto differiscono i due tensori;
  2. sul modello: le probabilità predette cambiano? È il test che conta davvero.

Uso (da python/src):
    python3 compare_preprocessing.py percorso/immagine.jpg

File che contiene le funzioni che verranno chiamate per fare la compare delle performance
"""

import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from cuda_preprocess import preprocess
from inference import load_model


# Pipeline a confronto
def torchvision_preprocess(image):
    """
    Pipeline di riferimento: esattamente quella del training (val_transform).
    Restituisce un array float32 (3, 224, 224) normalizzato con mean/std ImageNet.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),   # PIL -> (C, H, W), valori in [0, 1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).numpy()


def cuda_pipeline(image):
    """
    Pipeline CUDA. np.array(PIL) dà (H, W, 3) uint8 in ordine RGB: è l'input che
    preprocess() si aspetta. Restituisce (3, 224, 224) float32 già normalizzato.
    """
    return preprocess(np.array(image))


# Modello
def probabilities(model, device, array):
    """
    array: (3, 224, 224) float32 GIÀ normalizzato (non va normalizzato di nuovo).
    Restituisce le probabilità delle 5 classi come array NumPy.
    """
    tensor = torch.from_numpy(array).unsqueeze(0).to(device)   # aggiunge la dimensione batch
    with torch.no_grad():
        # Ritorna le probabilità per ogni classe (avremo un array di 5 parametri)
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


# Report
def report(name, reference, candidate):
    """Stampa differenza media e massima (assolute) tra due tensori della stessa forma."""
    diff = np.abs(reference - candidate)
    print(f"[{name}] media: {diff.mean():.4f} - max: {diff.max():.4f}")

# Main di test
def main():
    path = sys.argv[1]
    image = Image.open(path).convert("RGB")   # PIL -> RGB garantito (niente BGR di OpenCV)
    print(f"Immagine sorgente: {image.size[0]}x{image.size[1]} (W x H)")

    tv_out = torchvision_preprocess(image)    # riferimento (training)
    cuda_out = cuda_pipeline(image)           # pipeline CUDA

    # Confronto sui pixel per capire di quanto si discostano le due catene
    # Dopo la normalizzazione i valori non stanno più in [0, 1] ma circa in [-2.1, 2.6]:
    # le differenze sono quindi più grandi in valore assoluto rispetto a prima
    # (si dividono per std ~0.225, cioè si moltiplicano per ~4.4).
    print("\nShape torchvision:", tv_out.shape, "- CUDA:", cuda_out.shape)
    print(f"Range torchvision: {tv_out.min():.3f} {tv_out.max():.3f}")
    print(f"Range CUDA:        {cuda_out.min():.3f} {cuda_out.max():.3f}")

    print()
    report("CUDA vs torchvision", tv_out, cuda_out)

    # Differenza per canale: un canale molto diverso dagli altri indicherebbe un
    # errore nella scelta di mean/std o nell'ordine dei canali.
    for c, name in enumerate("RGB"):
        channel_diff = np.abs(tv_out[c] - cuda_out[c]).mean()
        print(f"  Canale {name}: media diff = {channel_diff:.4f}")

    # Confronto sul modello
    model, device = load_model()
    p_tv = probabilities(model, device, tv_out)
    p_cuda = probabilities(model, device, cuda_out)

    print("\nProbabilità del modello:")
    print(f"  {'classe':<8} {'torchvision':>12} {'CUDA':>8}")
    for i, name in enumerate(CLASSES):
        print(f"  {name:<8} {p_tv[i]:>12.4f} {p_cuda[i]:>8.4f}")
    print(f"  Classe predetta: torchvision = {CLASSES[p_tv.argmax()]}, CUDA = {CLASSES[p_cuda.argmax()]}")
    print(f"  Differenza massima tra probabilità: {np.abs(p_tv - p_cuda).max():.4f}")


if __name__ == "__main__":
    main()
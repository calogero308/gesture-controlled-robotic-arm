"""
Prova del ritaglio: stessa foto classificata intera e ritagliata (ROI centrale).

Uso (da python/src):
    python3 demo_image.py percorso/foto.jpg
Salva data/roi_check.png con il riquadro disegnato, per vedere se la mano ci sta dentro.
"""

import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from config import DATA_DIR
from cuda_preprocess import preprocess
from inference import load_model, predict
from roi import crop_roi, roi_box


def classify(model, device, rgb):
    """rgb: array (H, W, 3) uint8 in RGB. Restituisce (gesto, confidenza)."""

    # pipeline CUDA: resize, normalizzazione, layout CHW
    array = preprocess(rgb)

    # da array NumPy a tensore PyTorch
    tensor = torch.from_numpy(array)
    
    return predict(model, device, tensor)


def main():
    image = Image.open(sys.argv[1]).convert("RGB")
    rgb = np.array(image)                      # (H, W, 3) uint8, RGB

    model, device = load_model()

    gesture, confidence = classify(model, device, rgb)
    print(f"Foto intera: {gesture} ({confidence:.2%})")

    gesture, confidence = classify(model, device, crop_roi(rgb))
    print(f"Ritaglio:    {gesture} ({confidence:.2%})")

    # Disegna sul originale lo stesso riquadro usato per il ritaglio
    y0, x0, side = roi_box(*rgb.shape[:2])
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x0 + side, y0 + side], outline="red", width=5)
    out = DATA_DIR / "test_roi/roi_check.png"
    image.save(out)
    print("Salvato", out)


if __name__ == "__main__":
    main()
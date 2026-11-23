"""
Prova del ritaglio su una cartella di foto: gesto vero, predizione sulla foto intera
e predizione sul ritaglio centrale.

Il gesto vero si ricava dal NOME DEL FILE, che deve contenere il nome della classe
(per esempio fist_1.jpg, peace_casa.jpeg, rock2.jpeg). Se non ne contiene nessuno,
la foto viene classificata comunque ma non conta nel punteggio finale.

Uso (da python/src):
    python3 demo_folder.py ../../data/test_roi
    python3 demo_folder.py ../../data/test_roi 0.7     # riquadro al 70% del lato corto
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import CLASSES
from demo_image import classify
from inference import load_model
from roi import crop_roi

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Funzione che ritorna la classe vera dell'immagine passata
def true_label(path):
    """Classe il cui nome compare nel nome del file, oppure '?' se nessuna."""
    name = path.name.lower()
    for class_name in CLASSES:
        if class_name in name:
            return class_name
    return "?"


def main():
    folder = Path(sys.argv[1])
    fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    # Carica il modello
    model, device = load_model()

    print(f"Riquadro: {fraction:.0%} del lato corto\n")
    print(f"{'file':<24} {'vero':<7} {'foto intera':<18} {'ritaglio':<18}")

    labeled = correct_full = correct_crop = 0
    for path in paths:
        rgb = np.array(Image.open(path).convert("RGB"))
        label = true_label(path)
        # Classificazione dell'immagine completa e tagliata
        gesture_full, conf_full = classify(model, device, rgb)
        gesture_crop, conf_crop = classify(model, device, crop_roi(rgb, fraction))

        print(f"{path.name:<24} {label:<7} "
              f"{gesture_full:<6} ({conf_full:>6.1%})   {gesture_crop:<6} ({conf_crop:>6.1%})")

        # Controllo finale di sicurezza e incremento del conteggio
        if label != "?":
            labeled += 1
            correct_full += gesture_full == label
            correct_crop += gesture_crop == label

    print(f"\nSu {labeled} foto etichettate: corrette {correct_full} con la foto intera, "
          f"{correct_crop} con il ritaglio")


if __name__ == "__main__":
    main()
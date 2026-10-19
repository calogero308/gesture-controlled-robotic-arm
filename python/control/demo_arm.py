"""
Demo completa: gesti riconosciuti in un video -> braccio simulato.

Percorre il video con la stessa pipeline di live_demo.py (ritaglio, CUDA, modello,
filtro temporale) ma, invece di scrivere un video annotato, raccoglie il gesto stabile
di ogni fotogramma. Poi passa la lista a simulate.animate(), che muove il braccio.

Uso (da python/control):
    python3 demo_arm.py ../../data/videos/gesti1.mp4              # salva demo_arm.gif
    python3 demo_arm.py ../../data/videos/gesti1.mp4 mia_demo.gif # nome a scelta
"""

import sys
from collections import Counter
from pathlib import Path

# i moduli di percezione stanno in python/src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from camera import frames
from demo_image import classify
from gesture_filter import GestureFilter
from inference import load_model
from roi import crop_roi
from simulate import animate      # simulate aggiunge a sua volta python/arm al percorso

ROI_FRACTION = 0.6      # come in live_demo.py
EVERY_N = 3             # un comando ogni N fotogrammi: a 30 fps sono 10 passi al secondo
OUTPUT_GIF = "demo_arm.gif"


def collect_gestures(source):
    """Percorre il video e restituisce il gesto stabile di ogni fotogramma (o None)."""
    model, device = load_model()
    gesture_filter = GestureFilter()

    gestures = []
    for frame in frames(source):
        cropped = crop_roi(frame, ROI_FRACTION)
        gesture, confidence = classify(model, device, cropped)
        stable = gesture_filter.update(gesture, confidence)
        gestures.append(stable)
    return gestures


def main():
    arg = sys.argv[1]
    source = int(arg) if arg.isdigit() else arg
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_GIF

    gestures = collect_gestures(source)
    print(f"{len(gestures)} fotogrammi letti. Gesti stabili:", dict(Counter(gestures)))

    animate(gestures[::EVERY_N], out, substeps=1)


if __name__ == "__main__":
    main()
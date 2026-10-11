"""
Demo su video o webcam: per ogni frame ritaglia il riquadro, classifica con la pipeline
CUDA + modello e salva un video con riquadro, gesto e confidenza disegnati sopra.

Uso (da python/src):
    python3 live_demo.py ../../data/videos/gesti1.mp4     # file video
    python3 live_demo.py 0                                 # webcam 0 (quando sarà visibile)
Output: data/videos/out.mp4
"""

import sys
import time

import cv2

from camera import frames
from config import DATA_DIR
from demo_image import classify
from inference import load_model
from roi import crop_roi, roi_box
from gesture_filter import GestureFilter

OUTPUT_PATH = DATA_DIR / "videos" / "out.mp4"
OUTPUT_FPS = 30


def annotate(frame_rgb, stable, gesture, confidence):
    """Restituisce un'immagine BGR (il formato di OpenCV) con riquadro verde ed etichetta."""
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    y0, x0, side = roi_box(*frame_rgb.shape[:2])
    cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 3)
    text = f"{stable or '-'}  (grezzo: {gesture} {confidence:.0%})"
    cv2.putText(image, f"{gesture} {confidence:.0%}", (x0, max(y0 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return image


def main():
    arg = sys.argv[1]
    source = int(arg) if arg.isdigit() else arg      # "0" -> webcam 0, altrimenti percorso

    model, device = load_model()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    n_frames = 0
    total_time = 0.0

    gesture_filter = GestureFilter()

    for frame in frames(source):
        start = time.perf_counter()
        cropped = crop_roi(frame, 0.6)
        gesture, confidence = classify(model, device, cropped)
        stable = gesture_filter.update(gesture, confidence)
        total_time += time.perf_counter() - start
        n_frames += 1

        image = annotate(frame, stable, gesture, confidence)
        if writer is None:                           # crea il file video al primo frame
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, OUTPUT_FPS, (width, height))
        writer.write(image)

    if writer is not None:
        writer.release()
    if n_frames:
        print(f"{n_frames} frame, {total_time / n_frames * 1000:.1f} ms/frame "
              f"(ritaglio + preprocessing CUDA + modello)")
        print("Salvato", OUTPUT_PATH)
    else:
        print("Nessun frame letto: controlla la sorgente.")


if __name__ == "__main__":
    main()
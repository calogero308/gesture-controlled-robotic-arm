"""
Demo su video o webcam: per ogni frame ritaglia il riquadro, classifica con la pipeline
CUDA + modello, applica il filtro temporale e mostra/salva il risultato con riquadro,
gesto stabile e gesto grezzo disegnati sopra.

Uso (da python/src):
    python3 live_demo.py ../../data/videos/gesti1.mp4          # file video -> salva out.mp4
    python3 live_demo.py http://IP:8080/video --show --nosave  # webcam di Windows dal vivo
Opzioni:
    --show     apre una finestra con il video annotato (si chiude con il tasto q)
    --nosave   non scrive out.mp4
Output: data/videos/out.mp4
"""

import os
import sys
import time

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")   # tace gli avvisi sui font di Qt

import cv2

from camera import frames
from config import DATA_DIR
from demo_image import classify
from inference import load_model
from roi import crop_roi, roi_box
from gesture_filter import GestureFilter

OUTPUT_PATH = DATA_DIR / "videos" / "out.mp4"
OUTPUT_FPS = 30
WINDOW_NAME = "live_demo (q per uscire)"


def annotate(frame_rgb, stable, gesture, confidence):
    """Restituisce un'immagine BGR (il formato di OpenCV) con riquadro verde ed etichette."""
    image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    y0, x0, side = roi_box(*frame_rgb.shape[:2])
    cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 3)
    # gesto grezzo sopra il riquadro, gesto stabile (quello che comanda il braccio) in alto a sinistra
    cv2.putText(image, f"{gesture} {confidence:.0%}", (x0, max(y0 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(image, f"stabile: {stable or '-'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return image


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--show" in sys.argv
    save = "--nosave" not in sys.argv

    arg = args[0]
    source = int(arg) if arg.isdigit() else arg      # "0" -> webcam 0, altrimenti percorso o indirizzo

    model, device = load_model()
    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    n_frames = 0
    total_time = 0.0

    gesture_filter = GestureFilter()

    try:
        for frame in frames(source):
            start = time.perf_counter()
            cropped = crop_roi(frame, 0.6)
            gesture, confidence = classify(model, device, cropped)
            stable = gesture_filter.update(gesture, confidence)
            total_time += time.perf_counter() - start
            n_frames += 1

            image = annotate(frame, stable, gesture, confidence)

            if save:
                if writer is None:                   # crea il file video al primo frame
                    height, width = image.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, OUTPUT_FPS, (width, height))
                writer.write(image)

            if show:
                cv2.imshow(WINDOW_NAME, image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass                                         # Ctrl+C: esce in modo pulito
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if n_frames:
        print(f"{n_frames} frame, {total_time / n_frames * 1000:.1f} ms/frame "
              f"(ritaglio + preprocessing CUDA + modello)")
        if save:
            print("Salvato", OUTPUT_PATH)
    else:
        print("Nessun frame letto: controlla la sorgente.")


if __name__ == "__main__":
    main()
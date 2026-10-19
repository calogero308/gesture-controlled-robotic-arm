"""
Demo dal vivo: webcam (o video) -> gesto stabile -> braccio simulato in una seconda finestra.

Usa la stessa pipeline di live_demo.py (ritaglio, CUDA, modello, filtro). Ogni EVERY_N
fotogrammi applica il gesto stabile al braccio (arm_state.step) e manda il nuovo stato
alla finestra del braccio (arm_viewer.py, processo separato).

Uso (da python/control):
    python3 live_arm.py http://IP:8080/video          # webcam di Windows (webcam_server.py)
    python3 live_arm.py ../../data/videos/gesti1.mp4  # un video
Opzioni:
    --nowindow   non apre la finestra del video (solo il braccio)
Si esce con q nella finestra del video, chiudendo la finestra del braccio o con Ctrl+C.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# i moduli di percezione stanno in python/src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")   # tace gli avvisi sui font di Qt

import cv2

from arm_state import initial_state, step
from camera import frames
from demo_image import classify
from gesture_filter import GestureFilter
from inference import load_model
from live_demo import annotate
from roi import crop_roi

ROI_FRACTION = 0.6      # come in live_demo.py
EVERY_N = 3             # un comando ogni N fotogrammi: a 30 fps sono 10 passi al secondo
VIEWER = Path(__file__).with_name("arm_viewer.py")
WINDOW_NAME = "webcam (q per uscire)"


def send(viewer, state):
    """Manda lo stato alla finestra del braccio. Restituisce False se la finestra è stata chiusa."""
    if viewer.poll() is not None:
        return False
    message = {k: state[k] for k in ("angles", "gesture", "command", "status")}
    try:
        viewer.stdin.write(json.dumps(message) + "\n")
        viewer.stdin.flush()
    except (BrokenPipeError, OSError):
        return False
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    window = "--nowindow" not in sys.argv
    viewer_args = ["--png", sys.argv[sys.argv.index("--png") + 1]] if "--png" in sys.argv else []

    arg = args[0]
    source = int(arg) if arg.isdigit() else arg

    model, device = load_model()
    gesture_filter = GestureFilter()
    state = initial_state()

    viewer = subprocess.Popen([sys.executable, str(VIEWER), *viewer_args],
                              stdin=subprocess.PIPE, text=True)
    send(viewer, state)

    n_frames = 0
    total_time = 0.0
    try:
        for frame in frames(source):
            start = time.perf_counter()
            cropped = crop_roi(frame, ROI_FRACTION)
            gesture, confidence = classify(model, device, cropped)
            stable = gesture_filter.update(gesture, confidence)
            total_time += time.perf_counter() - start
            n_frames += 1

            if n_frames % EVERY_N == 0:
                state = step(state, stable)
                if not send(viewer, state):
                    break                                 # finestra del braccio chiusa

            if window:
                cv2.imshow(WINDOW_NAME, annotate(frame, stable, gesture, confidence))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if window:
            cv2.destroyAllWindows()
        if viewer.poll() is None:
            try:
                viewer.stdin.close()                      # il viewer, vedendo la fine dell'input, salva e chiude
            except OSError:
                pass
            viewer.wait(timeout=10)

    if n_frames:
        print(f"{n_frames} frame, {total_time / n_frames * 1000:.1f} ms/frame "
              f"(ritaglio + preprocessing CUDA + modello)")
    else:
        print("Nessun frame letto: controlla la sorgente.")


if __name__ == "__main__":
    main()

"""
Sorgente di frame: legge da una webcam o da un file video con la stessa interfaccia.

Restituisce i frame già in RGB (H, W, 3) uint8, l'ordine che si aspetta la pipeline
CUDA e che ha usato il training. OpenCV invece li legge in BGR, quindi la conversione
va fatta qui, una volta sola, all'ingresso del sistema.

Uso:
    for frame in frames("../../data/videos/gesti.mp4"):   # file video
        ...
    for frame in frames(0):                                # webcam numero 0
        ...
"""

import cv2


def frames(source):
    """
    Generatore: restituisce un frame alla volta. `source` è un indice intero (webcam)
    oppure un percorso di file (str o Path).

    È un generatore (usa yield) perché un video non va caricato tutto in memoria:
    20 secondi a 30 fps in Full HD sono 600 frame da 6 MB ciascuno, cioè circa 3.6 GB.
    Così se ne tiene in memoria uno alla volta.
    """
    if not isinstance(source, int):
        source = str(source)                 # accetta anche oggetti Path

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Impossibile aprire la sorgente video: {source}")

    try:
        while True:
            ok, frame_bgr = capture.read()   # ok = False a fine video o camera scollegata
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()                    # libera il file/la camera anche se il ciclo si interrompe
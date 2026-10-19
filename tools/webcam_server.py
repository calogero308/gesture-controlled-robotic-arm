"""
Server video per la webcam: da eseguire su WINDOWS (non in WSL2).

WSL2 non vede la webcam, quindi questo programma la legge su Windows e ne
trasmette i fotogrammi in rete come flusso MJPEG su HTTP. Dal lato WSL2 il flusso
si apre come qualsiasi altra sorgente video:

    cv2.VideoCapture("http://<indirizzo-di-windows>:8080/video")

Uso (da PowerShell, nella cartella del progetto):
    python tools\\webcam_server.py            # webcam 0
    python tools\\webcam_server.py 1          # webcam 1
    python tools\\webcam_server.py video.mp4  # un file video in ciclo, per provare senza webcam

Richiede: pip install opencv-python
"""

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

HOST = "0.0.0.0"        # ascolta su tutte le interfacce: WSL2 si collega da una rete diversa da localhost
PORT = 8080
JPEG_QUALITY = 80
WIDTH, HEIGHT = 640, 480
FILE_FPS = 30           # ritmo di riproduzione quando la sorgente è un file


class FrameGrabber(threading.Thread):
    """Legge di continuo la sorgente e tiene solo l'ultimo fotogramma.

    Così chi si collega riceve sempre il fotogramma più recente e non una coda di
    fotogrammi vecchi (che si vedrebbe come ritardo).
    """

    def __init__(self, source):
        super().__init__(daemon=True)
        self.is_file = not source.isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(source), backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire la sorgente: {source}")
        self.condition = threading.Condition()
        self.frame_id = 0
        self.jpeg = None

    def run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:                      # file finito: riparte dall'inizio
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)                      # webcam: riprova
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            with self.condition:
                self.jpeg = jpeg.tobytes()
                self.frame_id += 1
                self.condition.notify_all()
            if self.is_file:
                time.sleep(1 / FILE_FPS)

    def wait_next(self, last_id):
        """Aspetta un fotogramma più recente di last_id e restituisce (id, jpeg)."""
        with self.condition:
            self.condition.wait_for(lambda: self.frame_id > last_id)
            return self.frame_id, self.jpeg


class StreamHandler(BaseHTTPRequestHandler):
    grabber = None

    def do_GET(self):
        if self.path != "/video":
            self.send_error(404, "usa /video")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last_id = 0
        try:
            while True:
                last_id, jpeg = self.grabber.wait_next(last_id)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                      # il client si è scollegato

    def log_message(self, format, *args):
        pass                                          # niente righe di log per ogni richiesta


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    grabber = FrameGrabber(source)
    grabber.start()
    StreamHandler.grabber = grabber

    server = ThreadingHTTPServer((HOST, PORT), StreamHandler)
    print(f"Sorgente: {source}")
    print(f"Flusso attivo su http://{HOST}:{PORT}/video   (Ctrl+C per fermare)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFermato.")


if __name__ == "__main__":
    main()

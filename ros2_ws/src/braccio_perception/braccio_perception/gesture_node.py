"""
gesture_node: riceve i fotogrammi, li passa alla pipeline già scritta (ritaglio ROI -> CUDA ->
MobileNetV2 -> filtro temporale) e pubblica il gesto stabile su /gesture (std_msgs/String).
Stringa vuota = nessun gesto stabile (nessun comando).

Il codice di percezione sta in <repo>/python/src: il nodo lo importa aggiungendo quella cartella
al percorso di Python (parametro repo_dir).
"""

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

DEFAULT_REPO = "/mnt/c/Users/calog/Desktop/Secret_tasks/robot-gesti-cuda"
ROI_FRACTION = 0.6          # come in live_demo.py
LOG_EVERY = 100             # ogni quanti fotogrammi stampare il tempo medio di elaborazione


class GestureNode(Node):
    def __init__(self):
        super().__init__("gesture_node")
        self.declare_parameter("repo_dir", DEFAULT_REPO)
        self.declare_parameter("show", False)             # True: finestra con il video annotato
        repo = Path(str(self.get_parameter("repo_dir").value))
        self.show = bool(self.get_parameter("show").value)

        # i moduli di percezione si importano solo dopo aver aggiunto la cartella al percorso
        sys.path.insert(0, str(repo / "python" / "src"))
        from demo_image import classify
        from gesture_filter import GestureFilter
        from inference import load_model
        from roi import crop_roi

        self.classify = classify
        self.crop_roi = crop_roi
        self.model, self.device = load_model()
        self.filter = GestureFilter()
        if self.show:
            from live_demo import annotate
            self.annotate = annotate

        self.stable = None
        self.n_frames = 0
        self.total_time = 0.0

        self.publisher = self.create_publisher(String, "/gesture", 10)
        # coda di 1: se l'elaborazione è più lenta della camera si scartano i fotogrammi vecchi
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.on_image, 1)
        self.get_logger().info("gesture_node pronto (modello caricato)")

    def on_image(self, msg):
        if msg.encoding != "rgb8":
            self.get_logger().error(f"Codifica non supportata: {msg.encoding} (serve rgb8)", throttle_duration_sec=5.0)
            return

        start = time.perf_counter()
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        cropped = self.crop_roi(frame, ROI_FRACTION)
        gesture, confidence = self.classify(self.model, self.device, cropped)
        stable = self.filter.update(gesture, confidence)
        self.total_time += time.perf_counter() - start
        self.n_frames += 1

        out = String()
        out.data = stable or ""
        self.publisher.publish(out)

        if stable != self.stable:
            self.get_logger().info(f"gesto stabile: {stable}")
            self.stable = stable
        if self.n_frames % LOG_EVERY == 0:
            self.get_logger().info(f"{self.total_time / self.n_frames * 1000:.1f} ms/frame (ritaglio + CUDA + modello)")

        if self.show:
            import cv2
            cv2.imshow("gesture_node (q per uscire)", self.annotate(frame, stable, gesture, confidence))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

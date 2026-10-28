"""
camera_node: legge un flusso video (stream MJPEG della webcam di Windows, un file o un indice
di camera) e pubblica i fotogrammi RGB come sensor_msgs/Image su /camera/image_raw.

Il messaggio Image si costruisce a mano (senza cv_bridge): è un semplice contenitore di byte.
"""

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # parametri: si cambiano da riga di comando o dal launch senza toccare il codice
        self.declare_parameter("source", "0")
        self.declare_parameter("fps", 30.0)
        source = str(self.get_parameter("source").value)
        fps = float(self.get_parameter("fps").value)

        # un indice di camera ("0") va passato a OpenCV come numero, non come testo
        self.capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.capture.isOpened():
            self.get_logger().error(f"Impossibile aprire la sorgente: {source}")
        else:
            self.get_logger().info(f"Sorgente aperta: {source}")

        self.publisher = self.create_publisher(Image, "/camera/image_raw", 1)
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)   # chiama publish_frame a ritmo fisso

    def publish_frame(self):
        ok, frame = self.capture.read()                  # OpenCV restituisce BGR
        if not ok:
            self.get_logger().warn("Nessun fotogramma letto", throttle_duration_sec=2.0)
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # la pipeline si aspetta RGB

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        self.publisher.publish(msg)

    def destroy_node(self):
        self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)                                 # tiene il nodo in vita e fa girare il timer
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

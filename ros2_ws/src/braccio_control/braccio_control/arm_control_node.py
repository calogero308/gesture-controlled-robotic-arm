"""
arm_control_node: riceve il gesto stabile (/gesture), a ritmo fisso lo trasforma in un comando,
sposta la punta di un passo, calcola gli angoli con la cinematica inversa e pubblica
/joint_states (sensor_msgs/JointState). robot_state_publisher li usa per muovere il modello in RViz.

La logica (gesto -> comando -> nuova posizione -> angoli) è quella di <repo>/python/control/arm_state.py.
Sicurezza: se non arrivano gesti per più di gesture_timeout secondi, il gesto viene ignorato
(il braccio si ferma invece di ripetere l'ultimo comando all'infinito).
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray

DEFAULT_REPO = "/mnt/c/Users/calog/Desktop/Secret_tasks/robot-gesti-cuda"
# i nomi devono essere identici a quelli dell'URDF
JOINT_NAMES = ["base_to_shoulder", "shoulder_to_upper", "upper_arm_to_forearm"]


class ArmControlNode(Node):
    def __init__(self):
        super().__init__("arm_control_node")
        self.declare_parameter("repo_dir", DEFAULT_REPO)
        self.declare_parameter("rate", 10.0)               # comandi al secondo (10 passi/s = 10 cm/s con STEP 1 cm)
        self.declare_parameter("gesture_timeout", 0.5)     # secondi
        self.declare_parameter("sim", False)
        repo = Path(str(self.get_parameter("repo_dir").value))
        rate = float(self.get_parameter("rate").value)
        self.timeout = float(self.get_parameter("gesture_timeout").value)

        sys.path.insert(0, str(repo / "python" / "control"))   # arm_state aggiunge da solo python/arm
        from arm_state import initial_state, step
        self.arm_step = step
        self.state = initial_state()

        self.gesture = None
        self.last_gesture_time = None

        self.subscription = self.create_subscription(String, "/gesture", self.on_gesture, 10)
        self.publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.cmd_publisher = self.create_publisher(Float64MultiArray, "/arm_position_controller/commands", 10)
        self.timer = self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info("arm_control_node pronto")
        self.sim = self.get_parameter("sim").value

    def on_gesture(self, msg):
        self.gesture = msg.data or None                    # stringa vuota = nessun gesto
        self.last_gesture_time = self.get_clock().now()

    def tick(self):
        gesture = self.gesture
        if self.last_gesture_time is None:
            gesture = None
        else:
            age = (self.get_clock().now() - self.last_gesture_time).nanoseconds * 1e-9
            if age > self.timeout:
                gesture = None

        previous_status = self.state["status"]
        self.state = self.arm_step(self.state, gesture)
        status = self.state["status"]
        if status != previous_status and status.startswith("BLOCCATO"):
            self.get_logger().warn(status)

        if self.sim:
            msg = Float64MultiArray()
            msg.data = [float(a) for a in self.state["angles"]]
            self.cmd_publisher.publish(msg)
        else:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = JOINT_NAMES
            msg.position = [float(a) for a in self.state["angles"]]
            self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArmControlNode()
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

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        self.publisher_ = self.create_publisher(String, 'keyboard_input', 10)
        self.get_logger().info("Keyboard Publisher started. Type messages and press Enter...")
        self.timer = self.create_timer(0.1, self.run)

        self.get_logger().info("🚀 Keyboard Node started")

    def run(self):
        try:
            while rclpy.ok():
                user_input = input(">> ")  # 从终端读取输入
                msg = String()
                msg.data = user_input
                self.publisher_.publish(msg)
                self.get_logger().info(f"Published: {user_input}")
        except KeyboardInterrupt:
            self.get_logger().info("Keyboard Publisher stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Failed to start keyboard input node: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

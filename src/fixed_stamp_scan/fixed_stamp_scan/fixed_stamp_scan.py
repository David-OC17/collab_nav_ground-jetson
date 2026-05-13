import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanRestamper(Node):
    def __init__(self):
        super().__init__('scan_restamper')
        self.pub = self.create_publisher(LaserScan, '/scan_restamped', 10)
        self.create_subscription(LaserScan, '/scan', self.cb, 10)

    def cb(self, msg: LaserScan):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(ScanRestamper())

if __name__ == '__main__':
    main()
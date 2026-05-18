import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/wheel_odom', self.cb, 10)

    def cb(self, msg: Odometry):
        t = TransformStamped()
        #t.header.stamp = msg.header.stamp
        t.header.stamp = self.get_clock().now().to_msg()  # instead of msg.header.stamp
        t.header.frame_id = msg.header.frame_id      # e.g. wheel_odom
        t.child_frame_id  = msg.child_frame_id        # e.g. base_link
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation      = msg.pose.pose.orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(OdomToTF())

if __name__ == '__main__':
    main()

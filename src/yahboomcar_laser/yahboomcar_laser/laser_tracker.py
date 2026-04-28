#ros lib
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

#common lib
import numpy as np
from math import pi
from std_msgs.msg import Bool


class laserTracker(Node):
    def __init__(self, name):
        super().__init__(name)

        #create sub
        self.sub_laser = self.create_subscription(LaserScan, "/scan", self.registerScan, 1)
        self.sub_JoyState = self.create_subscription(Bool, "/JoyState", self.JoyStateCallback, 1)
        #create pub
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 1)

        #declare params
        self.right_warning = 0
        self.left_warning = 0
        self.front_warning = 0
        self.priorityAngle = 30
        self.offset = 0.5
        self.Joy_active = False
        self.Moving = False
        self.lin_pid = SinglePID(2.0, 0.0, 2.0)
        self.ang_pid = SinglePID(3.0, 0.0, 5.0)

        self.declare_parameter("Switch", False)
        self.Switch = self.get_parameter("Switch").get_parameter_value().bool_value
        self.declare_parameter("ResponseDist", 0.55)
        self.ResponseDist = self.get_parameter("ResponseDist").get_parameter_value().double_value
        self.declare_parameter("linear", 0.3)
        self.linear = self.get_parameter("linear").get_parameter_value().double_value
        self.declare_parameter("angular", 1.0)
        self.angular = self.get_parameter("angular").get_parameter_value().double_value
        self.declare_parameter("LaserAngle",90.0)
        self.LaserAngle = self.get_parameter('LaserAngle').get_parameter_value().double_value

        #create timer
        self.timer = self.create_timer(0.01, self.on_timer)

    def on_timer(self):
        self.Switch = self.get_parameter("Switch").get_parameter_value().bool_value
        self.ResponseDist = self.get_parameter("ResponseDist").get_parameter_value().double_value
        self.linear = self.get_parameter("linear").get_parameter_value().double_value
        self.angular = self.get_parameter("angular").get_parameter_value().double_value
        self.LaserAngle = self.get_parameter("LaserAngle").get_parameter_value().double_value

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data
    
    def registerScan(self, scan_data):
        if not isinstance(scan_data, LaserScan): return
        if self.Joy_active == True or self.Switch == False:
            if self.Moving == True:
                self.pub_vel.publish(Twist())
                self.Moving = not self.Moving
            return
        
        self.Moving = True

        frontDistList = []
        frontDistIDList = []
        minDistList = []
        minDistIDList = []
        ranges = np.array(scan_data.ranges)
        for i in range(len(ranges)):
            angle = (scan_data.angle_min + scan_data.angle_increment * i) * 180 / pi
            if angle > 180: angle = angle - 360
            if abs(angle) < self.priorityAngle:
                if 0 < ranges[i] < self.ResponseDist + self.offset:
                    frontDistList.append(ranges[i])
                    frontDistIDList.append(angle)
            elif abs(angle) < self.LaserAngle and ranges[i] > 0:
                minDistList.append(ranges[i])
                minDistIDList.append(angle)

        if len(frontDistIDList) != 0:
            minDist = min(frontDistList) 
            minDistID = frontDistIDList[frontDistList.index(minDist)]
        else:
            minDist = min(minDistList)
            minDistID = minDistIDList[minDistList.index(minDist)]

        velocity = Twist()
        if abs(minDist - self.ResponseDist) < 0.1: 
            velocity.linear.x = 0.0
        else:
            velocity.linear.x = -self.lin_pid.pid_compute(self.ResponseDist, minDist)
        
        angle_pid_compute = self.ang_pid.pid_compute(minDistID / 72, 0)
        if abs(angle_pid_compute) < 0.02:
            velocity.angular.z = 0.0
        else:
            velocity.angular.z = angle_pid_compute
        self.pub_vel.publish(velocity)


class SinglePID:
    def __init__(self, P=0.1, I=0.0, D=0.0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        print("Init PID: ", P, I, D)
        self.pid_reset()

    def set_pid(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        print("Set PID: ", P, I, D)
        self.pid_reset()

    def pid_compute(self, target, current):
        self.error = target - current
        self.integral += self.error
        self.differential = self.error - self.prevError
        self.prevError = self.error
        return self.Kp * self.error + self.Ki * self.integral + self.Kd * self.differential

    def pid_reset(self):
        self.error = 0
        self.integral = 0
        self.differential = 0
        self.prevError = 0


def main():
    rclpy.init()
    laser_tracker = laserTracker("laser_tracker_node")
    print("Start it.")
    rclpy.spin(laser_tracker)

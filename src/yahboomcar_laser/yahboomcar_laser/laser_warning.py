#ros lib
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

#common lib
import numpy as np
from math import pi
from std_msgs.msg import Bool


class laserWarning(Node):
    def __init__(self, name):
        super().__init__(name)

        #create sub
        self.sub_laser = self.create_subscription(LaserScan, "/scan", self.registerScan, 1)
        self.sub_JoyState = self.create_subscription(Bool, "/JoyState", self.JoyStateCallback, 1)
        #create pub
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 1)
        self.pub_Buzzer = self.create_publisher(Bool, "/Buzzer", 1)

        #declare params
        self.right_warning = 0
        self.left_warning = 0
        self.front_warning = 0
        self.Buzzer_state = False
        self.Joy_active = False
        self.Moving = False
        self.ang_pid = SinglePID(3.0, 0.0, 5.0)

        self.declare_parameter("Switch", False)
        self.Switch = self.get_parameter("Switch").get_parameter_value().bool_value
        self.declare_parameter("ResponseDist", 0.8)
        self.ResponseDist = self.get_parameter("ResponseDist").get_parameter_value().double_value
        self.declare_parameter("linear", 0.4)
        self.linear = self.get_parameter("linear").get_parameter_value().double_value
        self.declare_parameter("angular", 1.0)
        self.angular = self.get_parameter("angular").get_parameter_value().double_value
        self.declare_parameter("LaserAngle",60.0)
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
        
        minDistList = []
        minDistIDList = []
        ranges = np.array(scan_data.ranges)
        for i in range(len(ranges)):
            angle = (scan_data.angle_min + scan_data.angle_increment * i) * 180 / pi
            if angle > 180: angle = angle - 360
            if abs(angle) < self.LaserAngle and ranges[i] > 0:
                minDistList.append(ranges[i])
                minDistIDList.append(angle)
        if len(minDistIDList) != 0:
            minDist = min(minDistList)
            minDistID = minDistIDList[minDistList.index(minDist)]
        else:
            return

        velocity = Twist()
        angle_pid_compute = self.ang_pid.pid_compute(minDistID / 36, 0)
        if abs(angle_pid_compute) < 0.02:
            velocity.angular.z = 0.0
        else:
            velocity.angular.z = angle_pid_compute
        self.pub_vel.publish(velocity)

        if minDist <= self.ResponseDist: 
            if self.Buzzer_state == False:
                b = Bool()
                b.data = True
                self.pub_Buzzer.publish(b)
                self.Buzzer_state = True
        else:
            if self.Buzzer_state == True:
                self.pub_Buzzer.publish(Bool())
                self.Buzzer_state = False
        

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
    laser_warning = laserWarning("laser_warning_node")
    print("Start it.")
    rclpy.spin(laser_warning)

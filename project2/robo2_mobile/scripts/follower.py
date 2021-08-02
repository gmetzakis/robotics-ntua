#!/usr/bin/env python3


"""
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# Ros handlers services and messages
import rospy, roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
#Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv
import numpy as np
import time as t

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# from tf.transformations import euler_from_quaternion
# from tf.transformations import quaternion_matrix
# matrix = quaternion_matrix([1, 0, 0, 0])

def quaternion_to_euler(w, x, y, z):
    """Converts quaternions with components w, x, y, z into a tuple (roll, pitch, yaw)"""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class mymobibot_follower():
    """Class to compute and publish joints positions"""
    def __init__(self,rate):

        # linear and angular velocity
        self.velocity = Twist()
        # joints' states
        self.joint_states = JointState()
        # Sensors
        self.imu = Imu()
        self.imu_yaw = 0.0 # (-pi, pi]
        self.sonar_F = Range()
        self.sonar_FL = Range()
        self.sonar_FR = Range()
        self.sonar_L = Range()
        self.sonar_R = Range()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.velocity_pub = rospy.Publisher('/mymobibot/cmd_vel', Twist, queue_size=1)
        self.joint_states_sub = rospy.Subscriber('/mymobibot/joint_states', JointState, self.joint_states_callback, queue_size=1)
        # Sensors
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback, queue_size=1)
        self.sonar_front_sub = rospy.Subscriber('/sensor/sonar_F', Range, self.sonar_front_callback, queue_size=1)
        self.sonar_frontleft_sub = rospy.Subscriber('/sensor/sonar_FL', Range, self.sonar_frontleft_callback, queue_size=1)
        self.sonar_frontright_sub = rospy.Subscriber('/sensor/sonar_FR', Range, self.sonar_frontright_callback, queue_size=1)
        self.sonar_left_sub = rospy.Subscriber('/sensor/sonar_L', Range, self.sonar_left_callback, queue_size=1)
        self.sonar_right_sub = rospy.Subscriber('/sensor/sonar_R', Range, self.sonar_right_callback, queue_size=1)

        #Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    #SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of the left wheel is stored in :: self.joint_states.position[0])
        # (e.g. the angular velocity of the right wheel is stored in :: self.joint_states.velocity[1])  

    def imu_callback(self, msg):
        # ROS callback to get the /imu

        self.imu = msg
        # (e.g. the orientation of the robot wrt the global frome is stored in :: self.imu.orientation)
        # (e.g. the angular velocity of the robot wrt its frome is stored in :: self.imu.angular_velocity)
        # (e.g. the linear acceleration of the robot wrt its frome is stored in :: self.imu.linear_acceleration)

        #quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        #(roll, pitch, self.imu_yaw) = euler_from_quaternion(quaternion)
        (roll, pitch, self.imu_yaw) = quaternion_to_euler(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        # ROS callback to get the /sensor/sonar_F

        self.sonar_F = msg
        # (e.g. the distance from sonar_front to an obstacle is stored in :: self.sonar_F.range)

    def sonar_frontleft_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FL

        self.sonar_FL = msg
        # (e.g. the distance from sonar_frontleft to an obstacle is stored in :: self.sonar_FL.range)

    def sonar_frontright_callback(self, msg):
        # ROS callback to get the /sensor/sonar_FR

        self.sonar_FR = msg
        # (e.g. the distance from sonar_frontright to an obstacle is stored in :: self.sonar_FR.range)

    def sonar_left_callback(self, msg):
        # ROS callback to get the /sensor/sonar_L

        self.sonar_L = msg
        # (e.g. the distance from sonar_left to an obstacle is stored in :: self.sonar_L.range)

    def sonar_right_callback(self, msg):
        # ROS callback to get the /sensor/sonar_R

        self.sonar_R = msg
        # (e.g. the distance from sonar_right to an obstacle is stored in :: self.sonar_R.range)

    def publish(self):

        # set configuration
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime()
        time_zero = rostime_now.to_nsec()
        time_now = time_zero

        ### flags for loops etc.
        flag = 1
        error = 0
        first_time = 1
        

        ## only to add data for plotting, uncomment for plotting
        ''' 
        lin_vel = []
        ang_vel = []
        time1 = []
        dist = []
        alfa = []
        plot_flag = 1
        plot_flag2 = 1
        time_plot = time_now
        '''

        while not rospy.is_shutdown():

            sonar_front = self.sonar_F.range # and so on...
            sonar_right = self.sonar_R.range
            sonar_front_right = self.sonar_FR.range
            sonar_front_left = self.sonar_FL.range
            sonar_left = self.sonar_L.range

            """
            INSERT YOUR MAIN CODE HERE
            self.velocity.linear.x = ...
            self.velocity.angular.z = ...
            """

            #transpose the distances from sonar_r and sonar_fr to a common point (for accurate geometry)
            real_r = sonar_right + 0.2
            real_fr = sonar_front_right + (np.sqrt(2)*0.2) - np.sqrt(2)*0.018
            a = np.arctan((real_fr*0.7071 - real_r) / (real_fr*0.7071)) #angle from wall --> 0 equals parallel to the wall

            #searching for wall
            if (((sonar_front >= 0.33) or (sonar_front_right>= 0.33)) and (first_time == 1)):
                self.velocity.linear.x = 0.4
                self.velocity.angular.z = -0.02
                print ("State: Trying to reach the wall")
            else:    #found wall
                print ("State: Wall Following")
                first_time = 0 #so that it never gets in previous state (searching for wall)
                if (sonar_right < sonar_front+0.1 and sonar_front_right<sonar_front+0.1 and sonar_front_left >= 0.6): #enters PD
                    prev_error = error
                    error = 0.5 -np.cos(a)*real_r #compute error
                    self.velocity.angular.z = -14.0*error - 16.0*(error - prev_error)
                    self.velocity.linear.x = 0.2 
                    print("PD")
                else: #turn
                    self.velocity.angular.z = -0.4
                    self.velocity.linear.x = 0.1 
                    print("Turn")

            #printing some interesting values
            print("Sonar_right:", sonar_right, "m")
            print("Sonar_front:", sonar_front, "m") 
            print("Sonar_front_right:", sonar_front_right, "m")
            print("Sonar_front_left:", sonar_front_left, "m")
            print("a:", a, "rad")
            print("Time:", (time_now - time_plot)/1e9, "sec")
            print("\n")

            # Calculate time interval (in case is needed)
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9

            # Publish the new joint's angular positions
            self.velocity_pub.publish(self.velocity)


            '''
            #PLOTTING
            if (first_time == 0):
                if (plot_flag == 1):
                    plot_flag = 0
                    time_plot = time_now
                dist.append(sonar_right)
                lin_vel.append(self.velocity.linear.x)
                ang_vel.append(self.velocity.angular.z)
                time1.append((time_now - time_plot)/1e9)
                alfa.append(a)

                if ( ((time_now - time_plot)/1e9 >= 75) and (plot_flag2 == 1)):
                    plot_flag2 = 0

                    fig = plt.figure()
                    plt.plot(time1, dist)
                    plt.title('Vertical Distance from Wall')
                    plt.ylabel('m')
                    plt.xlabel('time (sec)')
                    plt.savefig('distance_full.png')

                    fig = plt.figure()
                    plt.plot(time1, lin_vel)
                    plt.title('Linear Velocity (x)')
                    plt.ylabel('m/s')
                    plt.xlabel('time (sec)')
                    plt.savefig('lin_vel_full.png')

                    fig = plt.figure()
                    plt.plot(time1, ang_vel)
                    plt.title('Angular Velocity (z)')
                    plt.ylabel('rad/s')
                    plt.xlabel('time (sec)')
                    plt.savefig('ang_vel_full.png')

                    fig = plt.figure()
                    plt.plot(time1, alfa)
                    plt.title('Angle from Wall')
                    plt.ylabel('Degrees')
                    plt.xlabel('time (sec)')
                    plt.savefig('angle_full.png')
            '''



            self.pub_rate.sleep()

    def turn_off(self):
        pass

def follower_py():
    # Starts a new node
    rospy.init_node('follower_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    follower = mymobibot_follower(rate)
    rospy.on_shutdown(follower.turn_off)
    rospy.spin()

if __name__ == '__main__':
    try:
        follower_py()
    except rospy.ROSInterruptException:
        pass

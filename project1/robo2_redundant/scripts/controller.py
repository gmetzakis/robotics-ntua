#!/usr/bin/env python3

"""
Start ROS node to publish angles for the position control of the xArm7.
"""

#cylinders_obstacles.world allaksa config tou gazebo se real-time(paizetai)

# Ros handlers services and messages
import rospy
import roslib
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
# Math imports
from math import sin, cos, atan2, pi, sqrt
from numpy.linalg import inv, det, norm, pinv, solve
import numpy as np
import time as t

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Arm parameters
# xArm7 kinematics class
from kinematics import xArm7_kinematics


class xArm7_controller():
    """Class to compute and publish joints positions"""

    def __init__(self, rate):

        # Init xArm7 kinematics handler
        self.kinematics = xArm7_kinematics()

        # joints' angular positions
        self.joint_angpos = [0, 0, 0, 0, 0, 0, 0]
        # joints' angular velocities
        self.joint_angvel = [0, 0, 0, 0, 0, 0, 0]
        # joints' states
        self.joint_states = JointState()
        # joints' transformation matrix wrt the robot's base frame
        self.A01 = self.kinematics.tf_A01(self.joint_angpos)
        self.A02 = self.kinematics.tf_A02(self.joint_angpos)
        self.A03 = self.kinematics.tf_A03(self.joint_angpos)
        self.A04 = self.kinematics.tf_A04(self.joint_angpos)
        self.A05 = self.kinematics.tf_A05(self.joint_angpos)
        self.A06 = self.kinematics.tf_A06(self.joint_angpos)
        self.A07 = self.kinematics.tf_A07(self.joint_angpos)
        # gazebo model's states
        self.model_states = ModelStates()

        # ROS SETUP
        # initialize subscribers for reading encoders and publishers for performing position control in the joint-space
        # Robot
        self.joint_states_sub = rospy.Subscriber(
            '/xarm/joint_states', JointState, self.joint_states_callback, queue_size=1)
        self.joint1_pos_pub = rospy.Publisher(
            '/xarm/joint1_position_controller/command', Float64, queue_size=1)
        self.joint2_pos_pub = rospy.Publisher(
            '/xarm/joint2_position_controller/command', Float64, queue_size=1)
        self.joint3_pos_pub = rospy.Publisher(
            '/xarm/joint3_position_controller/command', Float64, queue_size=1)
        self.joint4_pos_pub = rospy.Publisher(
            '/xarm/joint4_position_controller/command', Float64, queue_size=1)
        self.joint5_pos_pub = rospy.Publisher(
            '/xarm/joint5_position_controller/command', Float64, queue_size=1)
        self.joint6_pos_pub = rospy.Publisher(
            '/xarm/joint6_position_controller/command', Float64, queue_size=1)
        self.joint7_pos_pub = rospy.Publisher(
            '/xarm/joint7_position_controller/command', Float64, queue_size=1)
        # Obstacles
        self.model_states_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.model_states_callback, queue_size=1)

        # Publishing rate
        self.period = 1.0/rate
        self.pub_rate = rospy.Rate(rate)

        self.publish()

    # SENSING CALLBACKS
    def joint_states_callback(self, msg):
        # ROS callback to get the joint_states

        self.joint_states = msg
        # (e.g. the angular position of joint 1 is stored in :: self.joint_states.position[0])

    def model_states_callback(self, msg):
        # ROS callback to get the gazebo's model_states

        self.model_states = msg
        # (e.g. #1 the position in y-axis of GREEN obstacle's center is stored in :: self.model_states.pose[1].position.y)
        # (e.g. #2 the position in y-axis of RED obstacle's center is stored in :: self.model_states.pose[2].position.y)

    def publish(self):

        # set configuration
        
        self.joint_angpos = [0, 0.75, 0, 1.5, 0, 0.75, 0]
        tmp_rate = rospy.Rate(1)
        tmp_rate.sleep()
        self.joint4_pos_pub.publish(self.joint_angpos[3])
        tmp_rate.sleep()
        self.joint2_pos_pub.publish(self.joint_angpos[1])
        self.joint6_pos_pub.publish(self.joint_angpos[5])
        tmp_rate.sleep()

        print("The system is ready to execute your algorithm...")

        rostime_now = rospy.get_rostime() #get time
        time_zero = rostime_now.to_nsec() #set that time to the initial time of the whole task
        time_now = time_zero #the first time time_now is the initial time 
        first_time = 1 #first_time=1 means we start from y=0 and our goal is y=-0.2 (half of the original task)
        direction = 1    #if direction=1 we go from Pa to Pb(negative to positive) else if direction=0 we go from Pb to Pa(positive to negative)

        start_x = self.A07[0,3]
        start_y = self.A07[1,3]  #we take px, py, pz from the matrix A07 as the initial positions
        start_z = self.A07[2,3]


        kc0 = 50  #initial gain for the second subtask

        #desired starting points for x and z axis for the second and third phase of our trajectory
        start1_x = 0.6043
        start1_z = 0.1508   
        start2_x = 0.6043
        start2_z = 0.1508


        #initializing some variables for plotting
        '''
        q4 = []
        xdata = []
        ydata = []
        zdata = []
        velx = []
        vely = []
        velz = []
        errorx = []
        errory = []
        errorz = []
        pointL1 = []
        pointL4 = []
        time1 = []
        flag2 = 0
        '''


        while not rospy.is_shutdown():

            # Compute each transformation matrix wrt the base frame from joints' angular positions
            self.A01 = self.kinematics.tf_A01(self.joint_angpos)
            self.A02 = self.kinematics.tf_A02(self.joint_angpos)
            self.A03 = self.kinematics.tf_A03(self.joint_angpos)
            self.A04 = self.kinematics.tf_A04(self.joint_angpos)
            self.A05 = self.kinematics.tf_A05(self.joint_angpos)
            self.A06 = self.kinematics.tf_A06(self.joint_angpos)
            self.A07 = self.kinematics.tf_A07(self.joint_angpos)

            # Compute jacobian matrix
            J = self.kinematics.compute_jacobian(self.joint_angpos)

            # pseudoinverse jacobian
            pinvJ = pinv(J)

            """
            INSERT YOUR MAIN CODE HERE
            self.joint_angvel[0] = ...
            """
            ##################### 1ST SUBTASK #######################

            if (first_time==1):  #half trajectory lasts 3sec (real time)
                T1 = 1.0	 #this part moves the robot for the set configuration state to Pa
                T2 = 1.0
                T3 = 1.0

                # desired positions
                px0 = start_x               #our polyonomial trajectory consists of 3 phases 
                py0 = start_y		    
                pz0 = start_z		  
                                            #phase 1 => acceleration phase
                px1 = start1_x
                py1 = -0.065
                pz1 = start1_z
                                            #phase 2 => constant velocity 
                px2 = start2_x
                py2 = -0.135
                pz2 = start2_z
                                            #phase 3 => deceleration phase
                px3 = 0.6043
                py3 = -0.2
                pz3 = 0.1508

            elif (direction == 1):    #whole trajectory lasts 12sec (real time)
                T1 = 2.0  	      #this part moves the robot from point Pa to point Pb(6sec)
                T2 = 2.0
                T3 = 2.0
                # desired positions
                px0 = start_x
                py0 = start_y
                pz0 = start_z

                px1 = start1_x
                py1 = -0.07
                pz1 = start1_z

                px2 = start2_x
                py2 = 0.07
                pz2 = start2_z

                px3 = 0.6043
                py3 = 0.2
                pz3 = 0.1508

            elif (direction == 0):    #this part moves the robot from point Pb to point Pa(6sec)
                T1 = 2.0
                T2 = 2.0
                T3 = 2.0
                # desired positions
                px0 = start_x
                py0 = start_y
                pz0 = start_z

                px1 = start1_x
                py1 = 0.07
                pz1 = start1_z

                px2 = start2_x
                py2 = -0.07
                pz2 = start2_z

                px3 = 0.6043
                py3 = -0.2
                pz3 = 0.1508


            # desired velocities
            vx0 = 0
            vy0 = 0	#initial velocity is set to zero
            vz0 = 0

            vx1 = (px2 - px1)/T2  #calculate the velocity at the end of phase 1
            vy1 = (py2 - py1)/T2
            vz1 = (pz2 - pz1)/T2
				
            vx2 = vx1	  #constant velocity at phase 2	  
            vy2 = vy1
            vz2 = vz1

            vx3 = 0     #final velocity is set to zero as well
            vy3 = 0
            vz3 = 0

            # desired accelerations
            gx0 = 0
            gy0 = 0
            gz0 = 0

            gx1 = 0        #all desired accelerations are set to zero
            gy1 = 0
            gz1 = 0

            gx2 = 0
            gy2 = 0
            gz2 = 0

            gx3 = 0
            gy3 = 0
            gz3 = 0

    ############# POLYNOMIAL INTERPOLATION ###############

            t1 = (time_now - time_zero)/1e9
            t2 = (time_now - time_zero)/1e9 - T1
            t3 = (time_now - time_zero)/1e9 - (T1 + T2)

            #####  PHASE 1  #####

            #we are using a fifth degree polyonomial 

            A1 = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [1, T1, T1**2, T1**3, T1**4, T1**5],
                            [0, 1, 2*T1, 3*(T1**2), 4*(T1**3), 5*(T1**4)],
                            [0, 0, 2,  6*T1, 12*(T1**2), 20*(T1**3)]])

            B1x = np.array([px0,    vx0,   gx0,     px1,     vx1,    gx1])
            B1y = np.array([py0,    vy0,   gy0,     py1,     vy1,    gy1])
            B1z = np.array([pz0,    vz0,   gz0,     pz1,     vz1,    gz1])

           #we are using numpy.linalg.solve to get the constant values of the polyonomial

            S1x = solve(A1, B1x)
            S1y = solve(A1, B1y)	 						
            S1z = solve(A1, B1z)

            ax = np.empty_like(S1x)
            ay = np.empty_like(S1y)
            az = np.empty_like(S1z)

            for i in range(0, 6):
                ax[i] = S1x[i]
                ay[i] = S1y[i]
                az[i] = S1z[i]

            #we use the constants and the t1 to calculate the positions and velocities for phase 1
            pdx1 = ax[0] + ax[1]*t1 + ax[2] * \
                (t1**2) + ax[3]*(t1**3) + ax[4]*(t1**4) + ax[5]*(t1**5)
            pdy1 = ay[0] + ay[1]*t1 + ay[2] * \
                (t1**2) + ay[3]*(t1**3) + ay[4]*(t1**4) + ay[5]*(t1**5)
            pdz1 = az[0] + az[1]*t1 + az[2] * \
                (t1**2) + az[3]*(t1**3) + az[4]*(t1**4) + az[5]*(t1**5)

            vdx1 = ax[1] + 2*ax[2]*t1 + 3*ax[3] * \
                (t1**2) + 4*ax[4]*(t1**3) + 5*ax[5]*(t1**4)
            vdy1 = ay[1] + 2*ay[2]*t1 + 3*ay[3] * \
                (t1**2) + 4*ay[4]*(t1**3) + 5*ay[5]*(t1**4)
            vdz1 = az[1] + 2*az[2]*t1 + 3*az[3] * \
                (t1**2) + 4*az[4]*(t1**3) + 5*az[5]*(t1**4)

            #####  PHASE 2  #####

            pdx2 = px1 + vx1*t2  
            pdy2 = py1 + vy1*t2
            pdz2 = pz1 + vz1*t2

            vdx2 = vx1   #constant velocities
            vdy2 = vy1
            vdz2 = vz1


            #####  PHASE 3  #####

            #same procedure as phase 1

            A2 = np.matrix([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [1, T3, T3**2, T3**3, T3**4, T3**5],
                            [0, 1, 2*T3, 3*(T3**2), 4*(T3**3), 5*(T3**4)],
                            [0, 0, 2,  6*T3, 12*(T3**2), 20*(T3**3)]])

            B2x = np.array([px2,    vx2,   gx2,     px3,     vx3,    gx3])
            B2y = np.array([py2,    vy2,   gy2,     py3,     vy3,    gy3])
            B2z = np.array([pz2,    vz2,   gz2,     pz3,     vz3,    gz3])

            S2x = solve(A2, B2x)
            S2y = solve(A2, B2y)
            S2z = solve(A2, B2z)

            bx = np.empty_like(S2x)
            by = np.empty_like(S2y)
            bz = np.empty_like(S2z)

            for i in range(0, 6):
                bx[i] = S2x[i]
                by[i] = S2y[i]
                bz[i] = S2z[i]

            pdx3 = bx[0] + bx[1]*t3 + bx[2] * \
                (t3**2) + bx[3]*(t3**3) + bx[4]*(t3**4) + bx[5]*(t3**5)
            pdy3 = by[0] + by[1]*t3 + by[2] * \
                (t3**2) + by[3]*(t3**3) + by[4]*(t3**4) + by[5]*(t3**5)
            pdz3 = bz[0] + bz[1]*t3 + bz[2] * \
                (t3**2) + bz[3]*(t3**3) + bz[4]*(t3**4) + bz[5]*(t3**5)

            vdx3 = bx[1] + 2*bx[2]*t3 + 3*bx[3] * \
                (t3**2) + 4*bx[4]*(t3**3) + 5*bx[5]*(t3**4)
            vdy3 = by[1] + 2*by[2]*t3 + 3*by[3] * \
                (t3**2) + 4*by[4]*(t3**3) + 5*by[5]*(t3**4)
            vdz3 = bz[1] + 2*bz[2]*t3 + 3*bz[3] * \
                (t3**2) + 4*bz[4]*(t3**3) + 5*bz[5]*(t3**4)
            
            p = np.zeros(3)
            v = np.zeros(3)

            #according to the current time we fill the arrays p and v with the calculated positions and velocities of the current phase
            
            if ((time_now - time_zero)/1e9 < T1):

                start_x = self.A07[0,3]                
                start_z = self.A07[2,3]

                p[0] = pdx1
                p[1] = pdy1
                p[2] = pdz1
                v[0] = vdx1
                v[1] = vdy1
                v[2] = vdz1

            elif ((time_now - time_zero)/1e9 < T2 + T1):

                start1_x = self.A07[0,3]
                start1_z = self.A07[2,3]

                p[0] = pdx2
                p[1] = pdy2
                p[2] = pdz2
                v[0] = vdx2
                v[1] = vdy2
                v[2] = vdz2


            elif ((time_now - time_zero)/1e9 < T1 + T2 + T3):

                start2_x = self.A07[0,3]
                start2_z = self.A07[2,3]

                p[0] = pdx3
                p[1] = pdy3
                p[2] = pdz3
                v[0] = vdx3
                v[1] = vdy3
                v[2] = vdz3

            
            p_tonos_0 = np.array([v[0], v[1], v[2]])
            p_tonos = np.reshape(p_tonos_0, (3, 1))

            #first part of the desired equation is stored at self.joint_angvel_0

            self.joint_angvel_0 = pinvJ*p_tonos


            ##################### 2ND SUBTASK #######################

            self.model_states_callback(self.model_states)
            self.joint_states_callback(self.joint_states) 

            #the function ksis take the real positions of the robot's joints and the positions of the centers of the obstacles and returns the matrix ksss which contains all the partial derivatives of the function responsible for the obstacle avoidance over the angles of the joints
            #for more info about ksis check kinematics.py

            self.ksss = self.kinematics.compute_ksis(self.joint_states.position, self.model_states.pose[1].position.x, self.model_states.pose[1].position.y, self.model_states.pose[2].position.x, self.model_states.pose[2].position.y)

	
            I7 = np.identity(7)
            kc = np.identity(7)*kc0/5  #the gain for the second subtask is set to 10

            #now self.joint_angvel_0 contains the sum of the function for the first subtask and the function for the second subtask
            self.joint_angvel_0 = self.joint_angvel_0 + (I7 - pinvJ*J)*kc*self.ksss

            self.joint_angvel = np.zeros(7)
            for i in range(0,7):
                self.joint_angvel[i] = self.joint_angvel_0[i,0]
            
            # Convertion to angular position after integrating the angular speed in time
            # Calculate time interval
            time_prev = time_now
            rostime_now = rospy.get_rostime()
            time_now = rostime_now.to_nsec()
            dt = (time_now - time_prev)/1e9 

            # Integration
           
            for i in range(0,7):
                self.joint_angpos[i] = self.joint_angpos[i] + dt*self.joint_angvel[i]
              
            # Publish the new joint's angular positions
            
            self.joint1_pos_pub.publish(self.joint_angpos[0])
            self.joint2_pos_pub.publish(self.joint_angpos[1])
            self.joint3_pos_pub.publish(self.joint_angpos[2])
            self.joint4_pos_pub.publish(self.joint_angpos[3])
            self.joint5_pos_pub.publish(self.joint_angpos[4])
            self.joint6_pos_pub.publish(self.joint_angpos[5])
            self.joint7_pos_pub.publish(self.joint_angpos[6])

            #self.A07 = self.kinematics.tf_A07(self.joint_angpos)
	
            #we conmpute the A07 again with the new published angles and we change the direction of the trajectory when y is really close to one of the 2 points Pa, Pb

            if (first_time==1 and self.A07[1,3] < -0.1995):
                first_time = 0
                start_x = self.A07[0,3]
                start_y = self.A07[1,3]
                start_z = self.A07[2,3]

                start1_x = 0.6043
                start1_z = 0.1508
                start2_x = 0.6043
                start2_z = 0.1508

                time_zero = time_now
                time_test = time_zero
                #flag2 = 1            #only for plotting

            elif (first_time == 0):
                if (self.A07[1,3] > 0.1995 and direction == 1):
                    direction = 0
                    start_x = self.A07[0,3]
                    start_y = self.A07[1,3]
                    start_z = self.A07[2,3]
                    start1_x = 0.6043
                    start1_z = 0.1508
                    start2_x = 0.6043
                    start2_z = 0.1508

                    time_zero = time_now
                elif (self.A07[1,3] < -0.1995 and direction == 0):
                    direction = 1
                    start_x = self.A07[0,3]
                    start_y = self.A07[1,3]
                    start_z = self.A07[2,3]
                    start1_x = 0.6043
                    start1_z = 0.1508
                    start2_x = 0.6043
                    start2_z = 0.1508

                    time_zero = time_now
                    #if (flag2 == 1):           #only for plotting
                        #flag2 = 2

            print(self.A07) #printing A07 helps with debugging


            ############# PLOTTING #############
            #We use some flags to plot some interesting figures during first period

            '''
            if (flag2 == 1):
                J = self.kinematics.compute_jacobian(self.joint_angpos)
                self.L1 = self.kinematics.tf_L1(self.joint_angpos)
                self.L4 = self.kinematics.tf_L4(self.joint_angpos)
                q4.append(self.joint_angpos[3])
                zdata.append(self.A07[2,3])
                ydata.append(self.A07[1,3])
                xdata.append(self.A07[0,3])
                vel = J*self.joint_angvel_0
                velx.append(vel[0,0])
                vely.append(vel[1,0])
                velz.append(vel[2,0])
                errorx.append(abs(p[0] - self.A07[0,3]))
                errory.append(abs(p[1] - self.A07[1,3]))
                errorz.append(abs(p[2] - self.A07[2,3]))
                pointL1.append(np.sqrt( (self.L1[1,3] - (self.model_states.pose[1].position.y + 0.05))**2 + (self.L1[0,3] - (self.model_states.pose[1].position.x))**2 ))
                pointL4.append(np.sqrt( (self.L4[1,3] - (self.model_states.pose[2].position.y - 0.05))**2 + (self.L4[0,3] - (self.model_states.pose[2].position.x))**2 ))

                time1.append((time_now - time_test)/1e9)
            elif (flag2==2):
                fig = plt.figure()
                plt.plot(time1, q4)
                plt.title('q4 (Obstacles set to initial configuration)')
                plt.ylabel('rad')
                plt.xlabel('time (sec)')
                plt.savefig('q4.png')


                fig = plt.figure()
                #ax=plt.axes(projection='2d')
                plt.plot(time1, xdata)
                plt.title('Px (Obstacles set to initial configuration)')
                plt.ylabel('position (m)')
                plt.xlabel('time (sec)')
                plt.yticks(np.arange(0.603, 0.605, step=0.0001))
                #plt.savefig('x2_0.png')

                fig = plt.figure()
                plt.plot(time1, ydata)
                plt.title('Py (Obstacles set to initial configuration)')
                plt.ylabel('position (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('y2_0.png')

                fig = plt.figure()
                plt.plot(time1, zdata)
                plt.title('Pz (Obstacles set to initial configuration)')
                plt.ylabel('position (m)')
                plt.xlabel('time (sec)')
                plt.yticks(np.arange(0.150, 0.152, step=0.0001))
                #plt.savefig('z2_0.png')

                fig = plt.figure()
                plt.plot(time1, velx)
                plt.title('Vx (Obstacles set to initial configuration)')
                plt.yticks(np.arange(-0.05, 0.05, step=0.01))
                plt.ylabel('velocity (m/s)')
                plt.xlabel('time (sec)')
                #plt.savefig('velx2_0.png')

                fig = plt.figure()
                plt.plot(time1, vely)
                plt.title('Vy (Obstacles set to initial configuration)')
                plt.ylabel('velocity (m/s)')
                plt.xlabel('time (sec)')
                #plt.savefig('vely2_0.png')

                fig = plt.figure()
                plt.plot(time1, velz)
                plt.title('Vz (Obstacles set to initial configuration)')
                plt.yticks(np.arange(-0.05, 0.05, step=0.01))
                plt.ylabel('velocity (m/s)')
                plt.xlabel('time (sec)')
                #plt.savefig('velz2_0.png')

                fig = plt.figure()
                plt.plot(time1, errorx)
                plt.title('Error in Px (Obstacles set to initial configuration)')
                plt.yticks(np.arange(-0.005, 0.005, step=0.001))
                plt.ylabel('error (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('errorx_0.png')

                fig = plt.figure()
                plt.plot(time1, errory)
                plt.title('Error in Py (Obstacles set to initial configuration)')
                plt.yticks(np.arange(-0.05, 0.05, step=0.01))
                plt.ylabel('error (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('errory_0.png')

                fig = plt.figure()
                plt.plot(time1, errorz)
                plt.title('Error in Pz (Obstacles set to initial configuration)')
                plt.yticks(np.arange(-0.005, 0.005, step=0.001))
                plt.ylabel('error (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('errorz_0.png')

                fig = plt.figure()
                plt.plot(time1, pointL1)
                plt.title('Distance of point1 from green obstacle (Red obstacle moving towards robot)')
                #plt.yticks(np.arange(-0.05, 0.05, step=0.01))
                plt.ylabel('distance (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('point1_redmoving.png')

                fig = plt.figure()
                plt.plot(time1, pointL4)
                plt.title('Distance of point4 from red obstacle (Red obstacle moving towards robot)')
                #plt.yticks(np.arange(-0.05, 0.05, step=0.01))
                plt.ylabel('distance (m)')
                plt.xlabel('time (sec)')
                #plt.savefig('point4_redmoving.png')

                flag2=0

            '''

            self.pub_rate.sleep()

    def turn_off(self):
        pass


def controller_py():
    # Starts a new node
    rospy.init_node('controller_node', anonymous=True)
    # Reading parameters set in launch file
    rate = rospy.get_param("/rate")

    controller = xArm7_controller(rate)
    rospy.on_shutdown(controller.turn_off)
    rospy.spin()


if __name__ == '__main__':
    try:
        controller_py()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
from math import pi, sqrt, atan2, cos, sin
from turtle import position
import numpy as np
from numpy import NaN
import rospy
import tf
from std_msgs.msg import Empty, Float32
from nav_msgs.msg import Odometry
from mav_msgs.msg import Actuators
from geometry_msgs.msg import Twist, Pose2D
import pickle
import os
class Quadrotor():
    def __init__(self): 
        # publisher for rotor speeds
        self.motor_speed_pub = rospy.Publisher("/crazyflie2/command/motor_speed", Actuators, queue_size=10)
        
        # subscribe to Odometry topic
        self.odom_sub = rospy.Subscriber("/crazyflie2/ground_truth/odometry",Odometry, self.odom_callback)
       
        self.t0 = None
        self.t = None
        self.t_series = []
        self.x_series = []
        self.y_series = []
        self.z_series = []
        self.mutex_lock_on = False
        rospy.on_shutdown(self.save_data)

        # TODO: include initialization codes if needed
        # Physical parameters of the Crazyflie 2.0 hardware platform
        self.m = 27e-3
        self.l = 46e-3
        self.Ix = 16.571710e-6
        self.Iy = 16.571710e-6
        self.Iz = 29.261652e-6
        self.Ip = 12.65625e-8
        self.kF = 1.28192e-8
        self.kM = 5.964552e-3
        self.w_max = 2618
        self.w_min = 0
        self.g = 9.81
        self.allocation = np.array([
                                    [1/(4*self.kF), -sqrt(2)/(4*self.kF*self.l), -sqrt(2)/(4*self.kF*self.l), -1/(4*self.kF*self.kM)],
                                    [1/(4*self.kF), -sqrt(2)/(4*self.kF*self.l), sqrt(2)/(4*self.kF*self.l), 1/(4*self.kF*self.kM)],
                                    [1/(4*self.kF), sqrt(2)/(4*self.kF*self.l), sqrt(2)/(4*self.kF*self.l), -1/(4*self.kF*self.kM)],
                                    [1/(4*self.kF), sqrt(2)/(4*self.kF*self.l), -sqrt(2)/(4*self.kF*self.l), 1/(4*self.kF*self.kM)]
                                    ])
        self.Omega = 0

        # Tuning Parameters
        self.kp = 90 
        self.kd = 10    
                
        # Sliding Mode Gainsee
        self.lambda_u1 = 3 
        self.K_u1 = 10 
        self.boundary_u1 = 0.9 

        self.lambda_u2 = 12 
        self.K_u2 = 90 
        self.boundary_u2 = 0.5
        
        self.lambda_u3 = 12
        self.K_u3 = 90
        self.boundary_u3 = 0.5
        
        self.lambda_u4 = 3
        self.K_u4 = 5
        self.boundary_u4 = 0.9

        # Determing coefficients for Trajectory Equations

        self.time = [0, 5, 20, 35, 50, 65]
        self.i = 0
        self.x_way_points = np.array([0, 0, 1, 1, 0, 0])
        self.y_way_points = np.array([0, 0, 0, 1, 1, 0])
        self.z_way_points = np.array([0, 1, 1, 1, 1, 1])
        self.x_coeff = np.zeros((5,6))
        self.y_coeff = np.zeros((5,6))
        self.z_coeff = np.zeros((5,6))

        for i in range(len(self.time)-1):
            self.x_coeff[i] = self.traj(self.time[i], self.time[i+1], self.x_way_points[i], self.x_way_points[i+1])
            self.y_coeff[i] = self.traj(self.time[i], self.time[i+1], self.y_way_points[i], self.y_way_points[i+1])
            self.z_coeff[i] = self.traj(self.time[i], self.time[i+1], self.z_way_points[i], self.z_way_points[i+1])
        
    
    def WraptoPi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def sat(self, S,boundary):
        return min(max(S/boundary, -1), 1)

    def traj(self, t0, tf, q0, qf):
        A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                    [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                    [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                    [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
        B = np.array([q0, 0, 0, qf, 0 , 0])

        coeff = np.dot(np.linalg.inv(A),B)
        return coeff
    
    def traj_evaluate(self):
        # TODO: evaluating the corresponding trajectories designed in Part 1 to return the desired positions, velocities and accelerations

        if self.t > self.time[self.i+1]:
            self.i+=1

        xd = self.x_coeff[self.i][0] + self.x_coeff[self.i][1]*self.t + self.x_coeff[self.i][2]*self.t**2 + self.x_coeff[self.i][3]*self.t**3 + self.x_coeff[self.i][4]*self.t**4 + self.x_coeff[self.i][5]*self.t**5
        xd_dot = self.x_coeff[self.i][1] + 2 * self.x_coeff[self.i][2]*self.t + 3 * self.x_coeff[self.i][3]*self.t**2 + 4 * self.x_coeff[self.i][4]*self.t**3 + 5 * self.x_coeff[self.i][5]*self.t**4
        xd_ddot =  2 * self.x_coeff[self.i][2]+ 6 * self.x_coeff[self.i][3]*self.t + 12 * self.x_coeff[self.i][4]*self.t**2 + 20 * self.x_coeff[self.i][5]*self.t**3 

        yd = self.y_coeff[self.i][0] + self.y_coeff[self.i][1]*self.t + self.y_coeff[self.i][2]*self.t**2 + self.y_coeff[self.i][3]*self.t**3 + self.y_coeff[self.i][4]*self.t**4 + self.y_coeff[self.i][5]*self.t**5
        yd_dot = self.y_coeff[self.i][1] + 2 * self.y_coeff[self.i][2]*self.t + 3 * self.y_coeff[self.i][3]*self.t**2 + 4 * self.y_coeff[self.i][4]*self.t**3 + 5 * self.y_coeff[self.i][5]*self.t**4
        yd_ddot =  2 * self.y_coeff[self.i][2] + 6 * self.y_coeff[self.i][3]*self.t + 12 * self.y_coeff[self.i][4]*self.t**2 + 20 * self.y_coeff[self.i][5]*self.t**3 

        zd = self.z_coeff[self.i][0] + self.z_coeff[self.i][1]*self.t + self.z_coeff[self.i][2]*self.t**2 + self.z_coeff[self.i][3]*self.t**3 + self.z_coeff[self.i][4]*self.t**4 + self.z_coeff[self.i][5]*self.t**5
        zd_dot = self.z_coeff[self.i][1] + 2 * self.z_coeff[self.i][2]*self.t + 3 * self.z_coeff[self.i][3]*self.t**2 + 4 * self.z_coeff[self.i][4]*self.t**3 + 5 * self.z_coeff[self.i][5]*self.t**4
        zd_ddot =  2 * self.z_coeff[self.i][2] + 6 * self.z_coeff[self.i][3]*self.t + 12 * self.z_coeff[self.i][4]*self.t**2 + 20 * self.z_coeff[self.i][5]*self.t**3 
        
        return xd, xd_dot, xd_ddot, yd, yd_dot, yd_ddot, zd, zd_dot, zd_ddot

    def smc_control(self, xyz, xyz_dot, rpy, rpy_dot):
        # obtain the desired values by evaluating the corresponding trajectories

        xd, xd_dot, xd_ddot, yd, yd_dot, yd_ddot, zd, zd_dot, zd_ddot = self.traj_evaluate()
        x = xyz[0]
        x_dot = xyz_dot[0]

        y = xyz[1]
        y_dot = xyz_dot[1]

        z = xyz[2]
        z_dot = xyz_dot[2]

        phi = rpy[0]
        phi_dot = rpy_dot[0]
        
        theta = rpy[1]
        theta_dot = rpy_dot[1]
        
        psi = rpy[2]
        psi_dot = rpy_dot[2]

        # TODO: implement the Sliding Mode Control laws designed in Part 2 to calculate the control inputs "u"
        # REMARK: wrap the roll-pitch-yaw angle errors to [-pi to pi]

        # Sliding Mode Design for u1
        ez = z - zd
        ez_dot = z_dot - zd_dot

        Sz = ez_dot + self.lambda_u1 * ez

        u1 = (-self.K_u1 * self.sat(Sz, self.boundary_u1) + zd_ddot - self.lambda_u1 * ez_dot + self.g)/(cos(phi)*cos(theta)/self.m)
        

        Fx = self.m*( -self.kp*(x - xd) - self.kd*(x_dot - xd_dot) + xd_ddot)
        Fy = self.m*( -self.kp*(y - yd) - self.kd*(y_dot - yd_dot) + yd_ddot)

        thetad = np.arcsin(np.clip(Fx/u1, -1, 1))
        phid = np.arcsin(np.clip(Fy/u1, -1, 1))

        psid = 0

        phid_dot = 0
        thetad_dot = 0
        psid_dot = 0

        phid_ddot = 0
        thetad_ddot = 0
        psid_ddot = 0

        # Sliding Mode Design for phi
        ephi = self.WraptoPi(phi-phid)
        ephi_dot = phi_dot - phid_dot
        Sphi = ephi_dot + self.lambda_u2*ephi
        u2 = -self.Ix*self.K_u2*self.sat(Sphi, self.boundary_u2) - theta_dot*psi_dot*(self.Iy - self.Iz) + self.Ip*self.Omega*theta_dot + phid_ddot*self.Ix - self.lambda_u2*ephi_dot*self.Ix

        # Sliding Mode Design for theta
        etheta = self.WraptoPi(theta-thetad)
        etheta_dot = theta_dot - thetad_dot
        Stheta = etheta_dot + self.lambda_u3*etheta
        u3 = -self.Iy*self.K_u3*self.sat(Stheta, self.boundary_u3) - phi_dot*psi_dot*(self.Iz - self.Ix) - self.Ip*self.Omega*phi_dot + thetad_ddot*self.Iy - self.lambda_u3*etheta_dot*self.Iy

        # Sliding Mode Design for psi
        epsi = self.WraptoPi(psi-psid)
        epsi_dot = psi_dot - psid_dot
        Spsi = epsi_dot + self.lambda_u4*epsi
        u4 = -self.Iz*self.K_u4*self.sat(Spsi, self.boundary_u4) - phi_dot*theta_dot*(self.Ix - self.Iy) - self.lambda_u4*epsi_dot*self.Iz + psid_ddot*self.Iz
        

        # TODO: convert the desired control inputs "u" to desired rotor velocities "motor_vel" by using the "allocation matrix"
        U = np.array([u1, u2, u3, u4])
        temp = np.dot(self.allocation, U)
        temp = np.clip(temp, self.w_min**2, self.w_max**2)
        motor_vel = np.sqrt(temp)
        
        
        # TODO: maintain the rotor velocities within the valid range of [0 to 2618]    
        # publish the motor velocities to the associated ROS topic
        motor_speed = Actuators()
        motor_speed.angular_velocities = [motor_vel[0,0], motor_vel[1,0], motor_vel[2,0], motor_vel[3,0]]
        self.Omega = motor_vel[0,0] - motor_vel[1,0] + motor_vel[2,0] - motor_vel[3,0]
        self.motor_speed_pub.publish(motor_speed)
        
    # odometry callback function (DO NOT MODIFY)
    def odom_callback(self, msg):
        if self.t0 == None:
            self.t0 = msg.header.stamp.to_sec()
        self.t = msg.header.stamp.to_sec() - self.t0

        # convert odometry data to xyz, xyz_dot, rpy, and rpy_dot
        w_b = np.asarray([[msg.twist.twist.angular.x], [msg.twist.twist.angular.y], [msg.twist.twist.angular.z]])
        v_b = np.asarray([[msg.twist.twist.linear.x], [msg.twist.twist.linear.y], [msg.twist.twist.linear.z]])
        xyz = np.asarray([[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z]])
        q = msg.pose.pose.orientation
        T = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0:3, 3] = xyz[0:3, 0]
        R = T[0:3, 0:3]
        xyz_dot = np.dot(R, v_b)
        rpy = tf.transformations.euler_from_matrix(R, 'sxyz')
        rpy_dot = np.dot(np.asarray([
                                    [1, np.sin(rpy[0])*np.tan(rpy[1]), np.cos(rpy[0])*np.tan(rpy[1])],
                                    [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                                    [0, np.sin(rpy[0])/np.cos(rpy[1]), np.cos(rpy[0])/np.cos(rpy[1])]
                                    ]), w_b)

        rpy = np.expand_dims(rpy, axis=1)
        # store the actual trajectory to be visualized later
        if (self.mutex_lock_on is not True):
            self.t_series.append(self.t)
            self.x_series.append(xyz[0, 0])
            self.y_series.append(xyz[1, 0])
            self.z_series.append(xyz[2, 0])

        # call the controller with the current states
        self.smc_control(xyz, xyz_dot, rpy, rpy_dot)

    # save the actual trajectory data
    def save_data(self):
        # TODO: update the path below with the correct path
        
        with open("/home/saammmy/rbe502_project/src/log.pkl","wb") as fp:
            self.mutex_lock_on = True
            pickle.dump([self.t_series,self.x_series,self.y_series,self.z_series], fp)

if __name__ == '__main__':
    rospy.init_node("quadrotor_control")
    rospy.loginfo("Press Ctrl + C to terminate")
    whatever = Quadrotor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def traj(t0,tf,q0,qf):
    A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
    B = np.array([q0, 0, 0, qf, 0 , 0])

    coeff = np.dot(np.linalg.inv(A),B)

    return coeff

time = [0, 5, 20, 35, 50, 65]
x_way_points = np.array([0, 0, 1, 1, 0, 0])
y_way_points = np.array([0, 0, 0, 1, 1, 0])
z_way_points = np.array([0, 1, 1, 1, 1, 1])
x_coeff = np.zeros((5,6))
y_coeff = np.zeros((5,6))
z_coeff = np.zeros((5,6))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")


for i in range(len(time)-1):
    x_coeff[i] = traj(time[i], time[i+1], x_way_points[i], x_way_points[i+1])
    y_coeff[i] = traj(time[i], time[i+1], y_way_points[i], y_way_points[i+1])
    z_coeff[i] = traj(time[i], time[i+1], z_way_points[i], z_way_points[i+1])

xd = np.zeros(75)
xd_dot = np.zeros(75)
xd_ddot = np.zeros(75)
yd = np.zeros(75)
yd_dot = np.zeros(75)
yd_ddot = np.zeros(75)
zd = np.zeros(75)
zd_dot = np.zeros(75)
zd_ddot = np.zeros(75)

TIME = np.zeros(75)

for i in range(len(time)-1):
    j= i*15
    for t in np.linspace(time[i], time[i+1], 15):
        xd[j] = x_coeff[i][0] + x_coeff[i][1]*t + x_coeff[i][2]*t**2 + x_coeff[i][3]*t**3 + x_coeff[i][4]*t**4 + x_coeff[i][5]*t**5
        xd_dot[j] = x_coeff[i][1] + 2 * x_coeff[i][2]*t + 3 * x_coeff[i][3]*t**2 + 4 * x_coeff[i][4]*t**3 + 5 * x_coeff[i][5]*t**4
        xd_ddot[j] =  2 * x_coeff[i][2] + 6 * x_coeff[i][3]*t + 12 * x_coeff[i][4]*t**2 + 20 * x_coeff[i][5]*t**3 

        yd[j] = y_coeff[i][0] + y_coeff[i][1]*t + y_coeff[i][2]*t**2 + y_coeff[i][3]*t**3 + y_coeff[i][4]*t**4 + y_coeff[i][5]*t**5
        yd_dot[j] = y_coeff[i][1] + 2 * y_coeff[i][2]*t + 3 * y_coeff[i][3]*t**2 + 4 * y_coeff[i][4]*t**3 + 5 * y_coeff[i][5]*t**4
        yd_ddot[j] =  2 * y_coeff[i][2] + 6 * y_coeff[i][3]*t + 12 * y_coeff[i][4]*t**2 + 20 * y_coeff[i][5]*t**3 

        zd[j] = z_coeff[i][0] + z_coeff[i][1]*t + z_coeff[i][2]*t**2 + z_coeff[i][3]*t**3 + z_coeff[i][4]*t**4 + z_coeff[i][5]*t**5
        zd_dot[j] = z_coeff[i][1] + 2 * z_coeff[i][2]*t + 3 * z_coeff[i][3]*t**2 + 4 * z_coeff[i][4]*t**3 + 5 * z_coeff[i][5]*t**4
        zd_ddot[j] =  2 * z_coeff[i][2]+ 6 * z_coeff[i][3]*t + 12 * z_coeff[i][4]*t**2 + 20 * z_coeff[i][5]*t**3 

        TIME[j] = t

        j+=1
plt.title("Desired Trajectory for Drone")
plt.plot(xd,yd,zd,"--")
plt.figure(2)
plt.title("Desired Velocity Profile in X")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(TIME, xd_dot)
plt.figure(3)
plt.title("Desired Velocity Profile in Y")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(TIME, yd_dot)
plt.figure(4)
plt.title("Desired Velocity Profile in Z")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.plot(TIME, zd_dot)
plt.figure(5)
plt.title("Desired Acceleration Profile in X")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (ms-2)")
plt.plot(TIME, xd_ddot)
plt.figure(6)
plt.title("Desired Acceleration Profile in Y")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (ms-2)")
plt.plot(TIME, yd_ddot)
plt.figure(7)
plt.title("Desired Acceleration Profile in Z")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (ms-2)")
plt.plot(TIME, zd_ddot)
plt.show()
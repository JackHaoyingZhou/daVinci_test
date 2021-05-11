
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.polynomial as poly

def coef(b, dt):

    A = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [1.0, dt, dt**2, dt**3],
                  [0, 1.0, 2*dt, 3*dt**2]])


    return np.linalg.pinv(A).dot(b)


# seed random number generator
seed(1)
count = 200
# b = np.array([ [-0.267], [0.], [ -0.7 ], [0.0] ])
# x = coef(b, 10)
# fit = poly.Polynomial(x.flatten())
t = np.linspace(0,10,count)
#x_prime = np.cos(t)
x_prime = np.cos(t) + np.exp(0.1*t)

x_data = []
for i in range(5):
    x = x_prime + gauss(-.1, .1)
    x_data.append(x)

# b = np.array([ [0.221], [0.0], [0.5], [0.0] ])
# x = coef(b, 10)
# fit = poly.Polynomial(x.flatten())
t = np.linspace(0, 10, count)
# y_prime = np.sin(t)
y_prime = np.sin(t) - np.exp(0.1*t)
#Y Coordinate
y_data = []
for i in range(5):
    y = y_prime + gauss(-.1, .1)
    y_data.append(y)


# b = np.array([ [-0.147], [0.0], [-0.2 ], [0.0] ])
# x = coef(b, 10)
# fit = poly.Polynomial(x.flatten())
t = np.linspace(0,10,count)
# z_prime = t
z_prime = t + t**1.2 + np.exp(-t)
#Z Coordinate
z_data = []
for i in range(5):
    z = z_prime + gauss(-.1, .1)
    z_data.append(z)


trainer = TPGMMTrainer.TPGMMTrainer(demo=[x_data, y_data, z_data],
                                    file_name="spiraltest",
                                    n_rf=3,
                                    dt=0.05,
                                    reg=[1e-3],
                                    poly_degree=[3,3,3])
trainer.train()
runner = TPGMMRunner.TPGMMRunner("spiraltest")


path = runner.run()
x_coordinates = path[:,0]
y_coordinates = path[:,1]
z_coordinates = path[:,2]

x_data_arr = np.array(x_data)
y_data_arr = np.array(y_data)
z_data_arr = np.array(z_data)
# print(x_data_arr[0, :])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(x_data_arr[0, :], y_data_arr[0, :], z_data_arr[0, :], c='yellow',label='given trajectory 1')
ax.scatter3D(x_data_arr[1, :], y_data_arr[1, :], z_data_arr[1, :], c='blue',label='given trajectory 2')
ax.scatter3D(x_data_arr[2, :], y_data_arr[2, :], z_data_arr[2, :], c='green',label='given trajectory 3')
ax.scatter3D(x_data_arr[3, :], y_data_arr[3, :], z_data_arr[3, :], c='purple',label='given trajectory 4')
ax.scatter3D(x_data_arr[4, :], y_data_arr[4, :], z_data_arr[4, :], c='orange',label='given trajectory 5')
ax.scatter3D(x_coordinates, y_coordinates, z_coordinates, c='red',label='TPGMM generated trajectory')
# ax.legend(['Given Trajectory #1', 'Given Trajectory #2', 'Given Trajectory #3', 'Given Trajectory #4', 'Given Trajectory #5',  'TPGMM Genereated Trajectory'], loc="left", ncol=1)
ax.legend()
ax.set_title("Given Trajectories vs. TPGMM Generated Trajectory")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.tight_layout()
plt.show()

# fig, axs = plt.subplots(3)


# for p in knee:
#     axs[1].plot(p)
#     axs[1].plot(path[:, 1], linewidth=4, color='black')

# for p in ankle:
#     axs[2].plot(p)
#     axs[2].plot(path[:, 2], linewidth=4, color='black')

# plt.show()

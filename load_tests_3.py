
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy.polynomial.polynomial as poly
import pickle
import json

jp0 = []
jp1 = []
jp2 = []
jp3 = []
jp4 = []
jp5 = []

for i in range(1):
    with open('multi_test_{}'.format(i+1),'rb') as fp:
        itemlist = pickle.load(fp)
    itemlist = np.array(itemlist)
    jp_0 = itemlist[:,0]
    jp_1 = itemlist[:,1]
    jp_2 = itemlist[:,2]
    jp_3 = itemlist[:,3]
    jp_4 = itemlist[:,4]
    jp_5 = itemlist[:,5]
    jp0_list = 10*jp_0 #+ gauss(0, .01)
    jp1_list = 10*jp_1 #+ gauss(0, .01)
    jp2_list = 10*jp_2 #+ gauss(0, .01)
    jp3_list = 10*jp_3 #+ gauss(0, .01)
    jp4_list = 10*jp_4 #+ gauss(0, .01)
    jp5_list = 10*jp_5 #+ gauss(0, .01)
    jp0.append(jp0_list)
    jp1.append(jp1_list)
    jp2.append(jp2_list)
    jp3.append(jp3_list)
    jp4.append(jp4_list)
    jp5.append(jp5_list)


# jp0_list = jp_0 + gauss(0, .01)
# jp1_list = jp_1 + gauss(0, .01)
# jp2_list = jp_2 + gauss(0, .01)
# jp3_list = jp_3 + gauss(0, .01)
# jp4_list = jp_4 + gauss(0, .01)
# jp5_list = jp_5 + gauss(0, .01)
#
# s = [jp0_list, jp1_list, jp2_list, jp3_list, jp4_list, jp5_list]
#
# s_d = np.asarray(s)
# s_d = s_d.T
#
# s_l = s_d.tolist()

# s_d = np.zeros((2000,6))
# s_d = np.zeros((2000,6))

# s_d = np.array(s)
# seed random number generator
# seed(1)
# count = 200
# # b = np.array([ [-0.267], [0.], [ -0.7 ], [0.0] ])
# # x = coef(b, 10)
# # fit = poly.Polynomial(x.flatten())
# t = np.linspace(0,10,count)
# #x_prime = np.cos(t)
# x_prime = np.cos(t) + np.exp(0.1*t)
#
# x_data = []
# for i in range(5):
#     x = x_prime + gauss(-.1, .1)
#     x_data.append(x)
#
# # b = np.array([ [0.221], [0.0], [0.5], [0.0] ])
# # x = coef(b, 10)
# # fit = poly.Polynomial(x.flatten())
# t = np.linspace(0, 10, count)
# # y_prime = np.sin(t)
# y_prime = np.sin(t) - np.exp(0.1*t)
# #Y Coordinate
# y_data = []
# for i in range(5):
#     y = y_prime + gauss(-.1, .1)
#     y_data.append(y)
#
#
# # b = np.array([ [-0.147], [0.0], [-0.2 ], [0.0] ])
# # x = coef(b, 10)
# # fit = poly.Polynomial(x.flatten())
# t = np.linspace(0,10,count)
# # z_prime = t
# z_prime = t + t**1.2 + np.exp(-t)
# #Z Coordinate
# z_data = []
# for i in range(5):
#     z = z_prime + gauss(-.1, .1)
#     z_data.append(z)


#
# x_coordinates = path[:,0]
# y_coordinates = path[:,1]
# z_coordinates = path[:,2]
trainer = TPGMMTrainer.TPGMMTrainer(demo=[jp0,jp1,jp2,jp3,jp4,jp5],
                                    file_name="test_2",
                                    n_rf=3,
                                    dt=0.05,
                                    reg=[2e-3],
                                    poly_degree=[15,15,15,15,15,15])
trainer.train()
runner = TPGMMRunner.TPGMMRunner("test_2")


path = runner.run()

with open('result_test_2','w') as f:
    json.dump(path.tolist(),f)
# x_data_arr = np.array(x_data)
# y_data_arr = np.array(y_data)
# z_data_arr = np.array(z_data)
# # print(x_data_arr[0, :])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(x_data_arr[0, :], y_data_arr[0, :], z_data_arr[0, :], c='yellow',label='given trajectory 1')
# ax.scatter3D(x_data_arr[1, :], y_data_arr[1, :], z_data_arr[1, :], c='blue',label='given trajectory 2')
# ax.scatter3D(x_data_arr[2, :], y_data_arr[2, :], z_data_arr[2, :], c='green',label='given trajectory 3')
# ax.scatter3D(x_data_arr[3, :], y_data_arr[3, :], z_data_arr[3, :], c='purple',label='given trajectory 4')
# ax.scatter3D(x_data_arr[4, :], y_data_arr[4, :], z_data_arr[4, :], c='orange',label='given trajectory 5')
# ax.scatter3D(x_coordinates, y_coordinates, z_coordinates, c='red',label='TPGMM generated trajectory')
# # ax.legend(['Given Trajectory #1', 'Given Trajectory #2', 'Given Trajectory #3', 'Given Trajectory #4', 'Given Trajectory #5',  'TPGMM Genereated Trajectory'], loc="left", ncol=1)
# ax.legend()
# ax.set_title("Given Trajectories vs. TPGMM Generated Trajectory")
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')
# plt.tight_layout()
# plt.show()

# fig, axs = plt.subplots(3)


# for p in knee:
#     axs[1].plot(p)
#     axs[1].plot(path[:, 1], linewidth=4, color='black')

# for p in ankle:
#     axs[2].plot(p)
#     axs[2].plot(path[:, 2], linewidth=4, color='black')

# plt.show()

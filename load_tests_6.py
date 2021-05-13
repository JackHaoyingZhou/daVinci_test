
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

with open('test4','rb') as fp:
    itemlist = pickle.load(fp)

itemlist = np.array(itemlist)
jp_0 = itemlist[10:-500,0]
jp_1 = itemlist[10:-500,1]
jp_2 = itemlist[10:-500,2]
jp_3 = itemlist[10:-500,3]
jp_4 = itemlist[10:-500,4]
jp_5 = itemlist[10:-500,5]

jp0 = []
jp1 = []
jp2 = []
jp3 = []
jp4 = []
jp5 = []

for i in range(5):
    jp0_list = jp_0 + gauss(-0.01, 0.01)
    jp1_list = jp_1 + gauss(-0.01, 0.01)
    jp2_list = jp_2 + gauss(-0.01, 0.01)
    jp3_list = jp_3 + gauss(-0.01, 0.01)
    jp4_list = jp_4 + gauss(-0.01, 0.01)
    jp5_list = jp_5 + gauss(-0.01, 0.01)
    jp0.append(jp0_list)
    jp1.append(jp1_list)
    jp2.append(jp2_list)
    jp3.append(jp3_list)
    jp4.append(jp4_list)
    jp5.append(jp5_list)



# # #
trainer = TPGMMTrainer.TPGMMTrainer(demo=[jp0,jp1, jp2,jp3,jp4, jp5],
                                    file_name="real_test_2",
                                    n_rf=15,
                                    dt=0.01,
                                    reg=[1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2],
                                    poly_degree=[15, 15, 15, 15, 15, 15])
trainer.train()
runner = TPGMMRunner.TPGMMRunner("real_test_2")
#
#
path = runner.run()



fig, axs = plt.subplots(3,2)

for p in jp0:
    axs[0,0].plot(p)
    axs[0,0].plot(path[:, 0], linewidth=4, color='black')

for p in jp1:
    axs[1,0].plot(p)
    axs[1,0].plot(path[:, 1], linewidth=4, color='black')
#
for p in jp2:
    axs[2,0].plot(p)
    axs[2,0].plot(path[:, 2], linewidth=4, color='black')
#
for p in jp3:
    axs[0,1].plot(p)
    axs[0,1].plot(path[:, 3], linewidth=4, color='black')

for p in jp4:
    axs[1,1].plot(p)
    axs[1,1].plot(path[:, 4], linewidth=4, color='black')

for p in jp5:
    axs[2,1].plot(p)
    axs[2,1].plot(path[:, 5], linewidth=4, color='black')


plt.show()
# #
# with open('test4_result','w') as f:
#     json.dump(path.tolist(),f)
# x_coordinates = path[:,0]
# y_coordinates = path[:,1]
# z_coordinates = path[:,2]
#
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

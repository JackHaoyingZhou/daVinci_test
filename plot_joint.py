import numpy as np
import matplotlib.pyplot as plt
import json
import pickle


with open('multi_test_1','rb') as fp:
    jp_origins = pickle.load(fp)

jp_origins = np.asarray(jp_origins)

# with open('result_1','rb') as fp:
#     jp_values = pickle.load(fp)

jp_values = []

for i in range(6):
    with open('result1_try_{}'.format(i)) as f:
        itemlist = json.load(f)
    itemlist = np.array(itemlist)
    itemlist = np.asarray(itemlist)
    itemlist = np.reshape(itemlist,(1990,))
    jp_values.append(itemlist)

jp_values = np.asarray(jp_values)
jp_values = jp_values.T

with open('result_try','w') as f:
    json.dump(jp_values.tolist(),f)

# jp_values = jp_values.tolist()


# with open('result_test_4') as f:
#     jp_values = json.load(f)
#
# jp_origins = np.asarray(jp_origins)
# jp_values = np.asarray(jp_values)
#
# jp_values = np.reshape(jp_values,(1490,6))

# with open('result_1','w') as f:
#     json.dump(jp_values.tolist(),f)

fig = plt.figure()
# fig.suptitle('Optimal Trajectory',fontsize=20)
ax1 = fig.add_subplot(321)

ax1.plot(np.transpose(jp_origins[:,0]),'r-',label='origin')
ax1.plot(np.transpose(jp_values[:,0]),'g-',label='generated')
# ax1.plot(np.transpose(y1_follow[min_KL_index]),'b-',label='desired')
ax1.set_ylabel('Joint 1',fontsize=17)
# ax1.set_xlabel('Iteration',fontsize=15)
ax1.tick_params(axis="x",labelsize=15)
ax1.tick_params(axis="y",labelsize=15)
ax1.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
ax2 = fig.add_subplot(323)
ax2.plot(np.transpose(jp_origins[:,1]),'r-',label='origin')
ax2.plot(np.transpose(jp_values[:,1]),'g-',label='generated')
# ax2.plot(np.transpose(y2_follow[min_KL_index]),'b-',label='desired')
ax2.set_ylabel('Joint 2',fontsize=17)
# ax2.set_xlabel('Iteration',fontsize=15)
ax2.tick_params(axis="x",labelsize=15)
ax2.tick_params(axis="y",labelsize=15)
ax2.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
ax3 = fig.add_subplot(325)
ax3.plot(np.transpose(jp_origins[:,2]),'r-',label='origin')
ax3.plot(np.transpose(jp_values[:,2]),'g-',label='generated')
ax3.set_ylabel('Joint 3',fontsize=17)
ax3.set_xlabel('Iteration',fontsize=17)
ax3.tick_params(axis="x",labelsize=15)
ax3.tick_params(axis="y",labelsize=15)
ax3.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
ax4 = fig.add_subplot(322)
ax4.plot(np.transpose(jp_origins[:,3]),'r-',label='origin')
ax4.plot(np.transpose(jp_values[:,3]),'g-',label='generated')
ax4.set_ylabel('Joint 4',fontsize=17)
ax4.set_xlabel('Iteration',fontsize=15)
ax4.tick_params(axis="x",labelsize=15)
ax4.tick_params(axis="y",labelsize=15)
ax4.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
ax5 = fig.add_subplot(324)
ax5.plot(np.transpose(jp_origins[:,4]),'r-',label='origin')
ax5.plot(np.transpose(jp_values[:,4]),'g-',label='generated')
ax5.set_ylabel('Joint 5',fontsize=17)
ax5.set_xlabel('Iteration',fontsize=15)
ax5.tick_params(axis="x",labelsize=15)
ax5.tick_params(axis="y",labelsize=15)
ax5.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
ax6 = fig.add_subplot(326)
ax6.plot(np.transpose(jp_origins[:,5]),'r-',label='origin')
ax6.plot(np.transpose(jp_values[:,5]),'g-',label='generated')
ax6.set_ylabel('Joint6',fontsize=17)
ax6.set_xlabel('Iteration',fontsize=17)
ax6.tick_params(axis="x",labelsize=15)
ax6.tick_params(axis="y",labelsize=15)
ax6.legend(prop={'size':15})
#ax1.yticks(fontsize=15)
plt.show()

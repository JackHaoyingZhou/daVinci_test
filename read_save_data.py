from joint_pos_recorder import JointPosLoader
import pickle

# m,l = JointPosLoader.load_by_prefix('JP#2021-05-11 02:29')
#
# jp_values = []
#
# for i in range(len(m)):
#     for j in range(len(m[0])):
#         jp_values.append(m[i][j]['pos'])
#
# with open('test4','wb') as fp:
#     pickle.dump(jp_values,fp)

with open('test4','rb') as fp:
    itemlist = pickle.load(fp)

import pickle
import numpy as np
from tqdm import tqdm

# Linear
print('-' * 20 + 'Linear Eval' + '-' * 20)

joint_path = 'work_dir/ntu60_cs/skeletonclr_joint/semi_0.01/'   
bone_path = 'work_dir/ntu60_cs/skeletonclr_bone/semi_0.01/'  
motion_path = 'work_dir/ntu60_cs/skeletonclr_motion/semi_0.01/'

label = open('/home/zys/code/RVTCLR/data/NTU-RGB-D/xsub/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(joint_path + 'test_result.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(bone_path + 'test_result.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(motion_path + 'test_result.pkl', 'rb')
r3 = list(pickle.load(r3).items())

alpha = [1, 1, 1]

right_num = total_num = right_num_5 = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    r = r11 * alpha[0] + r33 * alpha[2] + r22 * alpha[1] 
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num
acc5 = right_num_5 / total_num
print('top1: ', acc)
print('top5: ', acc5)

# for m in np.arange(0,1,0.1):
#     for j in np.arange(0,1,0.1):
#         for k in np.arange(0,1,0.1): 

#             right_num = total_num = right_num_5 = 0
#             for i in tqdm(range(len(label[0]))):
#                 _, l = label[:, i]
#                 _, r11 = r1[i]
#                 _, r22 = r2[i]
#                 _, r33 = r3[i]
#                 r = r11 * m + r22 * j + r33 *  k 
#                 rank_5 = r.argsort()[-5:]
#                 right_num_5 += int(int(l) in rank_5)
#                 r = np.argmax(r)
#                 right_num += int(r == int(l))
#                 total_num += 1
#             acc = right_num / total_num
#             acc5 = right_num_5 / total_num
#             print('top1: ', acc)
#             #print('top5: ', acc5)           




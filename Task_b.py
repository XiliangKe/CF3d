""" 
一共两个任务,一是为了计算绘制PR图所需的数据
            二是为了计算TopN指标
            记得修改输入输出路径
"""


import numpy as np
import math
import os
import json

for i, file in enumerate(result):
  count = 0
  for j in range(N):
    if jsonfile[i] == jsonfile[file[j]]:
      count = count + 1
  ans = count / N
  sum = sum + ans
precision = sum / 2814
print(precision)

# # 任务一 计算绘制PR图像所需的数据
# feature = np.load('/userHOME/xzy/projects/kimmo/scanobject/feature_result/CF3d_hard_76.npy', allow_pickle=True)
# jsonfile = np.load('/userHOME/xzy/projects/kimmo/scanobject/scanobject_target.npy', allow_pickle=True)
# result = list(feature)
# jsonfile = list(jsonfile)
# num = []
# for j, model1 in enumerate(jsonfile):
#   total = 0
#   for i, model2 in enumerate(jsonfile):
#     if model1 == model2:
#       total = total + 1
#   num.append(total)
# p = 0
# r = 0
# precision = 0
# recall = 0
# N = 2882
# # 选择不同的 N 以获取数据
# M = 2882
# list = []
# for i, file in enumerate(result):
#     count = 0
#     for j in range(N):
#         if jsonfile[i] == jsonfile[file[j]]:
#             count = count + 1
#             if count == num[i] and j < (N-1):
#                 count = 0
#                 M = M - 1
#                 break
#     pans = count / N
#     rans = count / num[i]
#     p = p + pans
#     r = r + rans
# precision = p / M
# recall = r / M
# list.append(round(recall, 3))
# list.append(round(precision, 3))
# print(list)

# # scanobject
# # pointnet:     xy = [[0, 1], [0.1, 0.643], [0.2, 0.614], [0.3, 0.596], [0.4, 0.571], [0.5, 0.546], [0.6, 0.489], [0.7, 0.386], [0.8, 0.231], [0.9, 0.144], [1.0, 0.091]]
# # pointnet++:   xy = [[0, 1], [0.1, 0.762], [0.2, 0.736], [0.3, 0.713], [0.4, 0.688], [0.5, 0.657], [0.6, 0.613], [0.7, 0.515], [0.8, 0.280], [0.9, 0.141], [1.0, 0.068]]
# # dgcnn:        xy = [[0, 1], [0.1, 0.745], [0.2, 0.713], [0.3, 0.692], [0.4, 0.670], [0.5, 0.640], [0.6, 0.587], [0.7, 0.474], [0.8, 0.321], [0.9, 0.154], [1.0, 0.084]] 
# # PCT:          xy = [[0, 1], [0.1, 0.722], [0.2, 0.690], [0.3, 0.669], [0.4, 0.648], [0.5, 0.622], [0.6, 0.576], [0.7, 0.460], [0.8, 0.291], [0.9, 0.160], [1.0, 0.091]]
# # CF3d:         xy = [[0, 1], [0.1, 0.712], [0.2, 0.675], [0.3, 0.648], [0.4, 0.622], [0.5, 0.597], [0.6, 0.552], [0.7, 0.468], [0.8, 0.292], [0.9, 0.167], [1.0, 0.079]]




# 任务二 计算TopN指标
feature = np.load('/userHOME/xzy/projects/kimmo/scanobject/feature_result/CF3d_hard_76.npy', allow_pickle=True)
jsonfile = np.load('/userHOME/xzy/projects/kimmo/scanobject/scanobject_target.npy', allow_pickle=True)
result = list(feature)
jsonfile = list(jsonfile)
sum = 0
N = 20
# N 即TopN中的 N,取值[3, 5, 10, 20]
for i, file in enumerate(result):
  count = 0
  for j in range(N):
    if jsonfile[i] == jsonfile[file[j]]:
      count = count + 1
  ans = count / N
  sum = sum + ans
precision = sum / 2882
print(precision)

# # w/o cat.
# # PCT       Top3=0.9198 Top5=0.8441 Top10=0.7351 Top20=0.6459
# # dgcnn     Top3=0.9131 Top5=0.8402 Top10=0.7427 Top20=0.6555
# # pointnet  Top3=0.8012 Top5=0.7151 Top10=0.6188 Top20=0.5438
# # pointnet2 Top3=0.9139 Top5=0.8595 Top10=0.7806 Top20=0.7139
# # CF3d      Top3=0.9404 Top5=0.8724 Top10=0.7600 Top20=0.6664
# # w/ cat.
# # PCT       Top3=0.8987 Top5=0.8448 Top10=0.7828 Top20=0.7392
# # dgcnn     Top3=0.8994 Top5=0.8496 Top10=0.7971 Top20=0.7591
# # pointnet  Top3=0.8223 Top5=0.7618 Top10=0.7047 Top20=0.6637
# # pointnet2 Top3=0.9007 Top5=0.8520 Top10=0.8043 Top20=0.7730
# # CF3d      Top3=0.9032 Top5=0.8475 Top10=0.7805 Top20=0.7315


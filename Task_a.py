""" 
一共三个任务,一是为了获取含label的feature数据
            二是为了获取不含label的feature数据
            三是为了计算MAP
            记得修改输入输出路径
"""
import numpy as np
import math
import os
import json
from tqdm import tqdm

def bit_product_sum(x, y):
    return sum([item[0] * item[1] for item in zip(x, y)])

def cosine_similarity(x, y, norm=True):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if all(x == zero_list) or all(y == zero_list):
        return float(1) if x == y else float(0)
    cos = x@y/(np.sqrt(sum(np.power(x, 2)))*np.sqrt(sum(np.power(y, 2))))
    return 1.0-(0.5 * cos + 0.5) if norm else cos  # 归一化到[0, 1]区间内



# # 先判断label再对特征距离排序
# feature = np.load('/userHOME/xzy/projects/kimmo/scanobject/feature_result/CF3d_hard_76.npy', allow_pickle=True)
# label = np.load('/userHOME/xzy/projects/kimmo/scanobject/label_result/CF3d_hard_76_label.npy', allow_pickle=True)
# label = list(label)
# feature = list(feature)
# labelretrieval = []
# for i, pred in tqdm(enumerate(label,0),total=len(label),smoothing=0.9):
#   index = []
#   distance = []
#   sort = []
#   diffindex = []
#   diffdistance = []
#   for j, model in enumerate(label):
#     if pred == model:
#       index.append(j)
#       d = cosine_similarity(feature[i][0], feature[j][0])
#       # d = np.sqrt(np.sum(np.square(feature[i][0]-feature[j][0])))
#       distance.append(d)
#     else:
#       diffindex.append(j)
#       diffd = cosine_similarity(feature[i][0], feature[j][0])
#       # diffd = np.sqrt(np.sum(np.square(feature[i][0]-feature[j][0])))
#       diffdistance.append(diffd)
#   sorted_id = sorted(range(len(distance)), key=lambda k: distance[k], reverse=False)
#   diffsorted_id = sorted(range(len(diffdistance)), key=lambda k: diffdistance[k], reverse=False)
#   for t, i in enumerate(sorted_id):
#     sort.append(index[i])
#   for t, i in enumerate(diffsorted_id):
#     sort.append(diffindex[i])
#   labelretrieval.append(sort)
# labelretrieval = np.array(labelretrieval)
# print(labelretrieval)
# print(len(labelretrieval))
# np.save('/userHOME/xzy/projects/kimmo/scanobject/retrieval/CF3d_hard_76abel.npy', labelretrieval)



# # 直接对特征距离排序
# feature = np.load('/userHOME/xzy/projects/kimmo/scanobject/feature_result/CF3d_hard_76.npy', allow_pickle=True)
# retrieval = []
# for i, pred in tqdm(enumerate(feature,0),total=len(feature),smoothing=0.9):
#   distance = []
#   for j, model in enumerate(feature):
#     d = np.sqrt(np.sum(np.square(pred - model)))
#     distance.append(d)
#   sorted_id = sorted(range(len(distance)), key=lambda k: distance[k], reverse=False)
#   retrieval.append(sorted_id)
# np.save('/userHOME/xzy/projects/kimmo/scanobject/retrieval/CF3d_76.npy', retrieval)



# # 计算两种的MAP
jsonfile = np.load('/userHOME/xzy/projects/kimmo/scanobject/scanobject_target.npy', allow_pickle=True)
feature = np.load('/userHOME/xzy/projects/kimmo/scanobject/retrieval/CF3d_hard_76abel.npy', allow_pickle=True)
result = list(feature)
jsonfile = list(jsonfile)
AP = 0
for i, file in enumerate(result):
  count = 0
  precision = 0
  number = 0
  for j, model in enumerate(jsonfile):
    # if model[:3] == jsonfile[i][:3]:
    if model == jsonfile[i]:
      count += 1
  for t,index in enumerate(file):
    # if jsonfile[i][:3] == jsonfile[index][:3]:
    if jsonfile[i] == jsonfile[index]:
      number += 1
      precision += (number / (t + 1))
      if count == number:
        break
  precision = precision / count
  AP = AP + precision
MAP = AP / 2882
print(MAP)

# PCT       MAP 0.6252360824309484
# PCT       MAP 0.36075575306954427
# dgcnn     MAP 0.6408524908194857
# dgcnn     MAP 0.35841829185979357
# pointnet2 MAP 0.6615720389817821
# pointnet2 MAP 0.44543554200421215
# pointnet  MAP 0.5513416184380456
# pointnet  MAP 0.3150856311506599
# CF3d      MAP 0.6110461976264758
# CF3d      MAP 0.4200247911434806





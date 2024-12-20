import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
from glob import glob
from Common import point_operation, data_utils as d_utils
import os
warnings.filterwarnings('ignore')
from torchvision import transforms

def load_data(dir,partition="training"):

    all_data = []
    all_label = []
    midfilenames = ['main_split_nobg', 'split1_nobg', 'split2_nobg', 'split3_nobg', 'split4_nobg']
    for midfilename in midfilenames:
        dir_mid = dir
        dir_mid = os.path.join(dir_mid, midfilename)
        h5_name = os.path.join(dir_mid, '%s_objectdataset.h5'%partition)
        print(h5_name)
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_label = all_label.reshape(-1,1)
    print(all_data.shape)
    if len(all_data) > 11264:
        all_data = all_data[:11264][:]
        all_label = all_label[:11264][:]
    print(all_data.shape)
    print(all_label.shape)
    return all_data, all_label

point_transform = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
    ]
)

class ScanObjectDataLoader(Dataset):
    def __init__(self, opts,partition='training'):
        self.opts = opts
        opts.data_dir = '/userHOME/xzy/projects/kimmo/dataLab/h5_files'  
        self.data, self.label = load_data(opts.data_dir,partition=partition)
        self.num_points = opts.num_points
        self.partition = partition
        self.dim = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pc = self.data[index][:self.num_points,:self.dim].copy()
        label = self.label[index]
        # 获取feature和label信息时需要把以下 if代码 备注
        if self.opts.augment and self.partition == 'training':
            pc = point_operation.rotate_point_cloud_and_gt(pc)
            pc = point_operation.jitter_perturbation_point_cloud(pc)
            if self.opts.is_dg:
                pc,_ = point_operation.random_scale_point_cloud_and_gt(pc)
                pc = point_operation.rotate_perturbation_point_cloud(pc)
                pc = point_operation.shift_point_cloud_and_gt(pc)
        return pc.astype(np.float32), label.astype(np.int32)

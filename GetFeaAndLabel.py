import argparse
import os
import sys
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from Common.ModelNetDataLoader import ModelNetDataLoader, load_data
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.utils import test, save_checkpoint
from utils.pointconv import PointConvDensityClsSsg as PointConvClsSsg
# from data_utils.ShapeNetDataLoader import PartNormalDataset
from Common.ScanObjectDataLoader import ScanObjectDataLoader

def parse_args():
    '''PARAMETERS'''
    '''此处需设置模型的路径'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default='/userHOME/xzy/projects/kimmo/CF3dHardcopy/log/pointconv_cls/20220805-0006/pointconv-0.763012-0082.pth', help='checkpoint')
    # parser.add_argument('--checkpoint', type=str, default='/home/lizuo/data/PointAugment-master/log/pointconv_cls/20210811-1632/pointconv-0.924230-0121.pth', help='checkpoint')
    parser.add_argument('--num_view', type=int, default=4, help='num of view')
    parser.add_argument('--npoint', type=int, default=1024, help='num of view')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--num_points', type=int, default=2048, help='batch size in training')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--augment', type=str2bool, default=False)
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = '/home/lizuo/data/modelnet40_ply_hdf5_2048/'

    '''CREATE DIR'''
    experiment_dir = Path('./modelnet_eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')

    # root = '/home/lizuo/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    # train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    # logger.info("The number of training data is: %d",train_data.shape[0])
    # logger.info("The number of test data is: %d", test_data.shape[0])
    # testDataset = ModelNetDataLoader(test_data, test_label)
    # testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)
    testDataLoader = DataLoader(ScanObjectDataLoader(args,partition='test'), batch_size=args.batch_size, shuffle=False,)

    '''MODEL LOADING'''
    # num_class = 16
    num_class = 15
    classifier = PointConvClsSsg(num_class).cuda()
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
        print(args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('Please load Checkpoint to eval...')
        sys.exit(0)
        start_epoch = 0

    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    # total_correct = 0
    # total_seen = 0
    # # predictlabel = []
    # # predictfeature = []
    # for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
    #     pointcloud, target = data
    #     target = target[:, 0]
    #     #import ipdb; ipdb.set_trace()
    #     # pred_view = torch.zeros(pointcloud.shape[0], num_class).cuda()
    #     # global_feature = torch.zeros(pointcloud.shape[0], 1024).cuda()

    #     # pointcloud = generate_new_view(pointcloud)
    #     #import ipdb; ipdb.set_trace()
    #     #points = torch.from_numpy(pointcloud).permute(0, 2, 1)
    #     points = pointcloud.permute(0, 2, 1)
    #     points, target = points.cuda(), target.cuda()
    #     classifier = classifier.eval()
    #     with torch.no_grad():
    #         pred= classifier(points)
    #     # pred_view += pred
    #     # global_feature += feature

    #     pred_choice = pred.data.max(1)[1]
    #     # pred_choice = pred_choice.cpu().numpy()
    #     # global_feature = global_feature.cpu().numpy()
    #     # predictlabel.append(pred_choice)
    #     # predictfeature.append(global_feature)
    #     correct = pred_choice.eq(target.long().data).cuda().sum()
    #     total_correct += correct.item()
    #     total_seen += float(points.size()[0])
    # accuracy = total_correct / total_seen
    # print('Total Accuracy: %f'%accuracy)

    # logger.info('Total Accuracy: %f'%accuracy)
    # logger.info('End of evaluation...')



    total_correct = 0
    total_seen = 0
    predictlabel = []
    predictfeature = []
    testModel = []
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        pointcloud, target = data
        pointcloud = pointcloud.cpu().numpy()
        testModel.append(pointcloud)
        target = target[:, 0]
        pred_view = torch.zeros(pointcloud.shape[0], num_class).cuda()
        global_feature = torch.zeros(pointcloud.shape[0], 1024).cuda()

        for _ in range(args.num_view):
            pointcloud = generate_new_view(pointcloud)
            points = torch.from_numpy(pointcloud).permute(0, 2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            with torch.no_grad():
                a, pred, b, feature = classifier(points)
            pred_view += pred
            global_feature += feature
        pred_choice = pred_view.data.max(1)[1]
        pred_choice = pred_choice.cpu().numpy()
        pred_view = pred_view.cpu().numpy()
        global_feature = global_feature.cpu().numpy()
        # label
        predictlabel.append(pred_choice)
        # 特征
        predictfeature.append(global_feature)
        # predictfeature.append(pred_view)
    predictfeature = np.array(predictfeature)
    predictlabel = np.array(predictlabel)
    # testModel = np.array(testModel)
    #这里记得改名字
    np.save('/userHOME/xzy/projects/kimmo/experiment/feature_result/CF3d_hard_76.npy', predictfeature)
    np.save('/userHOME/xzy/projects/kimmo/experiment/label_result/CF3d_hard_76.npy', predictlabel)
    # np.save('/userHOME/xzy/projects/kimmo/experiment/testModel.npy', testModel)



def generate_new_view(points):
    points_idx = np.arange(points.shape[1])
    np.random.shuffle(points_idx)

    points = points[:, points_idx, :]
    return points


def rotate_point_cloud_by_angle(data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    :param batch_data: Nx3 array, original batch of point clouds
    :param rotation_angle: range of rotation
    :return:  Nx3 array, rotated batch of point clouds
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]], dtype=np.float32)
    rotated_data = np.dot(data, rotation_matrix)

    return rotated_data

if __name__ == '__main__':
    args = parse_args()
    main(args)

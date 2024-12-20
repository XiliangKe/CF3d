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
from utils.utils import test, save_checkpoint
from utils.pointconv import PointConvDensityClsSsg as PointConvClsSsg
from Common.ShapeNetDataLoader import PartNormalDataset


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('--num_view', type=int, default=4, help='num of view')
    parser.add_argument('--npoint', type=int, default=1024, help='num of view')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    datapath = '/data/modelnet40_ply_hdf5_2048/'

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

    # root = '/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    train_data, train_label, test_data, test_label = load_data(datapath, classification=True)
    logger.info("The number of training data is: %d",train_data.shape[0])
    logger.info("The number of test data is: %d", test_data.shape[0])
    testDataset = ModelNetDataLoader(test_data, test_label)
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=args.batchsize, shuffle=False)
    

    '''MODEL LOADING'''
    # num_class = 16
    num_class = 40
    classifier = PointConvClsSsg(num_class).cuda()
    if args.checkpoint is not None:
        print('Load CheckPoint...')
        logger.info('Load CheckPoint')
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

    total_correct = 0
    total_seen = 0
    for batch_id, data in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
        pointcloud, target = data
        target = target[:, 0]
        points = pointcloud.permute(0, 2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        with torch.no_grad():
            pred= classifier(points)
        pred_view += pred
        global_feature += feature

        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cuda().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])
    accuracy = total_correct / total_seen
    print('Total Accuracy: %f'%accuracy)

    logger.info('Total Accuracy: %f'%accuracy)
    logger.info('End of evaluation...')



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

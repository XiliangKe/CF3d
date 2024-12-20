#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: loss_utils.py 
@time: 2019/09/23
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools

NUM = 1.2#2.0
W = 1.0#10.0

def center_loss(feature, label, lambdas):
    """
    计算中心损失
    :param feature: 网络输出特征 (N, 2)
    :param label: 分类标签 如 tensor([0, 2, 1, 0, 1])
    :param lambdas: 参数 控制中心损失大小 即类内间距
    :return:
    """
    feature = feature.cuda()
    center = nn.Parameter(torch.randn(int(max(label).item() + 1), feature.shape[1]), requires_grad=True).cuda()
    center_exp = center.index_select(0, label.long())
    count = torch.histc(label, bins=int(max(label).item() + 1), min=0, max=int(max(label).item()))
    count_exp = count.index_select(dim=0, index=label.long()).float()
    loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(feature - center_exp, 2), dim=1), count_exp))
    return loss


def cal_loss_raw(pred, gold):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    #one_hot = F.one_hot(gold, pred.shape[1]).float()

    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss_raw = -(one_hot * log_prb).sum(dim=1)


    loss = loss_raw.mean()

    return loss,loss_raw

def mat_loss(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss



def cls_loss(pred, pred_aug, gold, pc_tran, aug_tran, pc_feat, aug_feat, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    # cls_pc, _ = cal_loss_raw(pred, gold)
    # cls_aug, _ = cal_loss_raw(pred_aug, gold)
    trip_hard_loss = TripHardLoss()
    cosin_loss = CosinLoss()
    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold)
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran)
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)

    #此处是需要修改的部分，要将以下内容改成余弦相似度
    feat_diff = 10.0*trip_hard_loss(pc_feat,aug_feat,gold)
    
    # feat_diff = 10.0*cosin_loss(pc_feat,aug_feat)
    # feat_diff = 10.0*mse_fn(pc_feat,aug_feat)
    # parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(1.0-cls_pc_raw)**2).cuda()
    # cls_diff = (torch.abs(cls_pc_raw - cls_aug_raw) * (parameters*2)).mean()
    cls_loss = cls_pc + cls_aug  + feat_diff# + cls_diff

    return cls_loss

def aug_loss(pred, pred_aug, gold, pc_tran, aug_tran, ispn = True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)

    cls_pc, cls_pc_raw = cal_loss_raw(pred, gold) #[], [24]
    cls_aug, cls_aug_raw = cal_loss_raw(pred_aug, gold)
    if ispn:
        cls_pc = cls_pc + 0.001*mat_loss(pc_tran) 
        cls_aug = cls_aug + 0.001*mat_loss(aug_tran)
    pc_con = F.softmax(pred, dim=-1)#.max(dim=1)[0]
    one_hot = F.one_hot(gold, pred.shape[1]).float()
    pc_con = (pc_con*one_hot).max(dim=1)[0]

     
    parameters = torch.max(torch.tensor(NUM).cuda(), torch.exp(pc_con) * NUM).cuda()
    
    # both losses are usable
    aug_diff = W * torch.abs(1.0 - torch.exp(cls_aug_raw - cls_pc_raw * parameters)).mean()
    #aug_diff =  W*torch.abs(cls_aug_raw - cls_pc_raw*parameters).mean()
    aug_loss = cls_aug + aug_diff

    return aug_loss


def triplet_loss(pc_feat, aug_feat, target, margin=0.3):
    """
    pc_feat: the point cloud feature [batch_size, dim]
    aug_feat: the augment feature [batch_size, dim]
    target: lable [batch_size]

    """
    batch_size = pc_feat.size()[0]
    mask = torch.zeros(batch_size*batch_size,1)
    loss = 0
    t = itertools.product(target,target)
    for i,elem in enumerate(t):
        sub = torch.sub(input=elem[0], alpha=1, other = elem[1])
        if sub == 0:
            mask[i] = 1
        else:
            mask[i] = -1
    p = itertools.product(pc_feat, aug_feat)
    mse_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    for i, elem in enumerate(p):
        loss = mse_fn(elem[0],elem[1])*mask[i] + loss
    loss = loss + margin
    if loss <0:
        loss = 0
    return loss

class CosinLoss(nn.Module):
    def __init__(self):
        super(CosinLoss, self).__init__()
        
    def forward(self,pc_feat, aug_feat):
        batch_size = pc_feat.size()[0]
        pc_feat_normal = F.normalize(pc_feat,dim=1)
        aug_feat_normal = F.normalize(aug_feat,dim=1)
        mm = torch.mm(pc_feat_normal, aug_feat_normal.t())
        ones = torch.ones_like(mm)
        mm_one = torch.sub(ones, mm)
        trace = mm_one.diag()
        loss = torch.sum(trace, dim=0)/batch_size
        return loss

class TripHardLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self ,margin=0.1):
        super(TripHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
 
    def forward(self, pc_feat, aug_feat, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = pc_feat.size(0)	# batch_size
        
        # Compute pairwise distance, replace by the official when merged
        dist_pc = torch.pow(pc_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist_aug = torch.pow(aug_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist_pc + dist_aug.t()
        dist.addmm_(1, -2, pc_feat, aug_feat.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

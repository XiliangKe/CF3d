"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.pointconv_util import PointConvDensitySetAbstraction

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x))) # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1) # b, n, c [16,256,64]
        x_k = self.k_conv(x)# b, c, n [16,64,256]
        x_v = self.v_conv(x) # [16,256,256]
        energy = x_q @ x_k # b, n, n  [16,256,256]
        attention = self.softmax(energy) # [16,256,256]
        # y = torch.sum(attention, dim=1, keepdim=True)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention # b, c, n 
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

class StackedAttention(nn.Module):
    def __init__(self, channels=256):
    # def __init__(self, channels=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        x = self.relu(self.bn1(self.conv1(x))) # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        
        x = torch.cat((x1, x2, x3), dim=1)

        return x

class PointConvDensityClsSsg(nn.Module):
    def __init__(self, num_classes = 3):
        super(PointConvDensityClsSsg, self).__init__()
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        
        self.sa2 = PointConvDensitySetAbstraction(npoint=256, nsample=32, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
               
        self.pt_last = StackedAttention()
        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None) #[12, 3, 512], [12, 128, 512] 

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) #[12, 3, 256], [12, 256, 256] 

        x = self.pt_last(l2_points)
        x = torch.cat([x, l2_points], dim=1)
        x = self.conv_fuse(x)
        x = torch.max(x, 2)[0] #[12, 1024]
        x = x.view(B, -1)
        y = x
        x = self.drop1(F.relu(self.bn1(self.fc1(x)))) #[12, 512]
        x = self.drop2(F.relu(self.bn2(self.fc2(x)))) #[12, 256]
        x = self.fc3(x) #[12, 40]
        f = x
        x = F.log_softmax(x, -1)
        return f, x, None, y
        

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    label = torch.randn(8,16)
    model = PointConvDensityClsSsg(num_classes=40)
    output= model(input)
    print(output.size())


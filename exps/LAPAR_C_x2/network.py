import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from utils.modules.lightWeightNet import WeightNet


class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)
        return out
    
class RightConv(nn.Module):
    def __init__(self, k_size):
        super(RightConv, self).__init__()

        kernel = np.load('/data/hdliu/TCAD/retrain_batch_64/Simple-SR/kernel/kernel.npy')
        bias = np.load('/data/hdliu/TCAD/retrain_batch_64/Simple-SR/kernel/bias.npy')
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        bias =  torch.from_numpy(bias).float().view(-1)
        self.register_buffer('weight', kernel)
        self.register_buffer('bias', bias)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=self.bias, stride=1, padding=2, groups=1)
        return out

'''
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.k_size = config.MODEL.KERNEL_SIZE
        self.s = config.MODEL.SCALE

        self.w_conv = WeightNet(config.MODEL)
        self.decom_conv = ComponentDecConv(config.MODEL.KERNEL_PATH, self.k_size)

        self.criterion = nn.L1Loss(reduction='mean')


    def forward(self, x, gt=None):
        B, C, H, W = x.size()

        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws

        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws

        out = torch.sum(weight * x_com, dim=2)

        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out
'''
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.layer_0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.pointConv = torch.nn.Conv2d(32, 32, 1, 1, 0)
        self.layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 1),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 1, 1),
        )
        # self.layer_5 = torch.nn.Conv2d(1, 8, 5, 1, 2)
        self.layer_5 = RightConv(5)
        self.pix_up = torch.nn.PixelShuffle(4)
        self.criterion = nn.L1Loss(reduction='mean')
        
    def forward(self, x, gt=None):
        # print(' x shape', x.size())
        x_0 = self.layer_0(x)
        # print('x size', x.size())
        # print('x_0 size', x_0.size())
        x_1 = self.layer_1(x_0)
        x_2 = self.layer_2(x_1)
        x_3 = self.layer_3(x_2)
       
        x_point = self.pointConv(x_2)
        x_4 = self.layer_4(torch.cat([x_3, x_point], 1))
       
        x_5 = self.pix_up(x_4)   
        bic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        # print('x5 shape', x_5.size())
        # print('bic shape', bic.size())
        # x_6 = torch.einsum('bmhw,bmhw->bhw',x_5, self.layer_5(right_input))
        # x_6 = torch.mul(x_5, self.layer_5(bic))
        out = torch.sum(x_5 * self.layer_5(bic), 1)
        N, H, W = out.size()
        out = out.view(N, 1, H, W)
        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out


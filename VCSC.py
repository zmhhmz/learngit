#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06
import torch
import torch.nn.functional as F
from .SubBlocks import conv3x3, activation_set
import torch.nn as nn

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=70, depth=4, wf=64, batch_norm=False,
                                                                  activation='relu', act_init=0.01):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
            batch_norm (bool): Use BatchNorm after layers with an activation function
        """
        super(UNet, self).__init__()
        self.depth = depth
        self.channels=in_channels
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        self.M_channels = (out_channels-2*in_channels)//2
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, batch_norm, activation,
                                                                                          act_init))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, batch_norm, activation,
                                                                                          act_init))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_channels, bias=True)
        self.ENet1 = nn.Sequential(
                torch.nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                torch.nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        self.ENet2 = nn.Sequential(
                torch.nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                torch.nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        self.ENet3 = nn.Sequential(
                torch.nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                torch.nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        output=self.last(x)
        x=output[:,:self.channels,:,:]
        sigmaX=output[:,self.channels:2*self.channels,:,:]
        M=output[:,2*self.channels:2*self.channels+self.M_channels,:,:]
        sigmaM=output[:,2*self.channels+self.M_channels:,:,:]
        E1=self.ENet1(M)
        E2=self.ENet2(M)
        E3=self.ENet3(M)
        E=torch.cat((E1,E2,E3),1)
        return x,sigmaX,M,sigmaM,E

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, activation, act_init):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(activation_set(activation, act_init, None))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(activation_set(activation, act_init, None))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, activation, act_init):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, batch_norm, activation, act_init)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

class VCSC(nn.Module):
    def __init__(self, in_channels, activation='relu', act_init=0.01, wf=64, dep_S=5, dep_U=4,
                                                                                   batch_norm=True):
        super(VCSC, self).__init__()
        net1 = UNet(in_channels, wf=wf, depth=dep_U, batch_norm=batch_norm,
                                                           activation=activation, act_init=act_init)
        self.Net = weight_init_kaiming(net1)

    def forward(self, y):
        x,sigmaX,M,sigmaM,E = self.Net(y)
        return y-x,sigmaX,M,sigmaM,E


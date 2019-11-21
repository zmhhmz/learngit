#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 19:35:06
import torch
import torch.nn as nn
from .DnCNN import DnCNN
from .UNet import UNet


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

class VCSC(nn.Module):
    def __init__(self, in_channels, activation='relu', act_init=0.01, wf=64, dep_S=5, dep_U=4,
                                                                                   batch_norm=True):
        super(VCSC, self).__init__()
        self.im_channels = in_channels
        self.M_channels = wf
        net1 = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, batch_norm=batch_norm,
                                                           activation=activation, act_init=act_init)
        self.DNet = weight_init_kaiming(net1)
        net2 = DnCNN(in_channels, 2*wf, dep=dep_S, num_filters=64, activation=activation,
                                                                                  act_init=act_init)
        self.SNet = weight_init_kaiming(net2)
        ENet1 = nn.Sequential(
                nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        self.ENet1 = weight_init_kaiming(ENet1)
        ENet2 = nn.Sequential(
                nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        self.ENet2 = weight_init_kaiming(ENet2)
        ENet3 = nn.Sequential(
                nn.Conv2d(self.M_channels,self.M_channels,kernel_size=3, stride=1,padding=1,  bias=False,groups=self.M_channels),
                nn.Conv2d(self.M_channels,1,1,bias=False)
                )
        self.ENet3 = weight_init_kaiming(ENet3)

    def forward(self, x):
        X_sigmaX = self.DNet(x)
        X = X_sigmaX[:,:self.im_channels,:,:]
        sigmaX=X_sigmaX[:,self.im_channels:,:,:]
        M_sigmaM = self.SNet(x)
        M = M_sigmaM[:,:self.M_channels,:,:]
        sigmaM = M_sigmaM[:,self.M_channels:,:,:]
        E1 = self.ENet1(M)
        E2 = self.ENet2(M)
        E3 = self.ENet3(M)
        E=torch.cat((E1,E2,E3),1)
        return x-X,sigmaX,M,sigmaM,E



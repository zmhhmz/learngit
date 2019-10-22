#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:39:52 2019

@author: zhouminghao
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from .SubBlocks import conv3x3, activation_set

# class Enet(nn.Module): 
#     def __init__(self, image_channels=3, n_channels=64):
#         super(Enet, self).__init__()
#         self.channels=image_channels
#         layers=[]
#         layers.append(nn.Conv2d(in_channels=image_channels+n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(3):
#             layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
#             layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.9))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4*image_channels, kernel_size=3, padding=1, bias=False))
#         self.cnnblock = nn.Sequential(*layers)
        
#     def forward(self, Y,M):
#         x1=torch.cat((Y,M),1)
#         out = self.cnnblock(x1)
# #        mu=out[:,:self.channels,:,:]+Y
# #        m2=out[:,self.channels:2*self.channels,:,:]
# #        alpha=out[:,2*self.channels:3*self.channels,:,:]
# #        beta = out[:,3*self.channels:,:,:]
#         #return mu,m2,alpha,beta
#         out[:,:self.channels,:,:]+=Y
#         return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=12, depth=4, wf=64, batch_norm=False,
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
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
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

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

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

class Mnet(nn.Module): 
    def __init__(self, image_channels=3, n_channels=64):
        super(Mnet, self).__init__()
        self.channels=n_channels
        layers=[]
        layers.append(nn.Conv2d(in_channels=5*image_channels+n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.9))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers)
        
    def forward(self, out,Y,M):
        x1=torch.cat((out,Y,M),1)
        out = self.cnnblock(x1)
        M=out+M
        return M


class VBEMnet(nn.Module):
    def __init__(self, level=4, n_channels=64, image_channels=3):
        super(VBEMnet, self).__init__()
        self.level = level
        self.n_channels=n_channels
        self.image_channels=image_channels
        self.initMNet = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False)
        net1=UNet(in_channels, in_channels*4, wf=n_channels, depth=4, batch_norm=True, activation='relu', act_init=0.01)
        self.Enet=weight_init_kaiming(net1)
        net2=Mnet(image_channels,n_channels)
        self.Mnet=weight_init_kaiming(net2)

    def forward(self, Y):
        Outlist=[]
        #initialize M
        M=self.initMNet(Y)
        for i in range(self.level):
            #VB-E
            inputYM = torch.cat((Y,M),1)
            out=self.Enet(inputYM)
            Outlist.append(out)
            #VB-M
            M=self.Mnet(out,Y,M)
        #final output
        inputYM = torch.cat((Y,M),1)
        out=self.Enet(inputYM)
        Outlist.append(out)
        return Outlist
    

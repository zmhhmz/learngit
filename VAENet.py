#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:39:52 2019

@author: zhouminghao
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class Xnet(nn.Module): 
    def __init__(self, image_channels=3, n_channels=64):
        super(Xnet, self).__init__()
        self.channels=image_channels
        layers=[]
        layers.append(nn.Conv2d(in_channels=image_channels+n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.9))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers)
        
    def forward(self, Y,Z):
        x1=torch.cat((Y,Z),1)
        out = self.cnnblock(x1)
        mu=out+Y
        #sigma=torch.sigmoid(out[:,self.channels:,:,:])
        #print(mu.size())
        return mu#,sigma

class Znet(nn.Module): 
    def __init__(self, image_channels=3, n_channels=64):
        super(Znet, self).__init__()
        self.channels=n_channels
        layers=[]
        layers.append(nn.Conv2d(in_channels=image_channels+n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.9))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers)
        
    def forward(self, Y,Z):
        x1=torch.cat((Y,Z),1)
        out = self.cnnblock(x1)
        mu=out#[:,:self.channels,:,:]#+Z
        #sigma=torch.sigmoid(out[:,self.channels:,:,:])
        return mu#,sigma


class ReconstructNet(nn.Module):
    def __init__(self, image_channels, n_channels):
        super(ReconstructNet, self).__init__()
        self.channel=image_channels
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels+n_channels, out_channels=2*n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1):
            layers.append(nn.Conv2d(in_channels=2*n_channels, out_channels=2*n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(2*n_channels, eps=0.0001, momentum = 0.9))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=2*n_channels, out_channels=image_channels+n_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers)

    def forward(self, Y, Z):
        x1 = torch.cat((Y, Z), 1)
        out = x1-self.cnnblock(x1)
        Y = out[:,:self.channel,:,:]  
        Z = out[:,self.channel:,:,:]  
        return Y, Z


class VAEnet(nn.Module):
    def __init__(self, level=4, n_channels=96, image_channels=3):
        super(VAEnet, self).__init__()
        self.level = level
        self.n_channels=n_channels
        self.image_channels=image_channels
        self.initZpNet = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False)
        self.Xnet1=Xnet(image_channels,n_channels)
        self.Xnet2=Xnet(image_channels,n_channels)
        self.Xnet3=Xnet(image_channels,n_channels)
        self.Xnet4=Xnet(image_channels,n_channels)

        self.Znet1=Znet(image_channels,n_channels)
        self.Znet2=Znet(image_channels,n_channels)
        self.Znet3=Znet(image_channels,n_channels)
        self.Znet4=Znet(image_channels,n_channels)

        self.Enet1 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False)
        self.Enet2 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False)
        self.Enet3 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False)
        self.Enet4 = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=3, padding=1, bias=False)

        self.Xnetlist=[self.Xnet1,self.Xnet2,self.Xnet3,self.Xnet4]
        self.Znetlist=[self.Znet1,self.Znet2,self.Znet3,self.Znet4]
        self.Enetlist=[self.Enet1,self.Enet2,self.Enet3,self.Enet4]
        self.Rnet = ReconstructNet(image_channels,n_channels)

        self._initialize_weights()

    def forward(self, Y):
        Zlist=[]
        Xlist=[]
        Elist=[]
        reYlist=[]
        reZlist=[]
        Z=self.initZpNet(Y)
        Xlist.append(Y)
        Zlist.append(Z)
        for i in range(self.level):
            Z=self.Znetlist[i](Y,Z)
            Zlist.append(Z)
            E=self.Enetlist[i](Z)
            Elist.append(E)
            Y=self.Xnetlist[i](Y,Z)
            Xlist.append(Y)
            reY,reZ=self.Rnet(Y,Z)
            reYlist.append(reY)
            reZlist.append(reZ)

        return Xlist,Zlist,Elist,reYlist,reZlist
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1/100)
                init.constant_(m.bias, 0)
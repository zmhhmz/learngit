#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:39:52 2019

@author: zhouminghao
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class Enet(nn.Module): 
    def __init__(self, image_channels=3, n_channels=64):
        super(Enet, self).__init__()
        self.channels=image_channels
        layers=[]
        layers.append(nn.Conv2d(in_channels=image_channels+n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(3):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.9))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4*image_channels, kernel_size=3, padding=1, bias=False))
        self.cnnblock = nn.Sequential(*layers)
        
    def forward(self, Y,M):
        x1=torch.cat((Y,M),1)
        out = self.cnnblock(x1)
#        mu=out[:,:self.channels,:,:]+Y
#        m2=out[:,self.channels:2*self.channels,:,:]
#        alpha=out[:,2*self.channels:3*self.channels,:,:]
#        beta = out[:,3*self.channels:,:,:]
        #return mu,m2,alpha,beta
        out[:,:self.channels,:,:]+=Y
        return out

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
    def __init__(self, level=4, n_channels=128, image_channels=3):
        super(VBEMnet, self).__init__()
        self.level = level
        self.n_channels=n_channels
        self.image_channels=image_channels
        self.initMNet = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=3, padding=1, bias=False)
        self.Enet1=Enet(image_channels,n_channels)
        self.Enet2=Enet(image_channels,n_channels)
        self.Enet3=Enet(image_channels,n_channels)
        self.Enet4=Enet(image_channels,n_channels)
#        self.Enet5=Enet(image_channels,n_channels)
#        self.Enet6=Enet(image_channels,n_channels)    	

        self.Mnet1=Mnet(image_channels,n_channels)
        self.Mnet2=Mnet(image_channels,n_channels)
        self.Mnet3=Mnet(image_channels,n_channels)
        self.Mnet4=Mnet(image_channels,n_channels)
#        self.Mnet5=Mnet(image_channels,n_channels)
#        self.Mnet6=Mnet(image_channels,n_channels)

        self.Enetlist=[self.Enet1,self.Enet2,self.Enet3,self.Enet4]#,self.Enet5,self.Enet6]
        self.Mnetlist=[self.Mnet1,self.Mnet2,self.Mnet3,self.Mnet4]#,self.Mnet5,self.Mnet6]

        self._initialize_weights()

    def forward(self, Y):
        Outlist=[]
        #initialize M
        M=self.initMNet(Y)
        for i in range(self.level):
            #VB-E
            out=self.Enetlist[i](Y,M)
            Outlist.append(out)
            #VB-M
            M=self.Mnetlist[i](out,Y,M)
        #final output
        out=self.Enetlist[i](Y,M)
        Outlist.append(out)
        return Outlist
    
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

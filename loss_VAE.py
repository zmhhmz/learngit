#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
#import torch.nn.functional as F

lamX = [0.1,0.1,0.1,0.1,0.1,1]
lamE = [0.1,0.1,0.1,0.1,0.1,1]
lamReY=0.05
lamReZ=0.005


def loss_fn(Xlist, Zlist,Elist,reYlist,reZlist , X,Y,batch_size):
    lossX=0
    lossE=0
    lossReY=0
    lossReZ=0
    for i in range(len(Xlist)-1):
        if(i==len(Xlist)-2):
            lossOut = lamX[i]*torch.nn.functional.mse_loss(Xlist[i+1], X,size_average=False,reduce=True).div_(2)
            lossX+=lossOut
        else:
            lossX += lamX[i]*torch.nn.functional.mse_loss(Xlist[i+1], X,size_average=False,reduce=True).div_(2)
        lossE += lamE[i]*torch.nn.functional.mse_loss(Elist[i], X-Xlist[i],size_average=False,reduce=True).div_(2)
        lossReY += lamReY*torch.nn.functional.mse_loss(reYlist[i], Xlist[i],size_average=False,reduce=True).div_(2)
        lossReZ += lamReZ*torch.nn.functional.mse_loss(reZlist[i], Zlist[i],size_average=False,reduce=True).div_(2)
    loss = lossX+lossE+lossReY+lossReZ
    return loss/batch_size,lossOut/batch_size,lossX/batch_size,lossE/batch_size,lossReY/batch_size,lossReZ/batch_size

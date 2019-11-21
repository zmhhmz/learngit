#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
from math import log
from utils import LogGamma

log_gamma = LogGamma.apply

# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def loss_fn(im_out,sigmaX,M,sigmaM,E, im_noisy, im_gt, eps2):
    '''
    Input:
        eps2: variance of the Gaussian prior of x
    '''

    # parameters predicted of Gaussain distribution
    sigmaX.clamp_(min=log_min, max=log_max)
    sigmaM.clamp_(min=log_min, max=log_max)
    
    m2 = torch.exp(sigmaX)   # variance
    M2 = torch.exp(sigmaM)

    # KL divergence for x
    m2_div_eps = torch.div(m2, eps2)
    kl_x = 0.5 * torch.mean((im_out-im_gt)**2/eps2 + (m2_div_eps - 1 - torch.log(m2_div_eps)))

    # KL divergence for M
    kl_M = 0.5 * torch.mean(M2 - 1 - sigmaM)

    # likelihood of im_gt
    lh = 0.5 * torch.mean((im_gt-im_out-E)**2)
    
    loss = lh + kl_x + kl_M

    return loss, lh, kl_x, kl_M
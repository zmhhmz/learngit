#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-10 22:41:49

from glob import glob
import warnings
import time
import numpy as np
import shutil
import torchvision.utils as vutils
from utils import batch_PSNR, batch_SSIM
from tensorboardX import SummaryWriter
from math import ceil
from loss_VCSC import loss_fn
from networks import VCSC
from datasets import DenoisingDatasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import sys
from pathlib import Path
from options_VCSC import set_opts
import matplotlib.pyplot as plt

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

args = set_opts()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if isinstance(args.gpu_id, int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args.gpu_id))

_lr_min = 1e-6
_modes = ['train', 'test_cbsd681', 'test_cbsd682', 'test_cbsd683']

def imshow(X):
    X = np.maximum(X, 0)
    X = np.minimum(X, 1)
    plt.imshow(X.squeeze(),cmap='gray')
    plt.axis('off')
    plt.show()

def train_model(net, datasets, optimizer, lr_scheduler, criterion):
    C = args.chn
    clip_grad = args.clip_grad
    batch_size = {'train': args.batch_size, 'test_cbsd681': 1, 'test_cbsd682': 1, 'test_cbsd683': 1}
    data_loader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size[phase],
          shuffle=True, num_workers=args.num_workers, pin_memory=True) for phase in datasets.keys()}
    num_data = {phase: len(datasets[phase]) for phase in datasets.keys()}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in datasets.keys()}
    writer = SummaryWriter(args.log_dir)
    if args.resume:
        step = args.step
        step_img = args.step_img
    else:
        step = 0
        step_img = {x: 0 for x in _modes}
    param = [x for name, x in net.named_parameters()]
    for epoch in range(args.epoch_start, args.epochs):
        loss_per_epoch = {x: 0 for x in ['Loss', 'rec', 'KLX', 'KLM']}
        mse_per_epoch = {x: 0 for x in _modes}
        grad_norm = 0
        tic = time.time()
        # train stage
        net.train()
        lr = optimizer.param_groups[0]['lr']
        if lr < _lr_min:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt, sigmaMapEst, sigmaMapGt = [x.cuda() for x in data]
            optimizer.zero_grad()
            im_out,sigmaX,M,sigmaM,E = net(im_noisy)
            loss, rec, kl_x, kl_M = criterion(im_out,sigmaX,M,sigmaM,E, im_noisy, im_gt,args.eps2)
            loss.backward()
            # clip the gradnorm
            total_norm = nn.utils.clip_grad_norm_(param, clip_grad)
            grad_norm = (grad_norm*(ii/(ii+1)) + total_norm/(ii+1))
            optimizer.step()

            loss_per_epoch['Loss'] += loss.item() / num_iter_epoch[phase]
            loss_per_epoch['rec'] += rec.item() / num_iter_epoch[phase]
            loss_per_epoch['KLX'] += kl_x.item() / num_iter_epoch[phase]
            loss_per_epoch['KLM'] += kl_M.item() / num_iter_epoch[phase]
            im_denoise = torch.clamp(im_out.detach().data, 0.0, 1.0)
            mse = F.mse_loss(im_denoise, im_gt)
            mse_per_epoch[phase] += mse
            
            
            if (ii+1) % args.print_freq == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>4d}/{:0>4d}, rec={:.4f}, ' + \
                        'KLX={:.4f}, KLM={:.4f}, lr={:.1e}'
                print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase],
                                         rec.item(), kl_x.item(), kl_M.item(), lr))
                writer.add_scalar('Train Loss Iter', loss.item(), step)
                writer.add_scalar('Train MSE Iter', mse, step)
                writer.add_scalar('Gradient Norm Iter', total_norm, step)

                step += 1
                #plot
                show1 = im_gt.cpu().detach().numpy()[0].transpose((1,2,0))
                #print(show1)
                show2 = im_noisy.cpu().detach().numpy()[0].transpose((1,2,0))
                show3 = im_denoise.cpu().detach().numpy()[0].transpose((1,2,0))
                #show4 = phi_sigma.cpu().detach().numpy()[0]
                #show5 = sigmaMapGt.cpu().detach().numpy()[0]
                #show6 = sigmaMapEst.cpu().detach().numpy()[0]
                toshow1 = np.hstack((show1,show2,show3))
                #print(toshow1.shape)
                #toshow2 = np.hstack((show4,show5,show6))
                #toshow = np.vstack((toshow1,toshow2))
                imshow(toshow1)
                
            if (ii+1) % (20*args.print_freq) == 0:

                x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                writer.add_image(phase+' Denoised images', x1, step_img[phase])
                x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                x3 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                writer.add_image(phase+' Noisy Image', x3, step_img[phase])
                step_img[phase] += 1

        mse_per_epoch[phase] /= (ii+1)
        log_str ='{:s}: Loss={:+.2e}, rec={:+.2e}, KLX={:+.2e}, KLM={:+.2e}, mse={:.3e}, ' + \
                                                      'GNorm_D={:.1e}/{:.1e}'
        print(log_str.format(phase, loss_per_epoch['Loss'], loss_per_epoch['rec'],
                                loss_per_epoch['KLX'], loss_per_epoch['KLM'], mse_per_epoch[phase],
                                                clip_grad, grad_norm))
        writer.add_scalar('Loss_epoch', loss_per_epoch['Loss'], epoch)
        writer.add_scalar('Mean Grad Norm epoch', grad_norm, epoch)
        clip_grad = min(clip_grad, grad_norm)
        print('-'*150)

        # test stage
        net.eval()
        psnr_per_epoch = {x: 0 for x in _modes[1:]}
        ssim_per_epoch = {x: 0 for x in _modes[1:]}
        for phase in _modes[1:]:
            for ii, data in enumerate(data_loader[phase]):
                im_noisy, im_gt = [x.cuda() for x in data]
                with torch.set_grad_enabled(False):
                    im_out,sigmaX,M,sigmaM,E = net(im_noisy)

                im_denoise = torch.clamp(im_out[:, :C, ].detach().data, 0.0, 1.0)
                mse = F.mse_loss(im_denoise, im_gt)
                mse_per_epoch[phase] += mse
                psnr_iter = batch_PSNR(im_denoise, im_gt)
                ssim_iter = batch_SSIM(im_denoise, im_gt)
                psnr_per_epoch[phase] += psnr_iter
                ssim_per_epoch[phase] += ssim_iter
                # print statistics every log_interval mini_batches
                if (ii+1) % 20 == 0:
                    log_str = '[Epoch:{:>3d}/{:<3d}] {:s}:{:0>5d}/{:0>5d}, mse={:.2e}, ' + \
                        'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch+1, args.epochs, phase, ii+1, num_iter_epoch[phase],
                                                                         mse, psnr_iter, ssim_iter))
                # tensorboardX summary
                    x1 = vutils.make_grid(im_denoise, normalize=True, scale_each=True)
                    writer.add_image(phase+' Denoised images', x1, step_img[phase])
                    x2 = vutils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase+' GroundTruth', x2, step_img[phase])
                    x3 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
                    writer.add_image(phase+' Noise Image', x3, step_img[phase])
                    step_img[phase] += 1

            psnr_per_epoch[phase] /= (ii+1)
            ssim_per_epoch[phase] /= (ii+1)
            mse_per_epoch[phase] /= (ii+1)
            log_str = '{:s}: mse={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'
            print(log_str.format(phase, mse_per_epoch[phase], psnr_per_epoch[phase],
                                 ssim_per_epoch[phase]))
            print('-'*90)

        # adjust the learning rate
        lr_scheduler.step()
        # save model
        if (epoch+1) % args.save_model_freq == 0 or epoch+1 == args.epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(args.model_dir, model_prefix+str(epoch+1))
            torch.save({
                'epoch': epoch+1,
                'step': step+1,
                'step_img': {x: step_img[x] for x in _modes},
                'grad_norm': clip_grad,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_state = os.path.join(args.model_dir, model_state_prefix+str(epoch+1))
            torch.save(net.state_dict(), save_path_model_state)

        writer.add_scalars('MSE_epoch', mse_per_epoch, epoch)
        writer.add_scalars('PSNR_epoch_test', psnr_per_epoch, epoch)
        writer.add_scalars('SSIM_epoch_test', ssim_per_epoch, epoch)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Reach the maximal epochs! Finish training')

def main():
    # build the model
    net = VCSC.VCSC(args.chn, activation=args.activation, act_init=args.relu_init, wf=args.wf,
                                                                            batch_norm=args.bn_UNet)
    # move the model to GPU
    net = nn.DataParallel(net).cuda()

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    print('\nStepLR with gamma={:.2f}, step size={:d}'.format(args.gamma, args.step_size))
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> Loading checkpoint {:s}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            args.step = checkpoint['step']
            args.step_img = checkpoint['step_img']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            net.load_state_dict(checkpoint['model_state_dict'])
            args.clip_grad_D = checkpoint['grad_norm_D']
            args.clip_grad_S = checkpoint['grad_norm_S']
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args.resume, checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args.epoch_start = 0
        if os.path.isdir(args.log_dir):
            shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)
        if os.path.isdir(args.model_dir):
            shutil.rmtree(args.model_dir)
        os.makedirs(args.model_dir)

    # print the arg pamameters
    for arg in vars(args):
        print('{:<15s}: {:s}'.format(arg,  str(getattr(args, arg))))

    # making traing data
    simulate_dir = Path(args.simulate_dir)
    train_im_list = list(simulate_dir.glob('*.jpg')) + list(simulate_dir.glob('*.png')) + \
                                                                    list(simulate_dir.glob('*.bmp'))
    train_im_list = sorted([str(x) for x in train_im_list])
    # making tesing data
    if args.chn == 3:
        test_case1_h5 = Path('test_data').joinpath('noise_niid_Color', 'CBSD68_niid_case1.hdf5')
        test_case2_h5 = Path('test_data').joinpath('noise_niid_Color', 'CBSD68_niid_case2.hdf5')
        test_case3_h5 = Path('test_data').joinpath('noise_niid_Color', 'CBSD68_niid_case3.hdf5')
    elif args.chn == 1:
        test_case1_h5 = Path('test_data').joinpath('noise_niid_Gray', 'CBSD68_niid_case1.hdf5')
        test_case2_h5 = Path('test_data').joinpath('noise_niid_Gray', 'CBSD68_niid_case2.hdf5')
        test_case3_h5 = Path('test_data').joinpath('noise_niid_Gray', 'CBSD68_niid_case3.hdf5')
    test_im_list = (Path('test_data') / 'CBSD68').glob('*.png')
    test_im_list = sorted([str(x) for x in test_im_list])
    datasets = {'train':DenoisingDatasets.SimulateTrain(train_im_list, 5000*args.batch_size,
        args.patch_size, radius=5, noise_type=args.noise, noise_estimate=True, chn=args.chn),
          'test_cbsd681':DenoisingDatasets.SimulateTest(test_im_list, test_case1_h5, chn=args.chn),
          'test_cbsd682': DenoisingDatasets.SimulateTest(test_im_list, test_case2_h5, chn=args.chn),
          'test_cbsd683': DenoisingDatasets.SimulateTest(test_im_list, test_case3_h5, chn=args.chn)}
    # train model
    print('\nBegin training with GPU: ' + str(args.gpu_id))
    train_model(net, datasets, optimizer, scheduler, loss_fn)

if __name__ == '__main__':
    main()

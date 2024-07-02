# -*- coding: utf-8 -*-
'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from utils_gdsr import DRSRH5Dataset, DRSRDataset, save_param, output_img
from model import DCTNet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
import os
import sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 

import numpy as np
from metrics import Rmse
import warnings
 
warnings.filterwarnings('ignore') 

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_str = 'DCTNet'
scale = 4

# . Set the hyper-parameters for training
num_epochs = 100
lr = 1e-3
weight_decay = 0
batch_size = 4
n_layer = 3 
n_feat = 64
patch_size = 256
depth_channels = 1
rgb_channels = 3
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
clip_grad_norm_value = 1
optim_step = 100
optim_gamma = 0.5
dataset_name = 'RGBDD'

# . Get your model 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = nn.DataParallel(DCTNet()).to(device)
net.load_state_dict(torch.load('/root/autodl-tmp/GDSR-DCTNet/models/DCTNet_4X.pth'), strict=True)  #

print(net)


# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=optim_step, gamma=optim_gamma)
Loss_l2 = nn.MSELoss() #MSELoss


# . Create your data loaders
    
trainloader = DataLoader(DRSRDataset('/root/autodl-tmp/dataset', scale, 'RGBDD', RGB2Y=False),
                              batch_size=batch_size,
                              num_workers=4)
validationloader = DataLoader(DRSRDataset('/root/autodl-tmp/dataset', scale, 'RGBDD', RGB2Y=False),
                              batch_size=batch_size,
                              num_workers=4)
testloader = DataLoader(DRSRDataset('/root/autodl-tmp/dataset', scale, 'RGBDD', RGB2Y=False),
                        batch_size=batch_size,
                        num_workers=4)

loader = {'train':      trainloader,
          'validation': validationloader}

# . Creat logger
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
save_path = os.path.join(
    'logs/%s' % (model_str),
    timestamp +
    '_scale%d_layer%d_filter%d_epochs%d_batch%d_lr%s(%s-%s)_grad%s_%s' % (
        scale, n_layer, n_feat, num_epochs, batch_size, lr, optim_step, optim_gamma, clip_grad_norm_value, GPU_number)
)
writer = SummaryWriter(save_path)
params = {'model': model_str,
          'scale': scale,
          'epoch': num_epochs,
          'lr': lr,
          'batch_size': batch_size,
          'n_feat': n_feat,
          'n_layer': n_layer,
          'clip_grad_norm_value': clip_grad_norm_value,
          'optim_step': optim_step,
          'optim_gamma': optim_gamma,
          'GPU_number': GPU_number,
          }

# save_param(params,
#            os.path.join(save_path, 'param.json'))


'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''


step = 0
current_rmse_val, rmse_val = 0., 0.
best_rmse_val = 100.

torch.backends.cudnn.benchmark = True
prev_time = time.time()


for epoch in range(num_epochs):
    ''' train '''
    for i, (Depth, RGB, gt,_,_) in enumerate(loader['train']):
        # 0. preprocess data
        Depth, RGB, gt = Depth.to(device), RGB.to(device), gt.to(device)
        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        imgf_raw = net(Depth, RGB)
        loss = Loss_l2(gt, imgf_raw)
        loss.backward()
        nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer.step() 
        
        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

        # 3. Log the scalar values
        writer.add_scalar('loss/1 Loss', loss.item(), step)
        writer.add_scalar('loss/2 learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step+=1
        
    # 4. adjust the learning rate
    scheduler.step()
    if optimizer.param_groups[0]['lr']<=1e-6:
        optimizer.param_groups[0]['lr']=1e-6

   
    # Save the current weight
    torch.save({'net':net.state_dict(),
                'optimizer':optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'epoch': epoch,
                'step': step},
                os.path.join(save_path, 'last_net.pth'))

    # Save the weight per 10 epoch
    if epoch % 10 == 0:
        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_schedule': scheduler.state_dict(),
            "epoch": epoch, 
            'step': step            
            }
        torch.save(
            checkpoint, os.path.join(
                save_path, 'ckpt_checkpoint_%s.pth' %(str(epoch))))
    
'''
------------------------------------------------------------------------------
Test
------------------------------------------------------------------------------
'''


# 1. Load the best weight and create the dataloader for testing
net.load_state_dict(torch.load(os.path.join(save_path, 'last_net.pth'))['net'])

# 2. Compute the metrics
metrics = torch.zeros(1, testloader.__len__())
with torch.no_grad():
    net.eval()
    for i, (Depth, RGB, gt, D_min, D_max) in enumerate(testloader):
        Depth, RGB, gt, D_min, D_max = Depth.to(device
        ), RGB.to(device), gt.to(device), D_min.to(device), D_max.to(device)
        imgf_raw = net(Depth, RGB).clamp(min=0, max=1)
        imgf = (imgf_raw * (D_max - D_min)[:,None,None,None]) + D_min[:,None,None,None]

        if dataset_name == 'Middlebury' and dataset_name == 'Lu':
            imgf2image = output_img(imgf).clip(min=0, max=255)
            gt2image = output_img(gt).clip(min=0, max=255)
        elif dataset_name == 'NYU':
            # clip borders (reference: https://github.com/cvlab-yonsei/dkn/issues/1)
            # Following DKN, FDSR
            imgf2image = output_img(imgf)[6:-6, 6:-6]
            gt2image = output_img(gt)[6:-6, 6:-6]
        else:
            imgf2image = output_img(imgf)
            gt2image = output_img(gt)
        metrics[:, i] = Rmse(imgf2image, gt2image)

Final_result = metrics.mean(dim=1).item()
print('\nThe RMSE value in testing is %f'%Final_result)
metrics_ = metrics.numpy()
np.save(os.path.join(save_path, 'test_result'+ dataset_name + '.npy'), metrics_)

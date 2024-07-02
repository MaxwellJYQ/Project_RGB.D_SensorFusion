# -*- coding: utf-8 -*-

from model import DCTNet
from torch.utils.data import DataLoader
import warnings
from metrics import Rmse
import numpy as np
from scipy.io import savemat
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from utils_gdsr import DRSRH5Dataset, DRSRDataset, save_param, output_img
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.filterwarnings('ignore') 
import cv2
def inference_net_eachDataset(dataset_name, net_Path, scale):
    start = time.time()
    # . Get your model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = nn.DataParallel(DCTNet()).to(device)

    # # 1. Load the best weight and create the dataloader for testing
    testloader = DataLoader(DRSRDataset('/root/autodl-tmp/dataset', scale, 'RGBDD', RGB2Y=False, test_flag=True),
                            batch_size=1,
                            num_workers=0)
    net.load_state_dict(torch.load('/root/autodl-tmp/GDSR-DCTNet/logs/DCTNet/06-05-22-56_scale4_layer3_filter64_epochs100_batch4_lr0.001(100-0.5)_grad1_0/ckpt_checkpoint_50.pth',map_location='cpu')['net'])
   
    # net.load_state_dict(torch.load('/root/autodl-tmp/GDSR-DCTNet/models/DCTNet_4X.pth'))

    # 2. Compute the metrics
    metrics = torch.zeros(1, testloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (Depth, RGB, gt, D_min, D_max) in tqdm(enumerate(testloader)):
            Depth, RGB, gt, D_min, D_max = Depth.cuda(
            ), RGB.cuda(), gt.cuda(), D_min.cuda(), D_max.cuda()
            imgf_raw = net(Depth, RGB)
            imgf = imgf_raw 
            imgf2image = output_img(imgf)
            gt2image = output_img(gt)
            metrics[:, i] = Rmse(imgf2image, gt2image)
    end = time.time()
    return metrics.mean(dim=1)


def infrence_all_datasets(net_Path, scale):
    Rmses = inference_net_eachDataset('RGBDD', net_Path, scale)
    return Rmses
def test():
    '''Calculate RMSE value'''
    print("RMSE: %f"%infrence_all_datasets("1",4).item())

    
test()

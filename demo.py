# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_gdsr import DRSRDataset, output_img
from model import DCTNet
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def reverse_preprocess_rgb(image):
    # Ensure the image is in the shape [3, H, W] before processing
    if image.shape[0] == 3:
        image = image * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    else:
        raise ValueError("Expected image with shape [3, H, W], but got shape {}".format(image.shape))
    image = image * 255
    return image

def save_images(rgb, depth, gt, prediction, index, output_dir):
    # Convert tensors to numpy arrays and move to CPU
    rgb = rgb.squeeze().cpu().numpy()
    depth = depth.squeeze().cpu().numpy()
    gt = gt.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()
    
    # Reverse preprocess RGB for display
    rgb = reverse_preprocess_rgb(rgb)
    rgb = rgb.transpose(1, 2, 0)  # Change shape from [3, H, W] to [H, W, 3]
    
    # Normalize other images for display
    depth = (depth / np.max(depth) * 255).astype(np.uint8)
    gt = (gt / np.max(gt) * 255).astype(np.uint8)
    prediction = (prediction / np.max(prediction) * 255).astype(np.uint8)
    
    rgb = rgb.astype(np.uint8)
    
    # Create a canvas to display images
    height, width = depth.shape
    canvas = np.zeros((height, width * 4, 3), dtype=np.uint8)
    
    canvas[:, :width, :] = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
    canvas[:, width:width * 2, :] = rgb
    canvas[:, width * 2:width * 3, :] = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    canvas[:, width * 3:, :] = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
    
    # Add text annotations
    cv2.putText(canvas, 'Depth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'RGB', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'GT', (width * 2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Prediction', (width * 3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the canvas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, f'inference_{index}.png'), canvas)

def inference_net_eachDataset(dataset_name, net_Path, scale, output_dir):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = nn.DataParallel(DCTNet()).to(device)

    testloader = DataLoader(DRSRDataset('/root/autodl-tmp/dataset', scale, 'RGBDD', RGB2Y=False, test_flag=True),
                            batch_size=1,
                            num_workers=0)
    net.load_state_dict(torch.load(net_Path, map_location=device)['net'])

    with torch.no_grad():
        net.eval()
        for i, (Depth, RGB, gt, D_min, D_max) in tqdm(enumerate(testloader)):
            Depth, RGB, gt, D_min, D_max = Depth.to(device), RGB.to(device), gt.to(device), D_min.to(device), D_max.to(device)
            imgf_raw = net(Depth, RGB)
            imgf = imgf_raw
            
            save_images(RGB, Depth, gt, imgf, i, output_dir)
            
    end = time.time()

def infrence_all_datasets(net_Path, scale, output_dir):
    inference_net_eachDataset('RGBDD', net_Path, scale, output_dir)

def test():
    '''Visualize inference results and save the images'''
    output_dir = './inference_results2'
    infrence_all_datasets('/root/autodl-tmp/GDSR-DCTNet/logs/DCTNet/06-05-22-56_scale4_layer3_filter64_epochs100_batch4_lr0.001(100-0.5)_grad1_0/ckpt_checkpoint_50.pth', 4, output_dir)

test()

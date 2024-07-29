from modules import setup

setup.seed_everything()

import argparse
import os
import importlib

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import matplotlib.pyplot as plt
import cv2

from pytorch_msssim import ssim

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils
from configs import CONFIGS

models = importlib.reload(models)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, required=True)
args = parser.parse_args()

curr_config = CONFIGS[args.config_name]

if __name__ == '__main__':
    utils.log('Starting SISR experiment')
    
    weight_init = False
    tvl = curr_config["tvl"]  # Total variation loss
    
    # Results
    mdict = {}  
    metrics = {}  

    # Image pre-processing 
    scale = curr_config["down_scale"]  # Downsampling factor
    scale_im = 1 / 3  # Initial image downsample 

    # Activation constants
    omega0 = 8.0  
    sigma0 = curr_config["scale"]  
    scale_tensor = torch.tensor(curr_config["scale_tensor"]).cuda()
    learning_rate = curr_config["learning_rate"]

    # Network parameters
    nonlin = curr_config["nonlin"]
    niters = curr_config["niters"]
    hidden_features = curr_config["hidden_features"]    
    scaled_hidden_features = curr_config["scaled_hidden_features"]  
    hidden_layers = 2  
    if nonlin == "bspline_mscale_1_new":
        in_features = 2 * len(scale_tensor) * scaled_hidden_features
    else:
        in_features = 2

    # Read image
    im = utils.normalize(
        plt.imread(
            '/rds/general/user/atk23/home/wire/data/butterfly.png').astype(
                np.float32), True)

    im = cv2.resize(im,
                    None,
                    fx=scale_im,
                    fy=scale_im,
                    interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape

    # Ensures image dimensions are multiples of scale
    im = im[:scale * (H // scale), :scale * (W // scale), :]
    H, W, _ = im.shape

    im_lr = cv2.resize(im,
                       None,
                       fx=1 / scale,
                       fy=1 / scale,
                       interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape

    # Low-resolution image
    x = torch.linspace(-1, 1, W2).cuda()
    y = torch.linspace(-1, 1, H2).cuda()
    # High-resolution image
    x_hr = torch.linspace(-1, 1, W).cuda()
    y_hr = torch.linspace(-1, 1, H).cuda()

    im_bi = cv2.resize(im_lr,
                       None,
                       fx=scale,
                       fy=scale,
                       interpolation=cv2.INTER_LINEAR)

    utils.log('System Information')
    utils.log(f'Non-linearity: {nonlin}, Learning rate: {learning_rate}, Scale: {sigma0}')
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        sidelength = int(max(H, W))
    else:
        posencode = False
        sidelength = H

    model = models.get_INR(nonlin=nonlin,
                        in_features=in_features,
                        out_features=3,
                        hidden_features=hidden_features,
                        scaled_hidden_features=scaled_hidden_features,
                        hidden_layers=hidden_layers,
                        first_omega_0=omega0,
                        hidden_omega_0=omega0,
                        scale=sigma0,
                        scale_tensor=scale_tensor,
                        pos_encode=posencode,
                        sidelength=sidelength)

    # Send model to CUDA
    model.cuda()

    # Create an optimizer
    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x / niters, 1))

    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack(
        (X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None, ...]

    gt = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]
    gt_lr = torch.tensor(im_lr).cuda().reshape(H2 * W2, 3)[None, ...]

    im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
    im_bi_ten = torch.tensor(im_bi).cuda().permute(2, 0, 1)[None, ...]

    mse_array = torch.zeros(niters, device='cuda')
    ssim_array = torch.zeros(niters, device='cuda')
    lpips_array = torch.zeros(niters, device='cuda')

    best_mse = float('inf')
    best_img = None

    downsampler = torch.nn.AvgPool2d(scale)

    tbar = range(niters)
    for epoch in tbar:
        rec_hr = model(coords_hr)
        rec = downsampler(
            rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...])

        loss = ((gt_lr - rec.reshape(1, 3, -1).permute(0, 2, 1))**2).mean()

        with torch.no_grad():
            rec_hr = model(coords_hr)

            im_rec = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

            mse_array[epoch] = ((gt - rec_hr)**2).mean().item()
            ssim_array[epoch] = ssim(im_gt,
                                    im_rec,
                                    data_range=1,
                                    size_average=True)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()

        if mse_array[epoch] < best_mse:
            best_mse = mse_array[epoch]
            best_img = imrec

    if posencode:
        nonlin = 'posenc'

    utils.log(f'Best MSE: {-10 * torch.log10(best_mse).item()}')
    utils.log(f'Best SSIM: {ssim_func(im, best_img, multichannel=True)}')

    folder_name = utils.make_unique(
        f"{curr_config['name']}",
        f"/rds/general/user/atk23/home/wire/multiscale_results/sisr/DS_{scale}")

    mdict[folder_name] = {
        'Scale': sigma0,
        'rec': best_img,
        'gt': im,
        'rec_bi': im_bi,
        'mse_array': mse_array.detach().cpu().numpy(),
        'ssim_array': mse_array.detach().cpu().numpy(),
    }

    metrics[folder_name] = {
        'Scale': sigma0,
        'Scale Tensor': scale_tensor,
        'Downscale': scale,
        'Learning rate': learning_rate,
        'Best MSE': -10 * torch.log10(best_mse).item(),
        'Best SSIM': ssim_func(im, best_img, multichannel=True)
    }
    
    filepath = f"/rds/general/user/atk23/home/wire/multiscale_results/sisr/DS_{scale}/{folder_name}"
    os.makedirs(filepath, exist_ok=True)
    
    plt.imsave(os.path.join(filepath, 'MSE_plot.png'), np.clip(abs(im-best_img), 0, 1),
    vmin=0.0, vmax=0.1)

    io.savemat(os.path.join(filepath, 'info.mat'), mdict)
    io.savemat(os.path.join(filepath, 'metrics.mat'), metrics)
    utils.tabulate_results(os.path.join(filepath, 'metrics.mat'), filepath)
    utils.display_image(os.path.join(filepath, 'info.mat'))

    utils.log('Finished SISR experiment')

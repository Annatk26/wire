from modules import setup

setup.seed_everything()

import argparse
import os
from scipy import io

import numpy as np

import cv2
import matplotlib.pyplot as plt
plt.gray()

from skimage.metrics import structural_similarity as ssim_func

import torch
from torch.optim.lr_scheduler import LambdaLR

from modules import models
from modules import utils
from modules import lin_inverse
from configs import CONFIGS

if __name__ == '__main__':
    utils.log('Starting CT experiment')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    curr_config = CONFIGS[args.config_name]

    # Results
    mdict = {}  
    metrics = {}  

    # Noise is not used in this script, but you can do so by modifying line 82 below
    added_noise = curr_config['added_noise']  # Add noise to measurements
    tau = curr_config['tau']  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = curr_config['noise_snr']  # Readout noise (dB)
    nmeas = 100  # Number of CT measurement

    weight_init = False
    tvl = curr_config["tvl"]  # Total variation loss

    # Activation function constants.
    omega0 = 3.0 
    scale_tensor = torch.tensor(curr_config["scale_tensor"]).cuda() 
    sigma0 = curr_config["scale"]  
    learning_rate = curr_config["learning_rate"]

    # Network parameters
    hidden_layers = 2  
    hidden_features = 300  # TODO: Number of hidden units per layer (300)
    scaled_hidden_features = curr_config["scaled_hidden_features"]
    nonlin = curr_config["nonlin"]
    niters = curr_config["niters"]
    if nonlin == "bspline_mscale_1_new":
        in_features = 2 * len(scale_tensor) * scaled_hidden_features
    else:
        in_features = 2

    # Generate sampling angles
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).cuda()
    scale_im = 1 / 2  # Initial image downsample for memory reasons

    # Create phantom
    img = cv2.imread(
        '/rds/general/user/atk23/home/wire/data/chest.png').astype(
            np.float32)[..., 1]
    img = utils.normalize(img, True)
    img = cv2.resize(img,
                None,
                fx=scale_im,
                fy=scale_im,
                interpolation=cv2.INTER_AREA)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].cuda()

    utils.log('System Information')
    utils.log(f'Non-linearity: {nonlin}, Learning Rate: {learning_rate}, Scale: {sigma0}') 
    if nonlin == 'wire':
        omega0 = 3.0
    elif nonlin == 'siren':
        omega0 = 12.0
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

    model = models.get_INR(nonlin=nonlin,
                        in_features=in_features,
                        out_features=1,
                        hidden_features=hidden_features,
                        hidden_layers=hidden_layers,
                        scaled_hidden_features=scaled_hidden_features,
                        first_omega_0=omega0,
                        hidden_omega_0=omega0,
                        scale=sigma0,
                        scale_tensor=scale_tensor,
                        pos_encode=posencode,
                        sidelength=nmeas)

    model = model.cuda()

    with torch.no_grad():
        sinogram = lin_inverse.radon(imten, thetas).detach().cpu()
        sinogram = sinogram.numpy()
        sinogram_noisy = utils.measure(sinogram, noise_snr,
                                    tau).astype(np.float32)
        # Set below to sinogram_noisy instead of sinogram to get noise in measurements
        if added_noise:
            sinogram = sinogram_noisy
        sinogram_ten = torch.tensor(sinogram).cuda()

    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()
    X, Y = torch.meshgrid(x, y, indexing='xy')

    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    if isinstance(learning_rate, list):
        param_groups = []
        for i, stage in enumerate(model.stages):
            param_groups.append({
                'params': stage.parameters(),
                'lr': learning_rate[i]
            })
            param_groups.append({
                'params': model.linears[i].parameters(),
                'lr': learning_rate[i]
            })
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.Adam(lr=learning_rate,
                                    params=model.parameters())

    # Schedule to 0.1 times the initial rate
    scheduler = LambdaLR(optimizer, lambda x: 0.1**min(x / niters, 1))

    best_loss = float('inf')
    loss_array = np.zeros(niters)
    best_im = None

    tbar = range(niters)
    for idx in tbar:
        # Estimate image
        img_estim = model(coords).reshape(-1, H, W)[None, ...]

        # Compute sinogram
        sinogram_estim = lin_inverse.radon(img_estim, thetas)

        loss = ((sinogram_ten - sinogram_estim)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            img_estim_cpu = img_estim.detach().cpu().squeeze().numpy()

            loss_gt = ((img_estim - imten)**2).mean()
            loss_array[idx] = loss_gt.item()

            if loss_gt < best_loss:
                best_loss = loss_gt
                best_im = img_estim

    img_estim_cpu = best_im.detach().cpu().squeeze().numpy()

    psnr2 = utils.psnr(img, img_estim_cpu)
    ssim2 = ssim_func(img, img_estim_cpu)
    utils.log(f'Best PSNR: {psnr2}')
    utils.log(f'Best SSIM: {ssim2}')

    if posencode:
        nonlin = "posenc"

    folder_name = utils.make_unique(
        f"{curr_config['name']}", "/rds/general/user/atk23/home/wire/multiscale_results/ct")

    mdict[folder_name] = {
        'Scale': sigma0,
        'rec': img_estim_cpu,
        'loss_array': loss_array,
        'sinogram': sinogram,
        'gt': img,
    }
    metrics[folder_name] = {
        'Scale': sigma0,
        'Scale Tensor': curr_config["scale_tensor"],
        'Learning Rate': learning_rate,
        'Best PSNR': psnr2,
        'Best SSIM': ssim2
    }

    os.makedirs(
        f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}",
        exist_ok=True)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}/info.mat",
        mdict)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}/metrics.mat",
        metrics)
    utils.tabulate_results(
        f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}/metrics.mat",
        f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}")
    utils.display_image(
    f"/rds/general/user/atk23/home/wire/multiscale_results/ct/{folder_name}/info.mat"
    )
    utils.log('CT experiment completed')

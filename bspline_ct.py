from modules import setup

setup.seed_everything()

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

if __name__ == '__main__':
    utils.log('Starting CT experiment with Multi-scale B-Spline (form)')
    nonlin = 'bspline_form'
    niters = 5000  # Number of SGD iterations

    #learning_rate = 5e-3        # Learning rate.

    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity
    nmeas = 100  # Number of CT measurement

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    # Noise is not used in this script, but you can do so by modifying line 82 below
    tau = 3e1  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    omega0 = 3.0  # Frequency of sinusoid
    sigma0 = 12.0  # Sigma of Gaussian (12.0)
    scale_tensor = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    learning_rate = 4e-3
    mutliscale_all = [True, False]
    weight_init = False

    utils.log(f'Scale: {sigma0}')
    utils.log(f'Scale tensor: {scale_tensor}')
    utils.log(f'{nonlin} Learning rate: {learning_rate}')

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = 300  # Number of hidden units per layer
    scaled_hidden_features = 32  # Number of hidden units in the first layer

    # Generate sampling angles
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).cuda()

    # Create phantom
    img = cv2.imread(
        '/rds/general/user/atk23/home/wire/data/chest.png').astype(
            np.float32)[..., 1]
    img = utils.normalize(img, True)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].cuda()

    # Create model
    for multiscale in mutliscale_all:
        utils.log(f'Multiscale: {multiscale}')
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
                               in_features=2,
                               out_features=1,
                               hidden_features=hidden_features,
                               hidden_layers=hidden_layers,
                               scaled_hidden_features=scaled_hidden_features,
                               first_omega_0=omega0,
                               hidden_omega_0=omega0,
                               scale=sigma0,
                               scale_tensor=scale_tensor,
                               pos_encode=posencode,
                               multi_scale=multiscale,
                               sidelength=nmeas)

        model = model.cuda()

        with torch.no_grad():
            sinogram = lin_inverse.radon(imten, thetas).detach().cpu()
            sinogram = sinogram.numpy()
            sinogram_noisy = utils.measure(sinogram, noise_snr,
                                           tau).astype(np.float32)
            # Set below to sinogram_noisy instead of sinogram to get noise in measurements
            sinogram_ten = torch.tensor(sinogram).cuda()

        x = torch.linspace(-1, 1, W).cuda()
        y = torch.linspace(-1, 1, H).cuda()

        X, Y = torch.meshgrid(x, y, indexing='xy')

        coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

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
                #if sys.platform == 'win32':
                #   cv2.imshow('Image', img_estim_cpu)
                #  cv2.waitKey(1)

                loss_gt = ((img_estim - imten)**2).mean()
                loss_array[idx] = loss_gt.item()

                if loss_gt < best_loss:
                    best_loss = loss_gt
                    best_im = img_estim

        img_estim_cpu = best_im.detach().cpu().squeeze().numpy()

        psnr2 = utils.psnr(img, img_estim_cpu)
        ssim2 = ssim_func(img, img_estim_cpu)
        utils.log(f'Best PSNR: {psnr2}, Best SSIM: {ssim2}')
        utils.log(f'Trained Scale: {model.net[1].scale_0.item()}')

        if posencode:
            nonlin = "posenc"
        if multiscale:
            label = "Multiscale"
        else:
            label = "No Multiscale"

        mdict[label] = {
            'Weight initialization': weight_init,
            'Scale': model.net[1].scale_0.item(),
            'rec': img_estim_cpu,
            'loss_array': loss_array,
            'sinogram': sinogram,
            'gt': img,
        }
        metrics[label] = {
            'Weight initialization': weight_init,
            'Scale': model.net[1].scale_0.item(),
            'Learning Rate': learning_rate,
            'Best PSNR': psnr2,
            'Best SSIM': ssim2
        }

    folder_name = utils.make_unique(
        f"form_mscale", "/rds/general/user/atk23/home/wire/bspline_results/ct")
    os.makedirs(
        f"/rds/general/user/atk23/home/wire/bspline_results/ct/{folder_name}",
        exist_ok=True)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/bspline_results/ct/{folder_name}/info.mat",
        mdict)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/bspline_results/ct/{folder_name}/metrics.mat",
        metrics)
    utils.tabulate_results(
        f"/rds/general/user/atk23/home/wire/bspline_results/ct/{folder_name}/metrics.mat",
        f"/rds/general/user/atk23/home/wire/bspline_results/ct/{folder_name}")
    utils.log('CT experiment completed')

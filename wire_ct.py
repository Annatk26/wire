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
    utils.log('Starting CT experiment')
    nonlin_types = [
        'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    ]  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 5000  # Number of SGD iterations
    #learning_rate = 5e-3        # Learning rate.

    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity
    nmeas = 100  # Number of CT measurement
    expected = {
        "Expected PSNR": [32.3, 30.3, 18.1, 0.0, 28.5, 29.2], 
        "Expected SSIM": [0.81, 0.76, 0.23, 0.0, 0.71, 0.73]
    }

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    # Noise is not used in this script, but you can do so by modifying line 82 below
    tau = 3e1  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    omega0 = 15.0  # Frequency of sinusoid
    sigma0 = 12.0  # Sigma of Gaussian
    utils.log(f'Omega0: {omega0}, Sigma0: {sigma0}')
    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = 300  # Number of hidden units per layer

    # Generate sampling angles
    thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).cuda()

    # Create phantom
    img = cv2.imread(
        '/home/atk23/wire/data/chest.png').astype(
            np.float32)[..., 1]
    img = utils.normalize(img, True)
    [H, W] = img.shape
    imten = torch.tensor(img)[None, None, ...].cuda()

    # Create model
    for i, nonlin in enumerate(nonlin_types):
        learning_rate = {
            #"wire2d": 5e-3,
            "wire": 5e-3,
            "siren": 2e-3,
            "mfn": 5e-2,
            "relu": 1e-3,
            "posenc": 1e-3,
            "gauss": 2e-3,
        }[nonlin]
        utils.log(f'{nonlin} learning rate: {learning_rate}')
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
                               first_omega_0=omega0,
                               hidden_omega_0=omega0,
                               scale=sigma0,
                               pos_encode=posencode,
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
        
        if posencode:
            nonlin = "posenc"

        mdict = {
            'rec': img_estim_cpu,
            'loss_array': loss_array,
            'sinogram': sinogram,
            'gt': img,
        }
        metrics[nonlin] = {
            'Omega0': omega0,
            'Sigma0': sigma0,
            'Learning Rate': learning_rate,
            'Best PSNR': psnr2,
            'Best SSIM': ssim2,
            'Expected PSNR': expected['Expected PSNR'][i],
            'Expected SSIM': expected['Expected SSIM'][i],
            'PSNR Difference': abs(psnr2 - expected['Expected PSNR'][i]),
            'SSIM Difference': abs(ssim2 - expected['Expected SSIM'][i]),
        }

    folder_name = utils.make_unique("ct", "/home/atk23/wire/baseline_results/")
    os.makedirs(f"/home/atk23/wire/baseline_results/{folder_name}",
                exist_ok=True)
    io.savemat(f"/home/atk23/wire/baseline_results/{folder_name}/info.mat",
               mdict)
    io.savemat(f"/home/atk23/wire/baseline_results/{folder_name}/metrics.mat", metrics)

    utils.tabulate_results(f"/home/atk23/wire/baseline_results/{folder_name}/metrics.mat")
    
from modules import setup

setup.seed_everything()

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

models = importlib.reload(models)

if __name__ == '__main__':
    utils.log('Starting SISR experiment')
    nonlin_types = [
        'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    ]  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = 2000  # Number of SGD iterations (2000)
    expected = {
        "Expected MSE": [27.3, 25.6, 21.9, 0, 26.3, 26.1],
        "Expected SSIM": [0.93, 0.90, 0.77, 0, 0.92, 0.91]
    }

    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity
    
    folder_name = utils.make_unique("sisr", "/rds/general/user/atk23/home/wire/baseline_results")
    os.makedirs(f"/rds/general/user/atk23/home/wire/baseline_results/{folder_name}",
                exist_ok=True)

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3
    scale = 4  # Downsampling factor
    scale_im = 1 / 3  # Initial image downsample for memory reasons

    # Gabor filter constants
    omega0 = 8.0  # Frequency of sinusoid
    sigma0 = 9.0  # Sigma of Gaussian
    utils.log(f'Omega0: {omega0}, Sigma0: {sigma0}')

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = 256  # Number of hidden units per layer

    # Read image
    im = utils.normalize(
        plt.imread(
            '/rds/general/user/atk23/home/wire/data/butterfly.png').astype(
                np.float32), True)
    #im = im[:1344, :, :]
    im = cv2.resize(im,
                    None,
                    fx=scale_im,
                    fy=scale_im,
                    interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape

    im = im[:scale * (H // scale), :scale * (W // scale), :]
    H, W, _ = im.shape

    im_lr = cv2.resize(im,
                       None,
                       fx=1 / scale,
                       fy=1 / scale,
                       interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape

    x = torch.linspace(-1, 1, W2).cuda()
    y = torch.linspace(-1, 1, H2).cuda()

    x_hr = torch.linspace(-1, 1, W).cuda()
    y_hr = torch.linspace(-1, 1, H).cuda()

    im_bi = cv2.resize(im_lr,
                       None,
                       fx=scale,
                       fy=scale,
                       interpolation=cv2.INTER_LINEAR)

    for i, nonlin in enumerate(nonlin_types):
        learning_rate = {
            'wire': 5e-3,
            'siren': 2e-3,
            'mfn': 5e-2,
            'relu': 1e-3,
            'posenc': 1e-3,
            'gauss': 2e-3
        }[nonlin]
        utils.log(f'{nonlin} learning rate: {learning_rate}')
        if nonlin == 'posenc':
            nonlin = 'relu'
            posencode = True
            sidelength = int(max(H, W))
        else:
            posencode = False
            sidelength = H

        model = models.get_INR(nonlin=nonlin,
                               in_features=2,
                               out_features=3,
                               hidden_features=hidden_features,
                               hidden_layers=hidden_layers,
                               first_omega_0=omega0,
                               hidden_omega_0=omega0,
                               scale=sigma0,
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
            #rec = model(coords)

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

            #if sys.platform == 'win32':
            #   cv2.imshow('Reconstruction', imrec[..., ::-1])
            #  cv2.waitKey(1)

            if mse_array[epoch] < best_mse:
                best_mse = mse_array[epoch]
                best_img = imrec

        if posencode:
            nonlin = 'posenc'

        mdict[nonlin] = {
            'rec': best_img,
            'gt': im,
            'rec_bi': im_bi,
            'mse_array': mse_array.detach().cpu().numpy(),
            'ssim_array': mse_array.detach().cpu().numpy(),
        }
        metrics[nonlin] = {
            'Omega0': omega0,
            'Sigma0': sigma0,
            'Learning rate': learning_rate,
            'Best MSE': -10 * torch.log10(best_mse).item(),
            'Best SSIM': ssim_func(im, best_img, multichannel=True),
            'Expected MSE': expected["Expected MSE"][i],
            'Expected SSIM': expected["Expected SSIM"][i],
            'MSE Difference': abs((-10 * torch.log10(best_mse).item()) - expected["Expected MSE"][i]),
            'SSIM Difference': abs(ssim_func(im, best_img, multichannel=True) - expected["Expected SSIM"][i])
            #'lpips': lpips_array.min().item()
        }

        # plt.plot(-10 * torch.log10(mse_array).detach().cpu().numpy())
        # plt.xlabel('Iterations')
        # plt.ylabel('MSE')
        # plt.title('MSE vs Iterations')
        # plt.grid(True)
        # plt.show()
        plt.imsave(
            f'/rds/general/user/atk23/home/wire/baseline_results/{folder_name}/MSE_{nonlin}.png',
            np.clip(abs(im - best_img), 0, 1),
            vmin=0.0,
            vmax=0.1)

   
    io.savemat(f"/rds/general/user/atk23/home/wire/baseline_results/{folder_name}/info.mat",
               mdict)
    io.savemat(f"/rds/general/user/atk23/home/wire/baseline_results/{folder_name}/metrics.mat", metrics)

    utils.tabulate_results(f"/rds/general/user/atk23/home/wire/baseline_results/{folder_name}/metrics.mat", f"/rds/general/user/atk23/home/wire/baseline_results/{folder_name}")
    utils.log('Finished SISR experiment')
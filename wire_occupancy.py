from modules import setup

setup.seed_everything()

import os
import time
import copy

import numpy as np
from scipy import io
from scipy import ndimage

import torch
from torch.optim.lr_scheduler import LambdaLR

import matplotlib.pyplot as plt

from modules import models
from modules import utils
from modules import volutils

if __name__ == '__main__':
    plt.gray()
    nonlin_types = [
        'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    ]  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    mdict = {}  # Dictionary to store results of each non-linearity
    # Save data
    os.makedirs('/rds/general/user/atk23/home/wire/results/occupancy',
                exist_ok=True)

    niters = 200  # Number of SGD iterations (initial: 200)
    expname = 'thai_statue'  # Volume to load
    scale = 1.0  # Run at lower scales to testing (initial: 1)
    mcubes_thres = 0.5  # Threshold for marching cubes

    # Gabor filter constants
    # These settings work best for 3D occupancies
    #omega0 = 10.0  # Frequency of sinusoid
    #sigma0 = 40.0  # Sigma of Gaussian

    # Network constants
    hidden_layers = 3  # Number of hidden layers in the mlp
    hidden_features = 300  # Number of hidden units per layer
    maxpoints = int(2e5)  # Batch size (inital: 2e5)

    if expname == 'thai_statue':
        occupancy = True
    else:
        occupancy = False

    # Load image and scale
    im = io.loadmat(f'/rds/general/user/atk23/home/wire/data/{expname}.mat'
                    )['hypercube'].astype(np.float32)
    im = ndimage.zoom(im / im.max(), [scale, scale, scale], order=0)

    # If the volume is an occupancy, clip to tightest bounding box
    if occupancy:
        hidx, widx, tidx = np.where(im > 0.99)
        im = im[hidx.min():hidx.max(),
                widx.min():widx.max(),
                tidx.min():tidx.max()]

    utils.log(im.shape)
    H, W, T = im.shape

    maxpoints = min(H * W * T, maxpoints)

    imten = torch.tensor(im).cuda().reshape(H * W * T, 1)

    # Create inputs
    coords = utils.get_coords(H, W, T)

    for nonlin in nonlin_types:
        # Set learning rate based on nonlinearity
        learning_rate = {
            'wire': 5e-3,
            'siren': 2e-3,
            'mfn': 5e-2,
            'relu': 1e-3,
            'posenc': 1e-3,
            'gauss': 2e-3
        }[nonlin]
        
        # omega0 = 20.0, sigma0 = 10.0 for wire
        # omega0 = 40.0 for siren
        # sigma0 = 30.0 for gauss
        # else omega0 = 10.0, sigma0 = 40.0
        if nonlin == 'wire':
            omega0 = 20.0
            sigma0 = 10.0
        elif nonlin == 'siren':
            omega0 = 40.0
        elif nonlin == 'gauss':
            sigma0 = 30.0
        else:
            omega0 = 10.0
            sigma0 = 40.0

        if nonlin == 'posenc':
            nonlin = 'relu'
            posencode = True
        else:
            posencode = False

        # Create model
        model = models.get_INR(nonlin=nonlin,
                               in_features=3,
                               out_features=1,
                               hidden_features=hidden_features,
                               hidden_layers=hidden_layers,
                               first_omega_0=omega0,
                               hidden_omega_0=omega0,
                               scale=sigma0,
                               pos_encode=posencode,
                               sidelength=max(H, W, T)).cuda()

        # Optimizer
        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())

        # Schedule to 0.1 times the initial rate
        scheduler = LambdaLR(optim, lambda x: 0.2**min(x / niters, 1))
        criterion = torch.nn.MSELoss()

        mse_array = np.zeros(niters)
        time_array = np.zeros(niters)
        best_mse = float('inf')
        best_img = None

        tbar = range(niters)

        im_estim = torch.zeros((H * W * T, 1), device='cuda')

        tic = time.time()
        utils.log(f'Running {nonlin} nonlinearity')
        for idx in tbar:
            indices = torch.randperm(H * W * T)

            train_loss = 0
            nchunks = 0
            for b_idx in range(0, H * W * T, maxpoints):
                b_indices = indices[b_idx:min(H * W * T, b_idx + maxpoints)]
                b_coords = coords[b_indices, ...].cuda()
                b_indices = b_indices.cuda()
                pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

                with torch.no_grad():
                    im_estim[b_indices, :] = pixelvalues

                loss = criterion(pixelvalues, imten[b_indices, :])

                optim.zero_grad()
                loss.backward()
                optim.step()

                lossval = loss.item()
                train_loss += lossval
                nchunks += 1

            if occupancy:
                mse_array[idx] = volutils.get_IoU(im_estim, imten,
                                                  mcubes_thres)
            else:
                mse_array[idx] = train_loss / nchunks
            time_array[idx] = time.time()
            scheduler.step()

            im_estim_vol = im_estim.reshape(H, W, T)

            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim)

        total_time = time.time() - tic
        nparams = utils.count_parameters(model)

        best_img = best_img.reshape(H, W, T).detach().cpu().numpy()

        if posencode:
            nonlin = 'posenc'

        indices, = np.where(time_array > 0)
        time_array = time_array[indices]
        mse_array = mse_array[indices]

        mdict[nonlin] = {
            'mse_array': mse_array,
            'time_array': time_array - time_array[0],
            'nparams': utils.count_parameters(model),
            'Number of parameters': nparams,
            'Best PSNR': utils.psnr(im, best_img),
            'Best IoU': volutils.get_IoU(best_img, im, mcubes_thres),
            'Total time': total_time/60
        }

        io.savemat(f'/rds/general/user/atk23/home/wire/results/occupancy/{nonlin}.mat', mdict[nonlin])

        # Generate a mesh with marching cubes if it is an occupancy volume
        if occupancy:
            savename = f'/rds/general/user/atk23/home/wire/results/occupancy/{nonlin}.dae'
            volutils.march_and_save(best_img, mcubes_thres, savename, True)

        utils.log(f'Total time {total_time/60} minutes')
        if occupancy:
            utils.log(f'IoU: {volutils.get_IoU(best_img, im, mcubes_thres)}')
        else:
            utils.log(f'Best PSNR: {utils.psnr(im, best_img)} dB')
        utils.log(f'Total pararmeters: {(nparams / 1e6)} million')
        

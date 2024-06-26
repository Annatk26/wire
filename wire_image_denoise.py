from modules import setup

setup.seed_everything()

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from modules import models, utils
from scipy import io
from torch.optim.lr_scheduler import LambdaLR

if __name__ == "__main__":
    utils.log(
        "Starting image denoising experiment with multi-scale activation functions"
    )
    plt.gray()
    # nonlin_types = ["wire", "siren", "mfn", "relu", "posenc", "gauss"]
    nonlin_types = ["wire"]
    baseline = False  # baseline implementations
    trainable = False  # if model parameters are trainable

    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity

    niters = 2000  # Number of SGD iterations (2000)
    expected = [30.2, 26.6, 28.1, 0, 29.2, 29.7]  # Expected PSNR values

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    tau = 3e1  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    omega0 = 7.0  # Frequency of sinusoid
    sigma0 = 4.0  # Sigma of Gaussian (8.0)

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    # hidden_features = 256  # Number of hidden units per layer
    hidden_features = 300  # Number of hidden units per layer
    maxpoints = 256 * 256  # Batch size

    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(
        plt.imread("/rds/general/user/atk23/home/wire/data/parrot.png").astype(
            np.float32),
        True,
    )
    im = cv2.resize(im, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape

    # Create a noisy image
    im_noisy = utils.measure(im, noise_snr, tau)

    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    gt = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]
    gt_noisy = torch.tensor(im_noisy).cuda().reshape(H * W, 3)[None, ...]

    for i, nonlin in enumerate(nonlin_types):
        # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
        # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3
        # Set learning rate based on nonlinearity
        learning_rate = {
            "wire": 5e-3,
            "siren": 2e-3,
            "mfn": 5e-2,
            "relu": 1e-3,
            "posenc": 2e-3,
            "gauss": 3e-3,
        }[nonlin]
        utils.log(f"{nonlin} learning rate: {learning_rate}")
        if nonlin == "wire":
            sigma0 = 6.0  # 8.0
            size = np.round(hidden_features / np.sqrt(2))
            if size % 3 == 0:
                scale_tensor = [np.repeat([2.0, 10.0, 20.0], int(size / 3))]
            else:
                scale_tensor = np.repeat([20.0, 10.0, 2.0], [71, 71, 70])            
            # scale_tensor = np.linspace(2.0, 20.0,
                                    #    int(hidden_features / np.sqrt(2)))
        utils.log(f"Omega0: {omega0}, Sigma0: {sigma0}")

        if nonlin == "posenc":
            nonlin = "relu"
            posencode = True
            if tau < 100:
                sidelength = int(max(H, W) / 3)
            else:
                sidelength = int(max(H, W))
        else:
            posencode = False
            sidelength = H

        model = models.get_INR(
            nonlin=nonlin,
            in_features=2,
            out_features=3,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=omega0,
            hidden_omega_0=omega0,
            scale=sigma0,
            scale_tensor=scale_tensor,
            pos_encode=posencode,
            sidelength=sidelength
        )

        model.cuda()

        # Create an optimizer
        optim = torch.optim.Adam(lr=learning_rate *
                                 min(1, maxpoints / (H * W)),
                                 params=model.parameters())

        # Schedule to reduce lr to 0.1 times the initial rate in final epoch
        scheduler = LambdaLR(optim, lambda x: 0.1**min(x / niters, 1))

        mse_array = torch.zeros(niters, device="cuda")
        mse_loss_array = torch.zeros(niters, device="cuda")
        time_array = torch.zeros_like(mse_array)

        best_mse = torch.tensor(float("inf"))
        best_img = None

        rec = torch.zeros_like(gt)

        tbar = range(niters)
        init_time = time.time()
        for epoch in tbar:
            indices = torch.randperm(H * W)

            for b_idx in range(0, H * W, maxpoints):
                b_indices = indices[b_idx:min(H * W, b_idx + maxpoints)]
                b_coords = coords[:, b_indices, ...].cuda()
                b_indices = b_indices.cuda()
                pixelvalues = model(b_coords)

                with torch.no_grad():
                    rec[:, b_indices, :] = pixelvalues

                loss = ((pixelvalues - gt_noisy[:, b_indices, :])**2).mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

            time_array[epoch] = time.time() - init_time

            with torch.no_grad():
                mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
                mse_array[epoch] = ((gt - rec)**2).mean().item()
                im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
                im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

                psnrval = -10 * torch.log10(mse_array[epoch])

            scheduler.step()

            imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

            # cv2.imshow('Reconstruction', imrec[..., ::-1])
            # cv2.waitKey(1)

            if (mse_array[epoch] < best_mse) or (epoch == 0):
                best_mse = mse_array[epoch]
                best_img = imrec

        if posencode:
            nonlin = "posenc"

        utils.log(f"Best PSNR for {nonlin}: {utils.psnr(im, best_img)}")
        utils.log(
            f"Trained scale0: {model.net[0].scale_0.item()} & Trained omega0: {model.net[0].omega_0.item()}"
        )
        mdict[nonlin] = {
            "rec": best_img,
            "gt": im,
            "im_noisy": im_noisy,
            "mse_noisy_array": mse_loss_array.detach().cpu().numpy(),
            "mse_array": mse_array.detach().cpu().numpy(),
            "time_array": time_array.detach().cpu().numpy(),
        }
        if baseline:
            metrics[nonlin] = {
                "Omega0": omega0,
                "Sigma0": sigma0,
                "Learning rate": learning_rate,
                "Number of parameters": utils.count_parameters(model),
                "Best PSNR": utils.psnr(im, best_img),
                "Expected PSNR": expected[i],
                "PSNR Difference": abs(utils.psnr(im, best_img) - expected[i])
            }
        else:
            metrics[nonlin] = {
                "Omega0": omega0,
                "Sigma0": sigma0,
                "Learning rate": learning_rate,
                "Number of parameters": utils.count_parameters(model),
                "Best PSNR": utils.psnr(im, best_img),
            }

    folder_name = utils.make_unique(
        "wire_denoising",
        "/rds/general/user/atk23/home/wire/multiscale_results")
    os.makedirs(
        f"/rds/general/user/atk23/home/wire/multiscale_results/{folder_name}",
        exist_ok=True)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/multiscale_results/{folder_name}/info.mat",
        mdict)
    io.savemat(
        f"/rds/general/user/atk23/home/wire/multiscale_results/{folder_name}/metrics.mat",
        metrics)

    utils.tabulate_results(
        f"/rds/general/user/atk23/home/wire/multiscale_results/{folder_name}/metrics.mat",
        f"/rds/general/user/atk23/home/wire/multiscale_results/{folder_name}")

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
    utils.log("Starting image denoising experiment")
   # utils.log("Analysis of different c")
    plt.gray()
    #nonlin_types = ["bspline_sig"]
    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity
    niters = 2000  # Number of SGD iterations (2000)
    learning_rate = np.linspace(1e-3, 5e-2, 30)  # Learning rate

    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3

    tau = 3e1  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    #omega0 = 5.0  # Frequency of sinusoid
    omega0 = 0.07 
    sigma0 = 23.68  # Sigma of Gaussian (8.0)

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = 256  # Number of hidden units per layer
    maxpoints = 256 * 256  # Batch size

    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(
        plt.imread("/home/atk23/wire/data/parrot.png").astype(
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

    #for sigma in sigma0:
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3
    best_psnr = []
    # Set learning rate based on nonlinearity
    for i, rate in enumerate(learning_rate):
        #utils.log(f"Omega0: {omega0}, Sigma0: {sigma}")
        #utils.log(f"BSpline learning rate: {learning_rate}")
        nonlin = "bspline_sig"
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
            pos_encode=posencode,
            sidelength=sidelength,
        )

        model.cuda()

        # Create an optimizer
        optim = torch.optim.Adam(lr=rate *
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

        mdict[str(i)] = {
            "Learning rate": rate,
            "rec": best_img,
            "gt": im,
            "im_noisy": im_noisy,
            "mse_noisy_array": mse_loss_array.detach().cpu().numpy(),
            "mse_array": mse_array.detach().cpu().numpy(),
            "time_array": time_array.detach().cpu().numpy(),
        }
        metrics[str(i)] = {
            "Omega0": omega0,
            "Sigma0": sigma0,
            "Learning Rate": rate,
            "Number of parameters": utils.count_parameters(model),
            "Best PSNR": utils.psnr(im, best_img),
        }
        best_psnr.append(utils.psnr(im, best_img))
        #utils.log(f"Number of parameters: {utils.count_parameters(model)}, Best PSNR: {utils.psnr(im, best_img)}")
    
    folder_name = utils.make_unique("sigmoid_rate", "/home/atk23/wire/bspline_results/")
    os.makedirs(f"/home/atk23/wire/bspline_results/{folder_name}",
                exist_ok=True)
    io.savemat(f"/home/atk23/wire/bspline_results/{folder_name}/info.mat",
               mdict)
    io.savemat(f"/home/atk23/wire/bspline_results/{folder_name}/metrics.mat", metrics)

    #utils.tabulate_results(f"/home/atk23/wire/bspline_results/{folder_name}/metrics.mat")
    plt.plot(learning_rate, best_psnr, label="Best PSNR")
    plt.xlabel("Learning rate")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Learning rate")
    plt.legend()
    plt.savefig(f"/home/atk23/wire/bspline_results/{folder_name}/best_psnr.png")
    utils.log(f"Best PSNR: {np.max(best_psnr)} at k = {learning_rate[np.argmax(best_psnr)]}")

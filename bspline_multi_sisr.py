from modules import setup

setup.seed_everything()

import argparse
import os
import importlib
import copy

import numpy as np
from scipy import interpolate
from scipy import io

from skimage.metrics import structural_similarity as ssim_func

import matplotlib.pyplot as plt

plt.gray()
import cv2

import torch
import torch.nn
from torch.utils.data import DataLoader

from modules import models, motion, utils
from configs import CONFIGS

models = importlib.reload(models)

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, required=True)
args = parser.parse_args()

curr_config = CONFIGS[args.config_name]

if __name__ == "__main__":
    utils.log("Starting multi-SR experiment")
    nonlin = curr_config[
        "nonlin"
    ]  # type of nonlinearity, 'wire', 'siren', 'mfn', 'relu', 'posenc', 'gauss'
    niters = curr_config["niters"]  # Number of SGD iterations (2000)
    learning_rate = curr_config["learning_rate"]  # Learning rate.

    os.makedirs(
        "/rds/general/user/atk23/home/wire/baseline_results/multi_SR", exist_ok=True
    )
    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store PSNR and SSIM values

    # Data generation constants
    scale_sr = 4  # Super resolution factor
    nimg = int(0.25 * scale_sr * scale_sr)  # Number of images to combine
    scaling = 0.5  # Scaling for the image (1)

    # Motion constants
    shift_max = 5 * scale_sr
    theta_max = np.pi / 10
    reg_method = cv2.MOTION_EUCLIDEAN

    # Gabor filter constants. These settings work for SIREN, Gauss, and WIRE
    if nonlin == "siren":
        omega0 = 30.0  # Frequency of sinusoid
    else:
        omega0 = 10.0
    sigma0 = curr_config["scale"]  # Sigma of Gaussian
    scale_tensor = torch.tensor(curr_config["scale_tensor"]).cuda()

    # Noise constants
    tau = 1000  # Max. photon count
    noise_snr = 2  # Readout noise
    use_gt = True  # Use ground truth calibration info

    # Neural network constants
    batch_size = 4  # Image batch size
    img_every = 4  # Display result every these iterations

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    hidden_features = curr_config["hidden_features"]  # Number of hidden units per layer
    scaled_hidden_features = curr_config["scaled_hidden_features"]

    # Read image
    im = cv2.resize(
        plt.imread("/rds/general/user/atk23/home/wire/data/kodak.png"),
        None,
        fx=scaling,
        fy=scaling,
        interpolation=cv2.INTER_AREA,
    )
    H, W, _ = im.shape
    utils.log(f"Image shape: {im.shape}")

    # Create a stack of images. Do not resize yet.
    data = motion.get_imstack(im, 1, shift_max, theta_max, nshifts=nimg)
    imstack_hr, Xstack_gt, Ystack_gt, ecc_mats_gt = data

    # Now resize
    imstack = np.zeros((nimg, H // scale_sr, W // scale_sr, 3), dtype=np.float32)

    for idx in range(nimg):
        imstack[idx, ...] = cv2.resize(
            imstack_hr[idx, ...],
            None,
            fx=1 / scale_sr,
            fy=1 / scale_sr,
            interpolation=cv2.INTER_AREA,
        )
    imstack = np.transpose(imstack, [0, 3, 1, 2])

    nimg, _, Hl, Wl = imstack.shape

    # Register the stack
    if use_gt:
        Xstack = Xstack_gt
        Ystack = Ystack_gt
        mask = np.ones(nimg)
    else:
        utils.log("Registering stack")
        Xstack, Ystack, mask, mats, align_err = motion.register_stack(
            imstack, (H, W), method=reg_method
        )

    masks = 1 - np.float32(imstack == 0)

    # Create a dataset
    dataset = motion.ImageSRDataset(imstack, Xstack, Ystack, masks, jitter=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True
    )

    # Create loss functions
    criterion_mmse = torch.nn.MSELoss()
    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
    else:
        posencode = False

    model = models.get_INR(
        nonlin=nonlin,
        in_features=2,
        out_features=3,
        hidden_features=hidden_features,
        scaled_hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        first_omega_0=omega0,
        hidden_omega_0=omega0,
        scale=sigma0,
        scale_tensor=scale_tensor,
        pos_encode=posencode,
        sidelength=max(H, W),
    )

    nparams = sum(p.numel() for p in model.parameters())
    compression = nimg * Hl * Wl / nparams
    utils.log(f"Learning with {nparams} parameters ({compression} compression)")

    # Send model to CUDA
    model.cuda()

    # Create an optimizer
    optimizer = torch.optim.Adam(lr=learning_rate, params=model.parameters())

    mse_array = np.zeros(niters)
    psnr_array = np.zeros(niters)
    best_loss = float("inf")
    best_output = None
    best_state_dict = model.state_dict()

    # Create full coordinates
    Y, X = np.mgrid[:H, :W]
    X = 2 * X / W - 1
    Y = 2 * Y / H - 1
    coords_full = (
        torch.stack(
            (torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32))),
            dim=-1,
        )
        .reshape(-1, 2)
        .cuda()
    )

    # Compute a simple grid data interpolation
    points = np.hstack(
        (
            Xstack[..., ::scale_sr, ::scale_sr].reshape(-1, 1),
            Ystack[..., ::scale_sr, ::scale_sr].reshape(-1, 1),
        )
    )

    im_interp = np.zeros_like(im)

    for idx in range(3):
        utils.log(
            f"Points: {points.shape} | Image: {imstack[:, idx, ...].reshape(-1, 1).shape}"
        )
        im_interp[..., idx] = interpolate.griddata(
            points, imstack[:, idx, ...].reshape(-1, 1), (X, Y), method="linear"
        )[..., 0]
    im_interp[np.isnan(im_interp)] = 0
    snr_interp = utils.psnr(im, im_interp)
    ssim_interp = ssim_func(im, im_interp.astype(np.float32), multichannel=True)

    # Create area downsampler
    downsampler = torch.nn.AvgPool2d(scale_sr)

    tbar = range(niters)
    for epoch in tbar:
        train_mse = 0
        for idx, data in enumerate(dataloader):
            coords, gt, mask = data
            coords, gt, mask = coords.cuda(), gt.cuda(), mask.cuda()

            output_hr = model(coords).reshape(-1, H, W, 3).permute(0, 3, 1, 2)
            output = downsampler(output_hr).permute(0, 2, 3, 1).reshape(-1, Hl * Wl, 3)

            mmse_loss = criterion_mmse(output * mask, gt * mask)

            loss = mmse_loss

            if mmse_loss < best_loss:
                best_loss = mmse_loss
                best_output = output
                best_state_dict = copy.deepcopy(model.state_dict())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse += mmse_loss.item()

        mse_array[epoch] = train_mse

        if epoch % img_every == 0:
            with torch.no_grad():
                output_full = model(coords_full[None, :, :]).reshape(H, W, 3)
                img_full = output_full.cpu().detach().numpy()
            img = output[0, ...].cpu().detach().numpy().reshape(Hl, Wl, 3)

            snrval = utils.psnr(im, img_full)
            ssimval = ssim_func(im, img_full, multichannel=True)
            txt = f"PSNR: {snrval} | SSIM: {ssimval}"

            psnr_array[epoch] = snrval

    with torch.no_grad():
        model.load_state_dict(best_state_dict)
        output_full = model(coords_full[None, :, :]).reshape(H, W, 3)
        img_full = output_full.cpu().detach().numpy()
    img_up = cv2.resize(imstack[0, ...], (W, H), interpolation=cv2.INTER_NEAREST)

    snrval = utils.psnr(im, img_full)
    ssimval = ssim_func(im, img_full, multichannel=True)

    # filename = f'/rds/general/user/atk23/home/wire/baseline_results/multi_SISR/{scale_sr}x_{nimg}images_{nonlin}'

    # if use_gt:
    #     filename += '_oracle_reg'
    # else:
    #     filename += '_estim_reg'

    mdict[nonlin] = {
        "rec": img_full,
        "psnr_rec": snrval,
        "ssim_rec": ssimval,
        "psnr_interp": snr_interp,
        "ssim_interp": ssim_interp,
        "rec_interp": im_interp,
    }

    metrics[nonlin] = {
        "psnr_rec": snrval,
        "ssim_rec": ssimval,
        "psnr_interp": snr_interp,
        "ssim_interp": ssim_interp,
    }

    utils.log(f"Nonlin: {nonlin} | PSNR: {snrval} | SSIM: {ssimval}")

    folder_name = utils.make_unique(
        f"{curr_config['name']}",
        "/rds/general/user/atk23/home/wire/multiscale_results/multi_sisr",
    )
    filepath = (
        f"/rds/general/user/atk23/home/wire/multiscale_results/multi_sisr/{folder_name}"
    )
    os.makedirs(filepath, exist_ok=True)
    io.savemat(os.path.join(filepath, "info.mat"), mdict)
    io.savemat(os.path.join(filepath, "metrics.mat"), metrics)
    utils.tabulate_results(os.path.join(filepath, "metrics.mat"), filepath)
    utils.display_image(os.path.join(filepath, "info.mat"))

    utils.log("Finished Multi-SISR experiment")

    # io.savemat(f'/rds/general/user/atk23/home/wire/baseline_results/multi_SR/{filename}.mat', mdict[nonlin])

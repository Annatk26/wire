from modules import setup

setup.seed_everything()

import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from scipy import io
from torch.optim.lr_scheduler import LambdaLR

from configs import CONFIGS
from modules import models, utils

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, required=True)
args = parser.parse_args()

curr_config = CONFIGS[args.config_name]

utils.log("Starting image denoising experiment")
plt.gray()

tvl = curr_config['tvl']  # Total variation loss
weight_init = False

mdict = {}  # Dictionary to store info of each non-linearity
metrics = {}  # Dictionary to store metrics of each non-linearity
best_psnr = 0

tau = curr_config["tau"]  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
noise_snr = curr_config["noise_snr"]  # Readout noise (dB)

# Activation function constants
omega0 = 7.0
nonlin = curr_config["nonlin"]
sigma0 = curr_config["scale"]
scale_tensor = torch.tensor(curr_config["scale_tensor"]).cuda()

# Network parameters
hidden_layers = 2  # Number of hidden layers in the MLP
hidden_features = curr_config["hidden_features"]  # Number of hidden units per layer
maxpoints = curr_config["maxpoints"]  # Batch size
niters = curr_config["niters"]  # Number of SGD iterations (2000)
scaled_hidden_features = curr_config[
    "scaled_hidden_features"
]  # Number of hidden units in the first layer
learning_rate = curr_config["learning_rate"]
if nonlin == "bspline_mscale_1_new":
    in_features = 2 * len(scale_tensor) * scaled_hidden_features
else:
    in_features = 2

# Read image and scale. A scale of 0.5 for parrot image ensures that it
# fits in a 12GB GPU
im = utils.normalize(
    plt.imread("/rds/general/user/atk23/home/wire/data/parrot.png").astype(np.float32),
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

utils.log("System Information")
utils.log(f"Non-linearity: {nonlin}, Learning Rate: {learning_rate}, Scale: {sigma0}")
utils.log(
    f"Scale tensor: {scale_tensor}, Hidden features (scaled layer): {scaled_hidden_features}"
)

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
    sidelength=sidelength,
)

model.cuda()

# Create an optimizer
optim = torch.optim.Adam(
    lr=learning_rate * min(1, maxpoints / (H * W)), params=model.parameters()
)

# Schedule to reduce lr to 0.1 times the initial rate in final epoch
scheduler = LambdaLR(optim, lambda x: 0.1 ** min(x / niters, 1))

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
        b_indices = indices[b_idx : min(H * W, b_idx + maxpoints)]
        b_coords = coords[:, b_indices, ...].cuda()
        b_indices = b_indices.cuda()
        pixelvalues = model(b_coords)

        with torch.no_grad():
            rec[:, b_indices, :] = pixelvalues

        mse_loss = ((pixelvalues - gt_noisy[:, b_indices, :]) ** 2).mean()

        lambda_tv = curr_config["lambda_tv"]
        tv_loss = 0.0
        if tvl:
            if b_idx % (maxpoints * 10) == 0:  # every 10 batches
                with torch.no_grad():
                    full_prediction = (
                        model(coords.cuda()).reshape(1, H, W, 3).permute(0, 3, 1, 2)
                    )
                    tv_loss = utils.total_variation_loss(full_prediction)
            else:
                tv_loss = 0.0

        loss = mse_loss + lambda_tv * tv_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

    time_array[epoch] = time.time() - init_time

    with torch.no_grad():
        mse_loss_array[epoch] = ((gt_noisy - rec) ** 2).mean().item()
        mse_array[epoch] = ((gt - rec) ** 2).mean().item()
        im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

        psnrval = -10 * torch.log10(mse_array[epoch])

    scheduler.step()
    imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

    if (mse_array[epoch] < best_mse) or (epoch == 0):
        best_mse = mse_array[epoch]
        best_img = imrec

if posencode:
    nonlin = "posenc"

utils.log(f"Best PSNR for {nonlin}: {utils.psnr(im, best_img)}")

folder_name = utils.make_unique(
    f"{curr_config['name']}", f"/rds/general/user/atk23/home/wire/multiscale_results/denoise/T{tau}_SNR{noise_snr}"
)
mdict[folder_name] = {
    "Scale": sigma0,
    "Learning rate": learning_rate,
    "rec": best_img,
    "gt": im,
    "im_noisy": im_noisy,
    "mse_noisy_array": mse_loss_array.detach().cpu().numpy(),
    "mse_array": mse_array.detach().cpu().numpy(),
    "time_array": time_array.detach().cpu().numpy(),
}
metrics[folder_name] = {
    "Scale": sigma0,
    "Scale tensor": curr_config["scale_tensor"],
    "Tau": tau,
    "Noise SNR": noise_snr,
    "Learning Rate": learning_rate,
    "Number of parameters": utils.count_parameters(model),
    "Best PSNR": utils.psnr(im, best_img),
}

filepath = f"/rds/general/user/atk23/home/wire/multiscale_results/denoise/T{tau}_SNR{noise_snr}/{folder_name}"
os.makedirs(filepath, exist_ok=True)
io.savemat(os.path.join(filepath, "info.mat"), mdict)
io.savemat(os.path.join(filepath, "metrics.mat"), metrics)
utils.tabulate_results(os.path.join(filepath, "metrics.mat"), filepath)
utils.display_image(os.path.join(filepath, "info.mat"))

utils.log("Image denoise experiment completed")

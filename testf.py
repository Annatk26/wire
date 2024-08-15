# from modules import setup

# setup.seed_everything()

import time

import cv2
# from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
from modules import models, utils
from torch.optim.lr_scheduler import LambdaLR


if __name__ == "__main__":
    utils.log(
        "Starting image denoising experiment for Quadratic B-spline non-linearity"
    )
    multiscale_opt = [True]
    # utils.log("Weight initialization: He Initialization")
    # utils.log("Analysis of different c")
    plt.gray()
    nonlin = "bspline_hier"  # Various implementations of B-spline
    utils.log(f"Non-linearity: {nonlin}")

    mdict = {}  # Dictionary to store info of each non-linearity
    metrics = {}  # Dictionary to store metrics of each non-linearity
    niters = 11  # Number of SGD iterations (2000)
    
    # WIRE works best at 5e-3 to 2e-2, Gauss and SIREN at 1e-3 - 2e-3,
    # MFN at 1e-2 - 5e-2, and positional encoding at 5e-4 to 1e-3
    
    # Noise parameters
    tau = 3e7  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
    noise_snr = 2  # Readout noise (dB)

    # Gabor filter constants.
    # We suggest omega0 = 4 and sigma0 = 4 for denoising, and omega0=20, sigma0=30 for image representation
    #omega0 = 5.0  # Frequency of sinusoid
    omega0 = 0.07
    # sigma0_all = [[1.0, 2.0, 6.0], 
    #             [0.3, 0.5, 0.7], [0.5, 1.0, 2.0]]  # Sigma of Gaussian (8.0)
    # sigma0 = 9.5522
    sigma0 = 1/9    # sigma0 = 0.5
    # sigma_tensor = [9.0, 8.0, 12.0, 16.0, 20.0, 24.0]
    # sigma_tensor = [0.05, 0.5, 0.75, 1.0]
    sigma_tensor = torch.tensor([(1/9),  4.0]) 
    learning_rate = 2e-2  # Learning rate
    # learning_rate = 2e-2
    utils.log(f"Sigma: {sigma0} & Learning rate: {learning_rate} & scale_tensor: {sigma_tensor}")

    # Network parameters
    hidden_layers = 2  # Number of hidden layers in the MLP
    scaled_hidden_feature = 16 # Number of hidden units in scaled layer layer
    hidden_features = 256  # Number of hidden units per layer
    maxpoints = 256 * 256  # Batch size
    # maxpoints = 50*50
    # maxpoints = (hidden_features*len(sigma_tensor))**2

    # Read image and scale. A scale of 0.5 for parrot image ensures that it
    # fits in a 12GB GPU
    im = utils.normalize(
        plt.imread("data\parrot.png").astype(
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

    indices = torch.randperm(H * W)
    split = int(H*W * 0.8)
    train_indices, test_indices = indices[:split], indices[split:]

    gt = torch.tensor(im).cpu().reshape(H * W, 3)[None, ...]
    gt_noisy = torch.tensor(im_noisy).cpu().reshape(H * W, 3)[None, ...]

    for multiscale in multiscale_opt:
        utils.log(f"Multiscale: {multiscale}")
        # Set learning rate based on nonlinearity
        #utils.log(f"Omega0: {omega0}, Sigma0: {sigma}")
        #utils.log(f"BSpline learning rate: {learning_rate}")
        if nonlin == "posenc":
            nonlin = "relu"
            posencode = True
            if tau < 100:
                sidelength = int(max(H, W) / 3)
            else:
                sidelength = int(max(H, W))
        # elif nonlin == "bspline_form":
        #     posencode = True
        #     if tau < 100:
        #         sidelength = int(max(H, W) / 3)
        #     else:
        #         sidelength = int(max(H, W))

        else:
            posencode = False
            sidelength = H

        model = models.get_INR(
            nonlin=nonlin,
            in_features=2,
            out_features=3,
            scaled_hidden_features=scaled_hidden_feature,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=omega0,
            hidden_omega_0=omega0,
            scale=sigma0,
            scale_tensor=sigma_tensor,
            pos_encode=posencode,
            sidelength=sidelength
        )

        model.cpu()

        utils.log(f"Number of parameters: {utils.count_parameters(model)}")

        # Create an optimizer
        # param_groups = []
        # for i, stage in enumerate(model.stages):
        #     param_groups.append({"params": stage.parameters(), "lr": learning_rate[i]*min(1, maxpoints / (H * W))})
        #     param_groups.append({"params": model.linears[i].parameters(), "lr": learning_rate[i]*min(1, maxpoints / (H * W))})

        optim = torch.optim.Adam(lr=learning_rate *
                                    min(1, maxpoints / (H * W)),
                                    params=model.parameters())

        # optim = torch.optim.Adam(param_groups)
        scheduler = LambdaLR(optim, lambda x: 0.1 ** min(x/niters, 1))


        # Schedule to reduce lr to 0.1 times the initial rate in final epoch
        # scheduler = LambdaLR(optim, lambda x: 0.1**min(x / niters, 1))

        mse_array = torch.zeros(niters, device="cpu")
        mse_loss_array = torch.zeros(niters, device="cpu")
        time_array = torch.zeros_like(mse_array)

        best_mse = torch.tensor(float("inf"))
        best_img = None
        train_mse = []
        test_mse = []

        rec = torch.zeros_like(gt)

        tbar = range(niters)
        init_time = time.time()
        for epoch in tbar:
            indices = torch.randperm(H * W)

            for b_idx in range(0, H * W, maxpoints):
                b_indices = train_indices[b_idx:min(H * W, b_idx + maxpoints)]
                b_coords = coords[:, b_indices, ...].cpu()
                b_indices = b_indices.cpu()
                pixelvalues = model(b_coords)
                # if epoch % 2 == 0 and b_idx == 0:
                    # scaled_layer = model.net[0]
                   # print(scaled_layer.linear.weight[:4,:])

                # with torch.no_grad():
                    # rec[:, b_indices, :] = pixelvalues

                loss = ((pixelvalues -
                            gt_noisy[:, b_indices, :])**2).mean()

                optim.zero_grad()
                loss.backward()
                optim.step()

            time_array[epoch] = time.time() - init_time
            

            # with torch.no_grad():
            #     mse_loss_array[epoch] = ((gt_noisy - rec)**2).mean().item()
            #     mse_array[epoch] = ((gt - rec)**2).mean().item()
            #     im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            #     im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

            #     psnrval = -10 * torch.log10(mse_array[epoch])

            scheduler.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    train_pred = model(coords[:, train_indices, ...].cpu())
                    test_pred = model(coords[:, test_indices, ...].cpu())
                    
                    train_mse.append(((train_pred - gt_noisy[:, train_indices, ...])**2).mean().item())
                    test_mse.append(((test_pred - gt_noisy[:, test_indices, ...])**2).mean().item())
                    
                    full_pred = model(coords.cpu())
                    full_mse = ((full_pred - gt_noisy)**2).mean().item()
                    
                    if full_mse < best_mse:
                        best_mse = full_mse
                        best_img = full_pred.reshape(gt_noisy.shape[1:]).cpu().numpy()
                    
                    print(f"Epoch {epoch}, Train MSE: {train_mse[-1]:.4f}, Test MSE: {test_mse[-1]:.4f}")

            # imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

            # if (mse_array[epoch] < best_mse) or (epoch == 0):
            #     best_mse = mse_array[epoch]
            #     best_img = imrec
            
            # if epoch % 100 == 0:
            #     utils.log(
            #         f"Epoch: {epoch}, PSNR: {psnrval}"
            #     )

        if posencode:
            nonlin = "posenc"
        # utils.log(
        #     f"Best PSNR for {nonlin}: {utils.psnr(im, best_img)} & Learning rate: {learning_rate}"
        # )

        mdict[str(sigma0)] = {
            "Weight Initialization": "No Initialization",
            # "scale": model.net[1].scale_0.item(),
            "Learning rate": learning_rate,
            "rec": best_img,
            "gt": im,
            "im_noisy": im_noisy,
            "mse_noisy_array": mse_loss_array.detach().cpu().numpy(),
            "mse_array": mse_array.detach().cpu().numpy(),
            "time_array": time_array.detach().cpu().numpy(),
        }
        metrics[str(sigma0)] = {
            "Weight Initialization": "No Initialization",
            # "Sigma0": model.net[1].scale_0.item(),
            "Learning Rate": learning_rate,
            "Number of parameters": utils.count_parameters(model),
            "Best PSNR": utils.psnr(im, best_img),
        }
        utils.log(
            f"Best PSNR for {nonlin}: {utils.psnr(im, best_img)}"
        )
        # best_psnr.append(utils.psnr(im, best_img))
        #utils.log(f"Number of parameters: {utils.count_parameters(model)}, Best PSNR: {utils.psnr(im, best_img)}")

    # folder_name = utils.make_unique(
    #     "form", "/rds/general/user/atk23/home/wire/bspline_cubic_results/denoise")
    # os.makedirs(
    #     f"/rds/general/user/atk23/home/wire/bspline_results/denoise/{folder_name}",
    #     exist_ok=True)
    # io.savemat(
    #     f"/rds/general/user/atk23/home/wire/bspline_results/denoise/{folder_name}/info.mat",
    #     mdict)
    # io.savemat(
    #     f"/rds/general/user/atk23/home/wire/bspline_results/denoise/{folder_name}/metrics.mat",
    #     metrics)
    # utils.log("Image denoise experiment completed")

#utils.tabulate_results(f"/home/atk23/wire/bspline_results/{folder_name}/metrics.mat")
# plt.plot(param['learning_rate'], best_psnr, label="Best PSNR")
# plt.xlabel("Learning rate")
# plt.ylabel("PSNR")
# plt.title("PSNR vs Learning rate")
# plt.legend()
# plt.savefig(
#     f"/rds/general/user/atk23/home/wire/bspline_results/{folder_name}/best_psnr.png"
# )

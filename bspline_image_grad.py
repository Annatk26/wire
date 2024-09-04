from modules import setup

setup.seed_everything()

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from modules import models
from scipy import io

import time
import seaborn as sns
from matplotlib.colors import ListedColormap

from configs import CONFIGS
from modules import models, utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, required=True)
args = parser.parse_args()

curr_config = CONFIGS[args.config_name]

utils.log("Starting image representation experiment -- gradient and laplacian")
plt.gray()

tvl = curr_config["tvl"]  # Total variation loss
weight_init = False

mdict = {}  # Dictionary to store info of each non-linearity
metrics = {}  # Dictionary to store metrics of each non-linearity

tau = curr_config[
    "tau"
]  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
noise_snr = curr_config["noise_snr"]  # Readout noise (dB)

# Activation function constants
omega0 = 15.0
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

folder_name = utils.make_unique(
    f"{curr_config['name']}",
    "/rds/general/user/atk23/home/wire/multiscale_results/gradients",
)
filepath = f"/rds/general/user/atk23/home/wire/multiscale_results/gradients/{folder_name}"
os.makedirs(filepath, exist_ok=True)

def get_image_tensor(sidelength):
    img = Image.fromarray(skimage.data.astronaut())
    img_array = np.array(img)

    # Convert the image to grayscale
    img_gray = skimage.color.rgb2gray(img_array)
    img = Image.fromarray((img_gray * 255).astype(np.uint8))
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_image_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        return self.coords, self.pixels

cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman,
                        batch_size=1,
                        pin_memory=True,
                        num_workers=0)
if nonlin == "posenc":
    nonlin = "relu"
    posencode = True
else:
    posencode = False

model = models.get_INR(
            nonlin=nonlin,
            in_features=2,
            out_features=1,
            scaled_hidden_features=scaled_hidden_features,
            hidden_features=hidden_features,
            hidden_layers=2,
            first_omega_0=omega0,
            hidden_omega_0=omega0,
            scale=sigma0,
            scale_tensor=scale_tensor,
            pos_encode=posencode,
            sidelength=256
        )

model.cpu()


total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 10
optim = torch.optim.Adam(lr=learning_rate,
                                    params=model.parameters())
model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cpu(), ground_truth.cpu()

for step in range(total_steps):
    model_output, coords = model.forward_with_grad(model_input)
    loss = ((model_output - ground_truth)**2).mean()

    if step == 499:
        print("Step %d, Total loss %0.6f" % (step, loss))
        img_grad = gradient(model_output, coords)
        img_laplacian = laplace(model_output, coords)

        mdict = {
            "model_output": model_output.cpu().view(256,256).detach().numpy(),
            "img_grad": img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy(),
            "img_laplacian": img_laplacian.cpu().view(256, 256).detach().numpy()
        }
        io.savemat(os.path.join(filepath, "info.mat"), mdict)


        fig, axes = plt.subplots(1,3, figsize=(18,6))
        axes[0].imshow(model_output.cpu().view(256,256).detach().numpy(), cmap='gray')
        axes[1].imshow(img_grad.norm(dim=-1).cpu().view(256, 256).detach().numpy(), cmap=sns.color_palette("icefire", as_cmap=True))
        axes[2].imshow(img_laplacian.cpu().view(256, 256).detach().numpy(), cmap=sns.color_palette("icefire", as_cmap=True))
        plt.savefig(os.path.join(filepath, "img_grad.png"))

    optim.zero_grad()
    loss.backward()
    optim.step()

utils.log("Image gradient experiment completed")

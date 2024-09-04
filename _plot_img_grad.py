import os
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def scale_func(x, pos):
    return f'{x/1e4:.0f}'  # Adjust scale to 10^4

filepath = "/rds/general/user/atk23/home/wire/multiscale_results/gradients/Bspline_s9_5_LR8e3_E2000_T3e7_2"
file = sio.loadmat(os.path.join(filepath, "info.mat"))
model_output = file["model_output"]
img_grad = file["img_grad"]
img_laplacian = file["img_laplacian"]
color_pallete = 'magma'
# print(img_grad)
fig, axes = plt.subplots(1,3, figsize=(18,6))
im0 = axes[0].imshow(model_output, cmap='gray')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
im1 = axes[1].imshow(img_grad, cmap=sns.color_palette(color_pallete, as_cmap=True))
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
im2 = axes[2].imshow(img_laplacian, cmap=sns.color_palette('rocket_r', as_cmap=True))
cbar = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

cbar.formatter = FuncFormatter(scale_func)
cbar.update_ticks()

# Add the *10^4 label on top of the colorbar
cbar.set_label(r'$\times 10^4$', rotation=0, labelpad=15, fontsize=12, loc='top')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4)
plt.savefig(os.path.join(filepath, "img_grad.png"))

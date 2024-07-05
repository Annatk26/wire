import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

mat = sio.loadmat(
    "/rds/general/user/atk23/home/wire/multiscale_results/denoise/MScale1_TVL_Scale9_Lr4e-3_1/info.mat"
)
for key in mat.keys():
    if not key.startswith("__"):
        print(key)
        img = mat[key][0, 0]
        # print(mat['MScale1']['rec'].shape)
        # print(img['rec'][0,0].shape)
        image = img['rec']
        plt.imsave(f"image_{key}.png", np.clip(abs(image), 0, 1), 
            vmin=0.0,
            vmax=1.0)

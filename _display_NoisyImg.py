from modules import utils
import cv2
import matplotlib.pyplot as plt
import numpy as np

tau = 3e7  # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
noise_snr = 2  # Readout noise (dB)

im = utils.normalize(
    plt.imread("/rds/general/user/atk23/home/wire/data/Sky.png").astype(np.float32),
    True,
)
im = cv2.resize(im, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
H, W, _ = im.shape

# Create a noisy image
im_noisy = utils.measure(im, noise_snr, tau)
plt.imsave(
    f"/rds/general/user/atk23/home/wire/data_noisy/Sky_noisy_T{tau}_snr{noise_snr}.png",
    np.clip(abs(im_noisy), 0, 1),
    vmin=0.0,
    vmax=1.0,
)

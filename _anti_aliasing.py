import numpy as np
import matplotlib.pyplot as plt
import cv2
from modules import utils

if __name__ == "__main__":
    # Image pre-processing
    scale = 16  # Downsampling factor
    scale_im = 1 / 6  # Initial image downsample (1/3)

    # Read image
    im = utils.normalize(
        plt.imread("/rds/general/user/atk23/home/wire/data/Face.png").astype(
            np.float32),
        True,
    )

    print(im.shape)
    if im.shape[-1] == 4:
        im = im[:, :, :3]

    im = cv2.resize(im,
                    None,
                    fx=scale_im,
                    fy=scale_im,
                    interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    print(H, W)

    # Ensures image dimensions are multiples of scale
    im = im[:scale * (H // scale), :scale * (W // scale), :]
    H, W, _ = im.shape

    im_lr = cv2.resize(im,
                       None,
                       fx=1 / scale,
                       fy=1 / scale,
                       interpolation=cv2.INTER_AREA)


    H2, W2, _ = im_lr.shape
    im_bi = cv2.resize(im_lr,
                       None,
                       fx=scale,
                       fy=scale,
                       interpolation=cv2.INTER_LINEAR)

    plt.imsave("data/LR_image.png", im_bi)
    # # Low-resolution image
    # x = torch.linspace(-1, 1, W2).cuda()
    # y = torch.linspace(-1, 1, H2).cuda()
    # # High-resolution image
    # x_hr = torch.linspace(-1, 1, W).cuda()
    # y_hr = torch.linspace(-1, 1, H).cuda()

    # im_bi = cv2.resize(im_lr,
    #                    None,
    #                    fx=scale,
    #                    fy=scale,
    #                    interpolation=cv2.INTER_LINEAR)

    # utils.log("System Information")
 

    # X, Y = torch.meshgrid(x, y, indexing="xy")
    # coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

    # X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing="xy")
    # coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None,
    #                                                                      ...]

    # gt = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]
    # gt_lr = torch.tensor(im_lr).cuda().reshape(H2 * W2, 3)[None, ...]

    # im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
    # im_bi_ten = torch.tensor(im_bi).cuda().permute(2, 0, 1)[None, ...]

    # best_mse = float("inf")
    # best_img = None

    # downsampler = torch.nn.AvgPool2d(scale)


    # # rec = downsampler(rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...])
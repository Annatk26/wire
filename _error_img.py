from modules import utils
import cv2
import matplotlib.pyplot as plt
import numpy as np  

# Load the images
# image1 = cv2.imread('/rds/general/user/atk23/home/wire/data/parrot.png')
image2 = utils.normalize(plt.imread('/rds/general/user/atk23/home/wire/multiscale_results/denoise/Mscale2_ST4_3_LR8e3_E4000_1/Output_img.png').astype(np.float32), True)
image1 = utils.normalize(
    plt.imread("/rds/general/user/atk23/home/wire/data/parrot.png").astype(np.float32),
    True,
)
image1 = cv2.resize(image1, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
# Convert images to grayscale if they are not already
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

gray_image1 = gray_image1.astype(np.float32)
gray_image2 = gray_image2.astype(np.float32)

# Compute the absolute difference
error_image = cv2.absdiff(gray_image1, gray_image2)

# Normalize the error image to the range [0, 255] for visualization
normalized_error = cv2.normalize(error_image, None, 0, 255, cv2.NORM_MINMAX)
normalized_error = np.uint8(normalized_error)
plt.imsave('/rds/general/user/atk23/home/wire/multiscale_results/denoise/Mscale2_ST4_3_LR8e3_E4000_1/Error_Img.png', normalized_error, cmap='hot')


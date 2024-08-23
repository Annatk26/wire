from PIL import Image, ImageDraw, ImageFont
import os
import scipy.io as sio
from modules import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_PSNR(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    for key in file.keys():
        if "__" not in key:
            return file[key]['Best PSNR'][0][0]
        
def get_SSIM(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    for key in file.keys():
        if "__" not in key:
            return file[key]['Best SSIM'][0][0]
    # print(file.keys())

def create_composite(filepath):
    # Load the original image
    image_path = os.path.join(filepath, "Output_img.png")
    # image_path = os.path.join(filepath)
    original = Image.open(image_path)

    box_height = 130 # denoise
    # box_height = 80 # ct
    # Create a new white canvas
    canvas = Image.new('RGB', (original.width , original.height + box_height), 'black')

    # Paste the original image
    canvas.paste(original, (0, 0))

    # Create the zoomed inset (you'll need to adjust these coordinates)
    eye_region = original.crop((300, 150, 400, 250))
    eye_region = eye_region.resize((200, 200))
    canvas.paste(eye_region, (canvas.width - 250, 50))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("Times New Roman Bold.ttf", size=24)
    # Add text and metrics
    # Draw white box for text
    # box_padding = 28
    # box_width = 180
    # box_position = (20, canvas.height - 20 - box_height, 20 + box_width, canvas.height - 20)
    # box_position = (canvas.width, canvas.height, canvas.width + box_width, canvas.height+ box_height)
    # draw.rectangle(box_position, fill="white", outline="black", width=2)

    # Add text to the white box
    psnr = get_PSNR(filepath)[0][0]
    # ssim = get_SSIM(filepath)[0][0]
    # parrot = Image.open("data/parrot.png")


#     im = utils.normalize(
#     plt.imread("/rds/general/user/atk23/home/wire/data/parrot.png").astype(np.float32),
#     True,
# )
#     im = cv2.resize(im, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
#     im2 = utils.normalize(plt.imread(image_path).astype(np.float32), True)
#     im2 = im2[...,:3]
#     psnr = utils.psnr(im, im2)
    text1 = f"PSNR: {psnr:.2f} dB"
    # text2 = f"SSIM: {ssim:.2f}"
    text_color = "white"
    text_position = ((canvas.height // 2)-150, (canvas.height - box_height) + (box_height // 8))
    draw.text(text_position, text1, fill=text_color, font=font)
    # draw.text((text_position[0], text_position[1] + 40), text2, fill=text_color, font=font)


    # draw.text((10, 10), "Noisy image", fill="black", font=font)
    # draw.text((10, canvas.height - 30), "17.6dB\n0.34", fill="black", font=font)

    # Add borders
    draw.rectangle([0, 0, canvas.width-1, canvas.height-1], outline="black", width=2)
    # draw.rectangle([canvas.width-321, 69, canvas.width-100, 271], outline="black")
    inset_size = (200, 200)
    inset_left = canvas.width - inset_size[0] - 50
    inset_top = 50
    inset_right = inset_left + inset_size[0]
    inset_bottom = inset_top + inset_size[1]
    draw.rectangle([inset_left-1, inset_top-1, inset_right+1, inset_bottom+1], outline="black", width=2)
    return canvas

# Use the function
filepath = "multiscale_results/denoise/T2.0_SNR1/MscaleHier_ST4_LR8e3_E4000_T2_SNR1_1"
# filepath = "data_noisy/parrot_noisy_T2.0_snr1.png"
result = create_composite(filepath)
# result.save(os.path.join('data_noisy', "PSNR_T3.0_snr1.png"))
result.save(os.path.join(filepath, "Result_img.png"))
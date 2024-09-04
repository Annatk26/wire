from PIL import Image, ImageDraw, ImageFont
import os
import scipy.io as sio
from modules import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_PSNR(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    for key in file.keys():
        if "__" not in key:
            return file[key]["Best PSNR"][0][0]

def get_MSE(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    # print(file['Bspline_s9__DS16_LR1e3_E4000_1']['Best MSE'][0][0])
    for key in file.keys():
        if "__" not in key:
            return file[key]["Best MSE"][0][0]

def get_SSIM(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    for key in file.keys():
        if "__" not in key:
            return file[key]["Best SSIM"][0][0]
    # print(file.keys())

def get_SISR_metrics(filepath):
    file = sio.loadmat(os.path.join(filepath, "metrics.mat"))
    for key in file.keys():
        if "__" not in key:
            return file[key]["psnr_rec"][0][0][0][0], file[key]["ssim_rec"][0][0][0][0]


def create_composite(filepath, exp, result_type):
    # Load the original image
    if result_type == 'NotOriginal':
        image_path = os.path.join(filepath, "Output_img.png")
    else:
        image_path = filepath
    # image_path = os.path.join(filepath)
    original = Image.open(image_path)
    if exp == 'denoise':
        box_height = 150
        if result_type == 'Original1':
            original = original.resize(
                (original.width // 2, original.height // 2))
        # Create a new white canvas
        canvas = Image.new("RGB",
                           (original.width, original.height + box_height),
                           "black")
        # Paste the original image
        canvas.paste(original, (0, 0))
        # Create the zoomed inset (you'll need to adjust these coordinates)
        eye_region = original.crop((300, 150, 400, 250))
        eye_region = eye_region.resize((200, 200))
        canvas.paste(eye_region, (canvas.width - 250, 50))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("Times New Roman Bold.ttf", size=100)
        if result_type == 'Original':
            im = utils.normalize(
                plt.imread("data/parrot.png").astype(np.float32),
                True,
            )
            im = cv2.resize(im,
                            None,
                            fx=1 / 2,
                            fy=1 / 2,
                            interpolation=cv2.INTER_AREA)
            im_noisy = utils.normalize(
                plt.imread(image_path).astype(np.float32), True)
            im_noisy = im_noisy[:, :, :3]
            psnr = utils.psnr(im, im_noisy)
            text1 = f"PSNR: {psnr:.2f} dB"
            text_color = "white"
            text_position = (
                (canvas.height // 2) - 230,
                (canvas.height - box_height) + (box_height // 8),
            )
            draw.text(text_position, text1, fill=text_color, font=font)
        elif result_type == 'NotOriginal':
            psnr = get_PSNR(filepath)[0][0]
            text1 = f"PSNR: {psnr:.2f} dB"
            text_color = "white"
            text_position = (
                (canvas.height // 2) - 230,
                (canvas.height - box_height) + (box_height // 8),
            )
            draw.text(text_position, text1, fill=text_color, font=font)
        # Add borders
        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                       outline="black",
                       width=2)
        inset_size = (200, 200)
        inset_left = canvas.width - inset_size[0] - 50
        inset_top = 50
        inset_right = inset_left + inset_size[0]
        inset_bottom = inset_top + inset_size[1]
        draw.rectangle(
            [inset_left - 1, inset_top - 1, inset_right + 1, inset_bottom + 1],
            outline="black",
            width=2,
        )

    if exp == 'generalization':
        box_height = 120
        # Create a new white canvas
        canvas = Image.new("RGB",
                           (original.width, original.height + box_height),
                           "black")
        # Paste the original image
        canvas.paste(original, (0, 0))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("Times New Roman Bold.ttf", size=100)
        if result_type == 'Original':
            im = utils.normalize(
                plt.imread("data/Sky.png").astype(np.float32),
                True,
            )
            im = cv2.resize(im,
                            None,
                            fx=1 / 2,
                            fy=1 / 2,
                            interpolation=cv2.INTER_AREA)
            im_noisy = utils.normalize(
                plt.imread(image_path).astype(np.float32), True)
            im_noisy = im_noisy[:, :, :3]
            psnr = utils.psnr(im, im_noisy)
        else:
            psnr = get_PSNR(filepath)[0][0]
        text1 = f"PSNR: {psnr:.2f} dB"
        text_color = "white"
        text_position = (
            (canvas.height // 2) - 230,
            (canvas.height - box_height) + (box_height // 8),
        )
        # draw.text(text_position, text1, fill=text_color, font=font)
        # Add borders
        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                       outline="black",
                       width=2)

    elif exp == 'ct':
        box_height = 80
        # Create a new white canvas
        if result_type == 'Original':
            original = original.resize(
                (original.width // 2, original.height // 2))

        canvas = Image.new(original.mode,
                           (original.width, original.height + box_height),
                           'black')
        # Paste the original image
        canvas.paste(original, (0, 0))
        canvas.save("data/chest_test.png")
        draw = ImageDraw.Draw(canvas)
        if result_type != 'Original':
            font = ImageFont.truetype("Times New Roman Bold.ttf", size=28)
            psnr = get_PSNR(filepath)[0][0]
            ssim = get_SSIM(filepath)[0][0]

            text1 = f"PSNR: {psnr:.2f} dB"
            text2 = f"SSIM: {ssim:.3f}"
            text_color = "white"
            text_position = (
                (canvas.height // 2) - 115,
                (canvas.height - box_height) + (box_height // 8),
            )
            draw.text(text_position, text1, fill=text_color, font=font)
            draw.text((text_position[0], text_position[1] + 35),
                      text2,
                      fill=text_color,
                      font=font)

        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                       outline="black",
                       width=2)

    elif exp == 'sisr':
        box_height = 150

        font = ImageFont.truetype("Times New Roman Bold.ttf", size=60)
        if result_type == 'Original':
            original = original.resize(
                (original.width // 6, original.height // 6))
            canvas = Image.new("RGB",
                               (original.width, original.height + box_height),
                               "black")
            # Paste the original image
            canvas.paste(original, (0, 0))
            draw = ImageDraw.Draw(canvas)

            draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                           outline="black",
                           width=2)
        else:
            canvas = Image.new("RGB",
                               (original.width, original.height + box_height),
                               "black")
            # Paste the original image
            canvas.paste(original, (0, 0))
            draw = ImageDraw.Draw(canvas)

            mse = get_MSE(filepath)[0][0]
            ssim = get_SSIM(filepath)[0][0]

            text1 = f"MSE: {mse:.2f} dB"
            text2 = f"SSIM: {ssim:.3f}"
            text_color = "white"
            text_position = (
                (canvas.height // 2) - 470,
                (canvas.height - box_height) + (box_height // 25),
            )
            draw.text(text_position, text1, fill=text_color, font=font)
            draw.text((text_position[0], text_position[1] + 70),
                      text2,
                      fill=text_color,
                      font=font)

            draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                           outline="black",
                           width=2)

    elif exp == 'representation':
        box_height = 130
        # Create a new white canvas
        canvas = Image.new("RGB",
                           (original.width, original.height + box_height),
                           "black")
        # Paste the original image
        canvas.paste(original, (0, 0))

        # Create the zoomed inset (you'll need to adjust these coordinates)
        eye_region = original.crop((15, 750, 90, 825))
        eye_region = eye_region.resize((200, 200))
        canvas.paste(eye_region, (canvas.width - 250, 50))

        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("Times New Roman Bold.ttf", size=90)
        if result_type == 'Original':
            im = utils.normalize(
                plt.imread("data/Boat.png").astype(np.float32),
                True,
            )
            im = cv2.resize(im,
                            None,
                            fx=1 / 2,
                            fy=1 / 2,
                            interpolation=cv2.INTER_AREA)
            im_noisy = utils.normalize(
                plt.imread(image_path).astype(np.float32), True)
            im_noisy = im_noisy[:, :, :3]
            psnr = utils.psnr(im, im_noisy)
        else:
            psnr = get_PSNR(filepath)[0][0]

        text1 = f"PSNR: {psnr:.2f} dB"
        text_color = "white"
        text_position = (
            (canvas.height // 2) - 550,
            (canvas.height - box_height) + (box_height // 8),
        )
        # draw.text(text_position, text1, fill=text_color, font=font)

        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                       outline="black",
                       width=2)

        inset_size = (200, 200)
        inset_left = canvas.width - inset_size[0] - 50
        inset_top = 50
        inset_right = inset_left + inset_size[0]
        inset_bottom = inset_top + inset_size[1]
        draw.rectangle(
            [inset_left - 1, inset_top - 1, inset_right + 1, inset_bottom + 1],
            outline="black",
            width=2,
        )

    elif exp == 'multi_sisr':
        box_height = 140

        font = ImageFont.truetype("Times New Roman Bold.ttf", size=52)
        if result_type == 'Original':
            original = original.resize(
                (original.width // 2, original.height // 2))
            canvas = Image.new("RGB",
                               (original.width, original.height + box_height),
                               "black")
            # Paste the original image
            canvas.paste(original, (0, 0))
            draw = ImageDraw.Draw(canvas)

            draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                           outline="black",
                           width=2)
        else:
            canvas = Image.new("RGB",
                               (original.width, original.height + box_height),
                               "black")
            # Paste the original image
            canvas.paste(original, (0, 0))
            draw = ImageDraw.Draw(canvas)

            psnr, ssim = get_SISR_metrics(filepath)
            # mse = get_PSNR_(filepath)[0][0]
            # ssim = get_SSIM(filepath)[0][0]

            text1 = f"PSNR: {psnr:.2f} dB"
            text2 = f"SSIM: {ssim:.3f}"
            text_color = "white"
            text_position = (
                (canvas.height // 2) - 190,
                (canvas.height - box_height) + (box_height // 25),
            )
            draw.text(text_position, text1, fill=text_color, font=font)
            draw.text((text_position[0], text_position[1] + 70),
                      text2,
                      fill=text_color,
                      font=font)

        draw.rectangle([0, 0, canvas.width - 1, canvas.height - 1],
                       outline="black",
                       width=2)

    return canvas

# Use the function
filepath = "data/parrot.png"
exp = 'denoise'
result_type = 'Original1'
# filepath = "data_noisy/parrot_noisy_T2.0_snr1.png"
result = create_composite(filepath, exp, result_type)
if result_type == 'NotOriginal':
    result.save(os.path.join(filepath, "Result_img.png"))
else:
    result.save(os.path.join('data_noisy', "Parrot_Original.png"))

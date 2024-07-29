import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

im = Image.open('/rds/general/user/atk23/home/wire/data/parrot.png')  
img_gray = im.convert('L')
img_array = np.array(img_gray)

# Compute the 2D Fourier Transform of the image
f_transform = np.fft.fft2(img_gray)

# Shift the zero frequency component to the center
f_transform_shifted = np.fft.fftshift(f_transform)

# Compute the magnitude spectrum
magnitude_spectrum = np.abs(f_transform_shifted)

min_frequency = np.min(magnitude_spectrum)
max_frequency = np.max(magnitude_spectrum)
magnitude_spectrum_log = np.log1p(magnitude_spectrum)

print(f"Min frequency: {min_frequency}, Max frequency: {max_frequency}")

# Plot the original image and its magnitude spectrum
plt.figure(figsize=(12, 6))

# Plot the original image
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title('Original Image')
plt.axis('off')

# Plot the magnitude spectrum
plt.subplot(1, 2, 2)
vmin = np.min(magnitude_spectrum_log)
vmax = np.max(magnitude_spectrum_log)
img = plt.imshow(magnitude_spectrum_log, cmap='YlOrRd', vmin=vmin, vmax=vmax)  # Use log scale for better visualization
plt.title('Magnitude Spectrum')
plt.axis('off')
plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)

# Display the minimum and maximum frequencies on the plot
plt.text(0.5, -0.1, f"Min Frequency Magnitude: {min_frequency:.2e}\nMax Frequency Magnitude: {max_frequency:.2f}",
         fontsize=12, ha='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

plt.savefig('/rds/general/user/atk23/home/wire/data_freq_content/parrot_original.png', bbox_inches='tight')

# Clear the current figure
plt.clf()
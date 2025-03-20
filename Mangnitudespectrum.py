import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import matplotlib.image as mpimg
from google.colab import files

# Upload image
uploaded = files.upload()

# Assuming you upload a file named 'test.png', use the uploaded file name
image_path = list(uploaded.keys())[0]  # Get the first uploaded file name

# Load the image
img = mpimg.imread(image_path)

# If the image is colored (e.g., RGB), convert it to grayscale
if img.ndim == 3:
    img = np.mean(img, axis=-1)  # Convert to grayscale by averaging the channels

# Compute 2D Fourier Transform (Frequency domain)
F_uv = fft2(img)

# Compute magnitude spectrum (normal frequency rectangle, no shift)
magnitude_spectrum_normal = np.abs(F_uv)

# Compute magnitude spectrum (centered frequency rectangle, shift zero-frequency component)
F_uv_shifted = fftshift(F_uv)
magnitude_spectrum_centered = np.abs(F_uv_shifted)

# Plot both the magnitude spectrums
plt.figure(figsize=(12, 6))

# Plot the normal (uncentered) frequency spectrum
plt.subplot(1, 2, 1)
plt.imshow(np.log(magnitude_spectrum_normal + 1), cmap='gray')  # Log scale for better visualization
plt.title('Normal Frequency Magnitude Spectrum')
plt.colorbar()

# Plot the centered frequency spectrum
plt.subplot(1, 2, 2)
plt.imshow(np.log(magnitude_spectrum_centered + 1), cmap='gray')  # Log scale for better visualization
plt.title('Centered Frequency Magnitude Spectrum')
plt.colorbar()

plt.tight_layout()
plt.show()

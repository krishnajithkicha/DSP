from google.colab import files
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import matplotlib.image as mpimg

# Upload image
uploaded = files.upload()

# Assuming you upload a file named 'test.png', use the uploaded file name
image_path = list(uploaded.keys())[0]  # Get the first uploaded file name

# Load the image (if it's a color image, it will be converted to grayscale)
img = mpimg.imread(image_path)

# If the image is colored (e.g., RGB), convert it to grayscale
if img.ndim == 3:
    img = np.mean(img, axis=-1)  # Convert to grayscale by averaging the channels

# Compute 2D Fourier Transform (Frequency domain)
F_uv = fft2(img)

# Shift the zero-frequency component to the center
F_uv_shifted = fftshift(F_uv)

# Calculate energy in the spatial domain
energy_spatial = np.sum(np.abs(img)**2)

# Calculate energy in the frequency domain
energy_frequency = np.sum(np.abs(F_uv_shifted)**2) / img.size

# Print the results
print(f"Energy in spatial domain: {energy_spatial}")
print(f"Energy in frequency domain: {energy_frequency}")

# Verify Parseval's theorem (energy conservation)
assert np.isclose(energy_spatial, energy_frequency), "Energy conservation does not hold!"

# Optionally, visualize the image and its frequency domain representation
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
# Visualize the log of the absolute value of the shifted frequency domain
plt.imshow(np.log(np.abs(F_uv_shifted) + 1), cmap='gray')  # Log scale for better visualization
plt.title('Frequency Domain')

plt.show()

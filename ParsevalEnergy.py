import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# Load or create an image (using a sample 2D array for demonstration)
# For a real image, you can use plt.imread() or similar methods
image = np.random.random((256, 256))  # Example random image

# Compute 2D Fourier Transform (Frequency domain)
F_uv = fft2(image)

# Shift the zero-frequency component to the center
F_uv_shifted = fftshift(F_uv)

# Calculate energy in the spatial domain
energy_spatial = np.sum(np.abs(image)**2)

# Calculate energy in the frequency domain
# Normalize the Fourier transform by the number of pixels and compute energy
energy_frequency = np.sum(np.abs(F_uv_shifted)**2) / image.size

# Print the results
print(f"Energy in spatial domain: {energy_spatial}")
print(f"Energy in frequency domain: {energy_frequency}")

# Verify Parseval's theorem (energy conservation)
assert np.isclose(energy_spatial, energy_frequency), "Energy conservation does not hold!"

# Optionally, visualize the image and its frequency domain representation
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
# Visualize the log of the absolute value of the shifted frequency domain
plt.imshow(np.log(np.abs(F_uv_shifted) + 1), cmap='gray')  # Log scale for better visualization
plt.title('Frequency Domain')

plt.show()

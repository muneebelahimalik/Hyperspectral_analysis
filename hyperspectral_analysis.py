import spectral
import numpy as np
import matplotlib.pyplot as plt

# Paths to my pc
hdr_path = r"C:\Users\mm17889\OneDrive - University of Georgia\Hyperspectral Data\emptyname_2024-04-17_19-42-01\capture\emptyname_2024-04-17_19-42-01.hdr"

img = spectral.open_image(hdr_path) # getting hdr file (plus the .raw file as its opening the .hdr directory)


# loding into np array
data = img.load()  # should be like [rows, cols, bands]
print("Data shape (rows, cols, bands):", data.shape) # seeing if its right

# getting metadata from hdr file

if 'wavelength' in img.metadata:
    wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
    print("Wavelength array shape:", wavelengths.shape)
else:
    wavelengths = None
    print("No wavelength info found in header.")

#to find the nearest band index to a given wavelength
def find_band_index(target_nm):
    if wavelengths is None:
        raise ValueError("No wavelength data to match band indices.")
    return np.argmin(np.abs(wavelengths - target_nm))

#for quick analysis hardcoding these values
blue_wl  = 450
green_wl = 550
red_wl   = 660
nir_wl   = 850

if wavelengths is not None:
    blue_idx  = find_band_index(blue_wl)
    green_idx = find_band_index(green_wl)
    red_idx   = find_band_index(red_wl)
    nir_idx   = find_band_index(nir_wl)
else:
    blue_idx  = 10
    green_idx = 30
    red_idx   = 50
    nir_idx   = 90

print("Band indices (B, G, R, NIR):", blue_idx, green_idx, red_idx, nir_idx)

# plotting order is [R,G,B]
rgb_bands = [red_idx, green_idx, blue_idx]
rgb_img = data[:, :, rgb_bands]

rgb_max = np.percentile(rgb_img, 99)
rgb_norm = np.clip(rgb_img / rgb_max, 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(rgb_norm)
plt.title("RGB Composite")
plt.axis("off")
plt.show()

red_band = data[:, :, red_idx].astype(np.float32)
nir_band = data[:, :, nir_idx].astype(np.float32)

epsilon = 1e-8 # just so that we do not get 0 in the denominator
ndvi = (nir_band - red_band) / (nir_band + red_band + epsilon)

plt.figure(figsize=(6, 6))
plt.imshow(ndvi, cmap='RdYlGn')
plt.title("NDVI")
plt.colorbar(label="NDVI value")
plt.axis("off")
plt.show()

import rasterio
import numpy as np
from scipy.ndimage import sobel

# =========================
# 📂 PATHS
# =========================
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"

twi_out = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\twi.tif"
curvature_out = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\curvature.tif"

# =========================
# 📥 LOAD DEM
# =========================
with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    profile = src.profile

print("✅ DEM loaded")

# =========================
# 🧬 CURVATURE (Laplacian)
# =========================
dx = sobel(dem, axis=0)
dy = sobel(dem, axis=1)

curvature = sobel(dx, axis=0) + sobel(dy, axis=1)

# Normalize
curvature = (curvature - np.nanmin(curvature)) / (np.nanmax(curvature) - np.nanmin(curvature))

# =========================
# 💧 APPROX TWI (NO richdem)
# =========================

# Gradient (slope proxy)
grad_x = np.gradient(dem, axis=0)
grad_y = np.gradient(dem, axis=1)

slope = np.sqrt(grad_x**2 + grad_y**2)

# Flow accumulation proxy (simple smoothing)
flow = np.sqrt(sobel(dem, axis=0)**2 + sobel(dem, axis=1)**2)

# TWI approximation
twi = np.log((flow + 1) / (slope + 1e-5))

# Normalize
twi = (twi - np.nanmin(twi)) / (np.nanmax(twi) - np.nanmin(twi))

# =========================
# 💾 SAVE
# =========================
profile.update(dtype=rasterio.float32)

with rasterio.open(curvature_out, "w", **profile) as dst:
    dst.write(curvature.astype(np.float32), 1)

with rasterio.open(twi_out, "w", **profile) as dst:
    dst.write(twi.astype(np.float32), 1)

print("\n🚀 Curvature & TWI saved (no richdem)!")

#validation 1
import rasterio

with rasterio.open(twi_out) as src:
    twi = src.read(1)
    print("TWI min:", twi.min(), "max:", twi.max())

with rasterio.open(curvature_out) as src:
    curv = src.read(1)
    print("Curvature min:", curv.min(), "max:", curv.max())


    #validation 2
import rasterio

with rasterio.open(twi_out) as src:
    print("TWI OK:", src.shape)

with rasterio.open(curvature_out) as src:
    print("Curvature OK:", src.shape)
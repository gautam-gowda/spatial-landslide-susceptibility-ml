import numpy as np
import rasterio

dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"

aspect_sin_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_sin.tif"
aspect_cos_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_cos.tif"

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(float)
    transform = src.transform
    meta = src.meta.copy()
    nodata = src.nodata

# Handle nodata
if nodata is not None:
    dem[dem == nodata] = np.nan

# Pixel size
dx = transform[0]
dy = -transform[4]

# Gradients
dz_dx = np.gradient(dem, axis=1) / dx
dz_dy = np.gradient(dem, axis=0) / dy

# 🧭 Aspect (degrees)
aspect = np.degrees(np.arctan2(dz_dy, -dz_dx))
aspect = np.where(aspect < 0, 90.0 - aspect, 450.0 - aspect)
aspect = aspect % 360

# 🔥 Circular encoding
aspect_sin = np.sin(np.radians(aspect))
aspect_cos = np.cos(np.radians(aspect))

# Save metadata
meta.update(dtype=rasterio.float32, count=1)

# Save sin
with rasterio.open(aspect_sin_path, "w", **meta) as dst:
    dst.write(aspect_sin.astype(np.float32), 1)

# Save cos
with rasterio.open(aspect_cos_path, "w", **meta) as dst:
    dst.write(aspect_cos.astype(np.float32), 1)

print("✅ Aspect (sin, cos) computed and saved!")

#validation 
print("Aspect_sin min/max:", np.nanmin(aspect_sin), np.nanmax(aspect_sin))
print("Aspect_cos min/max:", np.nanmin(aspect_cos), np.nanmax(aspect_cos))
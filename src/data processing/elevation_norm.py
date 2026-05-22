import numpy as np
import rasterio
import os

# Paths
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"
output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\elevation_norm.tif"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with rasterio.open(dem_path) as src:
    elevation = src.read(1).astype(float)
    meta = src.meta.copy()
    nodata = src.nodata

# Handle nodata
if nodata is not None:
    elevation[elevation == nodata] = np.nan

# 🔥 Fix 1: remove negative artifacts
elevation[elevation < 0] = 0

# Normalize
max_val = np.nanmax(elevation)
elevation_norm = elevation / max_val

# Save
meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(elevation_norm.astype(np.float32), 1)

print("✅ Elevation normalized (corrected)!")

# Validation
print("\n🔍 ELEVATION VALIDATION")
print("Min:", np.nanmin(elevation_norm))
print("Max:", np.nanmax(elevation_norm))
print("Mean:", np.nanmean(elevation_norm))
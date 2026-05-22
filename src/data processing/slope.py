import numpy as np
import rasterio

slope_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\slope.tif"
slope_norm_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\slope_norm.tif"

with rasterio.open(slope_path) as src:
    slope = src.read(1).astype(float)
    meta = src.meta.copy()
    nodata = src.nodata

# Handle nodata
if nodata is not None:
    slope[slope == nodata] = np.nan

# ✅ Normalize (0–1 scale)
slope_norm = slope / 90.0

# Save
meta.update(dtype=rasterio.float32, count=1)

with rasterio.open(slope_norm_path, "w", **meta) as dst:
    dst.write(slope_norm.astype(np.float32), 1)

print("✅ Slope normalized and saved!")

#validation 
print("Slope_norm min:", np.nanmin(slope_norm))
print("Slope_norm max:", np.nanmax(slope_norm))
print("Slope_norm mean:", np.nanmean(slope_norm))
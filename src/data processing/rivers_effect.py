import numpy as np
import rasterio
import os

# Paths
river_dist_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\aligned_river_distance.tif"
output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\river_effect.tif"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with rasterio.open(river_dist_path) as src:
    distance = src.read(1).astype(float)
    meta = src.meta.copy()
    nodata = src.nodata

# Handle nodata
if nodata is not None:
    distance[distance == nodata] = np.nan

# 🔥 Exponential decay (IMPORTANT PARAMETER)
river_effect = np.exp(-distance / 10000)

# Save
meta.update(dtype=rasterio.float32, count=1)

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(river_effect.astype(np.float32), 1)

print("✅ River effect (exp decay) saved!")
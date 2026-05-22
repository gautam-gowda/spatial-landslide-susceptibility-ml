import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt

# Paths
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"
river_raw_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\river_raw\rivers.tif"

output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\aligned_river_distance.tif"

# Load DEM (reference)
with rasterio.open(dem_path) as dem_src:
    dem_meta = dem_src.meta.copy()
    dem_shape = (dem_src.height, dem_src.width)
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs

# Step 1: Reproject river mask to DEM grid FIRST
with rasterio.open(river_raw_path) as src:
    river_reproj = np.empty(dem_shape, dtype=np.float32)

    reproject(
        source=src.read(1),
        destination=river_reproj,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.nearest
    )

# Step 2: Binary mask
river_binary = (river_reproj == 1)

# Step 3: Distance in pixels
distance_pixels = distance_transform_edt(~river_binary)

# Step 4: Convert to meters (NOW CORRECT)
pixel_size = dem_transform[0]
distance_m = distance_pixels * pixel_size

# Save
dem_meta.update(dtype=rasterio.float32, count=1)

with rasterio.open(output_path, "w", **dem_meta) as dst:
    dst.write(distance_m.astype(np.float32), 1)

print("✅ River distance corrected!")

# Validation
print("Min:", np.nanmin(distance_m))
print("Max:", np.nanmax(distance_m))
print("Mean:", np.nanmean(distance_m))
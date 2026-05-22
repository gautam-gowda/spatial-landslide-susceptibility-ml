import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os

# 📂 Paths
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"

rainfall_files = [
    r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_raw\rainfall_r3_mean.tif",
    r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_raw\rainfall_r7_mean.tif"
]

output_folder = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_processed"
os.makedirs(output_folder, exist_ok=True)

# 📥 Load DEM (reference grid)
with rasterio.open(dem_path) as dem_src:
    dem_meta = dem_src.meta.copy()
    dem_shape = (dem_src.height, dem_src.width)
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs

# 🔄 Process each rainfall file
for rain_path in rainfall_files:
    filename = os.path.basename(rain_path)
    output_path = os.path.join(output_folder, f"aligned_{filename}")

    print(f"\n🌧 Processing: {filename}")

    with rasterio.open(rain_path) as src:
        rain_data = src.read(1)

        # Prepare empty array with DEM shape
        aligned = np.empty(dem_shape, dtype=np.float32)

        # 🔥 Reproject + Resample + Align
        reproject(
            source=rain_data,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

    # Save output
    dem_meta.update(dtype=rasterio.float32, count=1)

    with rasterio.open(output_path, "w", **dem_meta) as dst:
        dst.write(aligned, 1)

    print("   ✅ Saved:", output_path)

print("\n🚀 Rainfall aligned with DEM!")


#validation 
with rasterio.open(output_path) as src:
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Shape:", src.width, src.height)
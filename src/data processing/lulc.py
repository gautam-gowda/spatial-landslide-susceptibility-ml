import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import os

# =========================
# 📂 PATHS
# =========================
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"
lulc_raw_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\lulc_raw\lulc.tif"

output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\aligned_lulc.tif"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =========================
# 📥 LOAD DEM (REFERENCE GRID)
# =========================
with rasterio.open(dem_path) as dem_src:
    dem_meta = dem_src.meta.copy()
    dem_shape = (dem_src.height, dem_src.width)
    dem_transform = dem_src.transform
    dem_crs = dem_src.crs

print("✅ DEM loaded")

# =========================
# 🌱 LOAD + CLEAN LULC
# =========================
with rasterio.open(lulc_raw_path) as src:
    lulc = src.read(1).astype(float)

    # 🔥 STEP 1: Remove background (0 → NaN)
    lulc[lulc == 0] = np.nan

    # Prepare aligned array
    aligned_lulc = np.empty(dem_shape, dtype=np.float32)

    # 🔥 STEP 2: Reproject + Align + FIX nodata
    reproject(
        source=lulc,
        destination=aligned_lulc,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=Resampling.nearest,   # IMPORTANT for categorical
        dst_nodata=np.nan                # 🔥 prevents zeros coming back
    )

# =========================
# 💾 SAVE OUTPUT
# =========================
dem_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)

with rasterio.open(output_path, "w", **dem_meta) as dst:
    dst.write(aligned_lulc, 1)

print("✅ LULC aligned and saved!")

# =========================
# 🧪 VALIDATION
# =========================
print("\n🔍 LULC VALIDATION")

print("CRS:", dem_crs)
print("Shape:", aligned_lulc.shape)

print("Min:", np.nanmin(aligned_lulc))
print("Max:", np.nanmax(aligned_lulc))
print("Mean:", np.nanmean(aligned_lulc))

unique_vals = np.unique(aligned_lulc[~np.isnan(aligned_lulc)])
print("Unique sample:", unique_vals[:10])

# 🔥 Check for unwanted zeros
if 0 in unique_vals:
    print("⚠️ WARNING: Zero values still present!")
else:
    print("✅ No zero contamination — correct!")
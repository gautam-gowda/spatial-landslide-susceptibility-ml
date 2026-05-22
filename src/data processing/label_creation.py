#CSV → Filter → UTM (READY FOR RASTER)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# =========================
# 📂 PATH
# =========================
csv_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\landslides.csv"

# =========================
# 📥 LOAD CSV
# =========================
df = pd.read_csv(csv_path)

print("🌍 Total global records:", len(df))

# =========================
# 🌐 CREATE GEODATAFRAME
# =========================
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]

gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print("✅ GeoDataFrame created")

# =========================
# 📍 FILTER USING BOUNDING BOX (KARNATAKA)
# =========================
gdf_ka = gdf[
    (gdf["latitude"] >= 11.5) &
    (gdf["latitude"] <= 18.5) &
    (gdf["longitude"] >= 74.0) &
    (gdf["longitude"] <= 78.5)
]

print("🎯 Karnataka landslides (bbox):", len(gdf_ka))

# =========================
# 🚨 SAFETY CHECK
# =========================
if len(gdf_ka) == 0:
    raise ValueError("❌ No points found in Karnataka bounding box!")

# =========================
# 🔄 REPROJECT TO DEM CRS
# =========================
gdf_utm = gdf_ka.to_crs("EPSG:32643")

print("✅ Reprojected to EPSG:32643")

# =========================
# 📊 FINAL CHECK
# =========================
print("\n🔍 FINAL DATA CHECK")
print("CRS:", gdf_utm.crs)
print("Total points:", len(gdf_utm))
print(gdf_utm.head())




#BUFFER + RASTERIZE

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.features import rasterize
import numpy as np
import os

# =========================
# 📂 PATHS (KEEP r"...")
# =========================
csv_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\landslides.csv"
dem_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"
output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\landslide_labels.tif"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# =========================
# 📥 LOAD CSV
# =========================
df = pd.read_csv(csv_path)
print("🌍 Total global records:", len(df))

# =========================
# 🌐 CREATE GEODATAFRAME
# =========================
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

print("✅ GeoDataFrame created")

# =========================
# 📍 FILTER KARNATAKA (BBOX)
# =========================
gdf_ka = gdf[
    (gdf["latitude"] >= 11.5) &
    (gdf["latitude"] <= 18.5) &
    (gdf["longitude"] >= 74.0) &
    (gdf["longitude"] <= 78.5)
]

print("🎯 Karnataka landslides:", len(gdf_ka))

if len(gdf_ka) == 0:
    raise ValueError("❌ No points found in region!")

# =========================
# 🔄 REPROJECT TO DEM CRS
# =========================
gdf_utm = gdf_ka.to_crs("EPSG:32643")
print("✅ Reprojected to EPSG:32643")

# =========================
# 📥 LOAD DEM (REFERENCE GRID)
# =========================
with rasterio.open(dem_path) as src:
    transform = src.transform
    shape = (src.height, src.width)
    meta = src.meta.copy()

print("✅ DEM loaded")

# =========================
# 🔥 BUFFER LANDSLIDE POINTS
# =========================
buffer_distance = 120  # meters

gdf_buffered = gdf_utm.copy()
gdf_buffered["geometry"] = gdf_buffered.geometry.buffer(buffer_distance)

print("✅ Buffer applied")

# =========================
# 🔄 RASTERIZE
# =========================
landslide_raster = rasterize(
    [(geom, 1) for geom in gdf_buffered.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# =========================
# 💾 SAVE OUTPUT (FIXED METADATA)
# =========================
meta.update(
    dtype="uint8",
    count=1,
    nodata=0   # 🔥 FIXED
)

with rasterio.open(output_path, "w", **meta) as dst:
    dst.write(landslide_raster, 1)

print("🚀 Landslide raster created!")
print("📁 Saved at:", output_path)

# =========================
# 🧪 VALIDATION
# =========================
print("\n🔍 VALIDATION")

print("Min:", landslide_raster.min())
print("Max:", landslide_raster.max())

landslide_pixels = np.sum(landslide_raster == 1)
print("Total landslide pixels:", landslide_pixels)




#load , sample and create a final dataset

import rasterio
import numpy as np
import pandas as pd

# =========================
# 📂 PATHS
# =========================
files = {
    "slope": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\slope_norm.tif",
    "aspect_sin": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_sin.tif",
    "aspect_cos": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_cos.tif",
    "rain_r3": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_processed\aligned_rainfall_r3_mean.tif",
    "rain_r7": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_processed\aligned_rainfall_r7_mean.tif",
    "elevation": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\elevation_norm.tif",
    "lulc": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\aligned_lulc.tif",
    "river": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\river_effect.tif",
    "label": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\landslide_labels.tif"
}

print("🔄 Loading rasters into memory...")

arrays = {}
for name, path in files.items():
    with rasterio.open(path) as src:
        arr = src.read(1)

        if src.nodata is not None:
            arr[arr == src.nodata] = 0

        arrays[name] = arr
        print(f"✅ {name} loaded")

# =========================
# 🌋 GET LANDSLIDE PIXELS
# =========================
label = arrays["label"]

landslide_idx = np.argwhere(label == 1)
non_idx = np.argwhere(label == 0)

print("🌋 Landslide pixels:", len(landslide_idx))

# =========================
# 🎯 BALANCED SAMPLING
# =========================
sample_size = len(landslide_idx)

non_sampled = non_idx[
    np.random.choice(len(non_idx), sample_size, replace=False)
]

all_idx = np.vstack((landslide_idx, non_sampled))

print("🎯 Total samples:", len(all_idx))

# =========================
# 🧬 EXTRACT FEATURES (FAST)
# =========================
data = []

for r, c in all_idx:
    row = [arrays[name][r, c] for name in files.keys()]
    data.append(row)

print("✅ Extraction complete")

# =========================
# 💾 SAVE
# =========================
df = pd.DataFrame(data, columns=list(files.keys()))

output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\final_dataset.csv"
df.to_csv(output_path, index=False)

print("\n🚀 FINAL DATASET READY!")
print(df["label"].value_counts())
import glob
import rasterio
from rasterio.merge import merge
import os

input_folder = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\dem_utm"
output_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load all tiles
dem_files = glob.glob(os.path.join(input_folder, "*.tif"))

print(f"🔍 Found {len(dem_files)} UTM tiles")

src_files = []
for f in dem_files:
    src = rasterio.open(f)
    src_files.append(src)

# Merge
mosaic, transform = merge(src_files)

# Metadata
out_meta = src_files[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": transform
})

# Save
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

print("✅ DEM merged successfully")
print("📍 Saved at:", output_path)



#validation 
merged_path = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\dem_merged.tif"

with rasterio.open(merged_path) as src:
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Shape:", src.width, src.height)
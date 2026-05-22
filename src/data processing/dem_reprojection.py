import os
import glob
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

# 📂 Paths
input_folder = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\dem_raw\dem"
output_folder = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\dem_utm"

os.makedirs(output_folder, exist_ok=True)

# 🎯 Target CRS (Karnataka UTM)
dst_crs = "EPSG:32643"

# 📄 Get all tiles
dem_files = glob.glob(os.path.join(input_folder, "*.tif"))

print(f"🔍 Found {len(dem_files)} DEM tiles")

if not dem_files:
    raise Exception("❌ No DEM files found!")

# 🔄 Process each tile
for dem_path in dem_files:
    filename = os.path.basename(dem_path)
    output_path = os.path.join(output_folder, f"utm_{filename}")

    print(f"\n📦 Processing: {filename}")

    try:
        with rasterio.open(dem_path) as src:

            # Skip if already projected
            if src.crs.to_string() == dst_crs:
                print("⚠️ Already in UTM, skipping")
                continue

            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update({
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open(output_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        # ✅ VALIDATION STEP (IMPORTANT)
        with rasterio.open(output_path) as check:
            print("   ✅ CRS:", check.crs)
            print("   📏 Resolution:", check.res)

            # ⚠️ Reviewer check
            if check.res[0] > 100:
                print("   ⚠️ WARNING: Resolution too large!")
            if "4326" in str(check.crs):
                raise Exception("❌ Still in degrees!")

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

print("\n🚀 All tiles processed!")
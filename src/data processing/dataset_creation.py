import rasterio
import numpy as np
import pandas as pd
import os

# =========================================================
# 📂 FILE PATHS
# =========================================================
files = {
    "slope": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\slope_norm.tif",

    "aspect_sin": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_sin.tif",

    "aspect_cos": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\aspect_cos.tif",

    # 🌧️ Keep ONLY 7-day rainfall
    "rain_r7": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\rainfall_processed\aligned_rainfall_r7_mean.tif",

    "elevation": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\merged\elevation_norm.tif",

    "river": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\river_effect.tif",

    "twi": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\twi.tif",

    "curvature": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\curvature.tif",

    "label": r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\landslide_real.tif"
}

# =========================================================
# 💾 OUTPUT CSV
# =========================================================
output_csv = (
    r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\final_dataset_real.csv"
)

# =========================================================
# 🔍 CHECK FILES
# =========================================================
print("🔍 Checking raster files...\n")

for name, path in files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing file:\n{path}")

    print(f"✅ {name}")

# =========================================================
# 📥 LOAD RASTERS
# =========================================================
arrays = {}

print("\n🔄 Loading rasters...\n")

for name, path in files.items():

    with rasterio.open(path) as src:

        arr = src.read(1).astype(np.float32)

        # Handle nodata
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        # Replace infinities
        arr[np.isinf(arr)] = np.nan

        arrays[name] = arr

        print(f"✅ Loaded: {name}")

# =========================================================
# 📐 CHECK SHAPE
# =========================================================
height, width = arrays["label"].shape

print(f"\n📐 Raster size: {height} x {width}")

# =========================================================
# 🏷️ CLEAN LABEL
# =========================================================
label = arrays["label"]

# Replace NaN labels with 0
label = np.nan_to_num(label, nan=0)

arrays["label"] = label

# =========================================================
# 🌋 GET LANDSLIDE / NON-LANDSLIDE PIXELS
# =========================================================
landslide_idx = np.argwhere(label == 1)

non_idx = np.argwhere(label == 0)

print(f"\n🌋 Landslide pixels: {len(landslide_idx)}")
print(f"🌍 Non-landslide pixels: {len(non_idx)}")

# =========================================================
# 🎯 INITIAL BALANCED SAMPLING
# =========================================================
np.random.seed(42)

n_ls = len(landslide_idx)

non_sampled = non_idx[
    np.random.choice(len(non_idx), n_ls, replace=False)
]

all_idx = np.vstack((landslide_idx, non_sampled))

print(f"\n🎯 Initial balanced samples: {len(all_idx)}")

# =========================================================
# 🧬 FEATURE EXTRACTION
# =========================================================

# ✅ Use ONLY features (exclude label)
feature_names = [k for k in files.keys() if k != "label"]

data = []

print("\n🔄 Extracting features...\n")

for r, c in all_idx:

    row_data = {
        "row": r,
        "col": c
    }

    valid_pixel = True

    # -----------------------------
    # Extract feature values
    # -----------------------------
    for name in feature_names:

        val = arrays[name][r, c]

        # Skip invalid pixels
        if np.isnan(val):
            valid_pixel = False
            break

        row_data[name] = val

    # -----------------------------
    # Add label separately
    # -----------------------------
    if valid_pixel:

        row_data["label"] = arrays["label"][r, c]

        data.append(row_data)

print("✅ Feature extraction complete")

# =========================================================
# 📊 CREATE DATAFRAME
# =========================================================
df = pd.DataFrame(data)

# =========================================================
# 🔍 CHECK NaNs
# =========================================================
print("\n🔍 NaN Summary:\n")
print(df.isna().sum())

# =========================================================
# 🧹 REMOVE INVALID ROWS
# =========================================================
before_clean = len(df)

df = df.dropna()

after_clean = len(df)

print(f"\n🧹 Removed invalid samples: {before_clean - after_clean}")

# =========================================================
# 🎯 RE-BALANCE AFTER CLEANING
# =========================================================
landslide_df = df[df["label"] == 1]

non_df = df[df["label"] == 0]

n_final = min(len(landslide_df), len(non_df))

landslide_df = landslide_df.sample(
    n=n_final,
    random_state=42
)

non_df = non_df.sample(
    n=n_final,
    random_state=42
)

# Combine
df = pd.concat([landslide_df, non_df])

# Shuffle
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\n🎯 Final balanced dataset:")
print(df["label"].value_counts())

# =========================================================
# 📊 FEATURE STATISTICS
# =========================================================
print("\n📊 Feature Statistics:\n")
print(df.describe())

# =========================================================
# 📊 FINAL DATASET INFO
# =========================================================
print("\n📊 Final dataset shape:", df.shape)

# =========================================================
# 💾 SAVE DATASET
# =========================================================
df.to_csv(output_csv, index=False)

print("\n🚀 CLEAN DATASET READY!")
print(f"📁 Saved at:\n{output_csv}")

# =========================================================
# ✅ IMPORTANT FOR TRAINING
# =========================================================
print("\n⚠️ IMPORTANT:")
print("When training models, REMOVE row and col:")
print('X = df.drop(columns=["label", "row", "col"])')
print('y = df["label"]')
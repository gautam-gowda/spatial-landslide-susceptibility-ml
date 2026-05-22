import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# =========================================================
# 🌍 BASE DIRECTORY
# =========================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# =========================================================
# 📂 INPUT RASTER
# =========================================================
raster_path = os.path.join(
    BASE_DIR,
    "outputs",
    "raster",
    "susceptibility_full.tif"
)

# =========================================================
# 🛰️ LOAD RASTER
# =========================================================
with rasterio.open(raster_path) as src:
    susceptibility = src.read(1)

# =========================================================
# 🚫 HANDLE INVALID VALUES
# =========================================================
susceptibility = np.where(
    susceptibility < 0,
    np.nan,
    susceptibility
)

# =========================================================
# 🎯 REMOVE LOW-CONFIDENCE PIXELS
# Removes ugly green padding + raster seams
# =========================================================
susceptibility[susceptibility < 0.22] = np.nan

# =========================================================
# 🎨 IEEE-STYLE SOFT COLORMAP
# =========================================================
colors = [
    "#d9f0d3",   # light green
    "#a6d96a",
    "#fefcbf",   # pale yellow
    "#fdae61",   # orange
    "#d73027"    # deep red
]

cmap = LinearSegmentedColormap.from_list(
    "landslide",
    colors,
    N=256
)

# =========================================================
# 📊 CREATE FIGURE
# =========================================================
fig, ax = plt.subplots(
    figsize=(8, 9),
    facecolor="white"
)

# =========================================================
# 🌍 DISPLAY MAP
# =========================================================
img = ax.imshow(
    susceptibility,
    cmap=cmap,
    interpolation="bilinear"
)

# =========================================================
# 🗺️ REMOVE AXES
# =========================================================
ax.axis("off")

# =========================================================
# 📝 TITLE
# =========================================================
ax.set_title(
    "AI-Based Landslide Susceptibility Map\nWestern Ghats, India",
    fontsize=14,
    fontweight="bold",
    pad=10
)

# =========================================================
# 🌈 COLORBAR
# =========================================================
cbar = fig.colorbar(
    img,
    ax=ax,
    fraction=0.046,
    pad=0.03
)

cbar.set_label(
    "Landslide Susceptibility",
    fontsize=12,
    fontweight="bold",
    labelpad=14
)

cbar.set_ticks([
    0.25,
    0.45,
    0.65,
    0.85
])

cbar.set_ticklabels([
    "Low",
    "Moderate",
    "High",
    "Very High"
])

cbar.ax.tick_params(
    labelsize=10
)

# =========================================================
# ✨ CLEAN LAYOUT
# =========================================================
fig.patch.set_facecolor("white")

plt.subplots_adjust(
    left=0.02,
    right=0.88,
    top=0.93,
    bottom=0.02
)

# =========================================================
# 💾 OUTPUT DIRECTORY
# =========================================================
output_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# =========================================================
# 💾 SAVE FIGURE
# =========================================================
output_path = os.path.join(
    output_dir,
    "final_ieee_susceptibility_map.png"
)

plt.savefig(
    output_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\n✅ FINAL IEEE-STYLE SUSCEPTIBILITY MAP SAVED")
print(output_path)
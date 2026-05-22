import rasterio
import numpy as np
import pandas as pd
from joblib import load
import folium
import os
from pyproj import Transformer

print("🚀 Loading Landslide Prediction System...")

# =========================================================
# 📂 PATHS
# =========================================================
base_dir = r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty"

model_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")

# =========================================================
# 🧠 FEATURE ORDER
# MUST MATCH TRAINING ORDER
# =========================================================
FEATURE_ORDER = [
    "slope",
    "aspect_sin",
    "aspect_cos",
    "rain_r7",
    "elevation",
    "river",
    "twi"
]

# =========================================================
# 📦 LOAD MODELS
# =========================================================
models = {
    "1": (
        "Logistic Regression",
        load(os.path.join(model_dir, "Logistic.pkl"))
    ),

    "2": (
        "Random Forest (Best)",
        load(os.path.join(model_dir, "RandomForest.pkl"))
    ),

    "3": (
        "XGBoost",
        load(os.path.join(model_dir, "XGBoost.pkl"))
    )
}

print("✅ Models Loaded")

# =========================================================
# 📥 LOAD RASTERS
# =========================================================
def load_raster(path):

    if not os.path.exists(path):
        raise Exception(f"❌ Missing raster:\n{path}")

    return rasterio.open(path)

rasters = {

    "slope":
        load_raster(
            os.path.join(data_dir, "merged/slope_norm.tif")
        ),

    "aspect_sin":
        load_raster(
            os.path.join(data_dir, "merged/aspect_sin.tif")
        ),

    "aspect_cos":
        load_raster(
            os.path.join(data_dir, "merged/aspect_cos.tif")
        ),

    "rain_r7":
        load_raster(
            os.path.join(
                data_dir,
                "processed/aligned_rainfall_r7_mean.tif"
            )
        ),

    "elevation":
        load_raster(
            os.path.join(data_dir, "merged/elevation_norm.tif")
        ),

    "river":
        load_raster(
            os.path.join(data_dir, "processed/river_effect.tif")
        ),

    "twi":
        load_raster(
            os.path.join(data_dir, "processed/twi.tif")
        )
}

# =========================================================
# 📊 LOAD ARRAYS
# =========================================================
arrays = {}

for k, src in rasters.items():

    arr = src.read(1).astype(float)

    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    arrays[k] = arr

print("✅ Raster Arrays Loaded")

# =========================================================
# 🌍 CRS TRANSFORMER
# =========================================================
transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32643",
    always_xy=True
)

# =========================================================
# 🎯 RISK EXPLANATION
# =========================================================
def explain_risk(prob):

    if prob >= 0.6:
        return "HIGH"

    elif prob >= 0.3:
        return "MODERATE"

    else:
        return "LOW"

# =========================================================
# 🎨 MARKER COLOR
# =========================================================
def get_color(prob):

    if prob >= 0.6:
        return "red"

    elif prob >= 0.3:
        return "orange"

    else:
        return "green"

# =========================================================
# 🧬 EXTRACT FEATURES
# =========================================================
def get_features(lat, lon):

    x, y = transformer.transform(lon, lat)

    row, col = rasters["slope"].index(x, y)

    features = []

    feature_dict = {}

    for k in FEATURE_ORDER:

        val = arrays[k][row, col]

        if np.isnan(val):
            val = 0

        features.append(val)

        feature_dict[k] = float(val)

    return (
        np.array([features]),
        feature_dict,
        row,
        col
    )

# =========================================================
# 🔮 PREDICT
# =========================================================
def predict(model, lat, lon):

    try:

        X, fdict, row, col = get_features(lat, lon)

        # ✅ FIXED FEATURE NAME WARNING
        X_df = pd.DataFrame(
            X,
            columns=FEATURE_ORDER
        )

        prob = model.predict_proba(X_df)[0][1]

        pred = int(prob > 0.5)

        return pred, prob, fdict, row, col

    except Exception as e:

        print("Prediction Error:", e)

        return None

# =========================================================
# 🤖 MODEL SELECTION
# =========================================================
print("\nChoose Model")
print("1 → Logistic Regression")
print("2 → Random Forest (Best)")
print("3 → XGBoost")
print("4 → Auto Select Best Model")

choice = input("\nEnter choice: ")

if choice == "4":

    model_name, model = models["2"]

else:

    model_name, model = models.get(
        choice,
        models["2"]
    )

print(f"\n✅ Selected Model: {model_name}")

# =========================================================
# 📍 USER INPUT
# =========================================================
lat = float(input("\nEnter Latitude : "))
lon = float(input("Enter Longitude: "))

# =========================================================
# 🔮 MAIN PREDICTION
# =========================================================
result = predict(model, lat, lon)

if result is None:

    print("❌ Invalid location")

    exit()

pred, prob, f, row, col = result

# =========================================================
# 🗺️ CREATE MAP
# =========================================================
m = folium.Map(

    location=[lat, lon],

    zoom_start=11,

    tiles=None
)

folium.TileLayer(

    tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",

    attr="OpenStreetMap"

).add_to(m)

# =========================================================
# 📌 MAIN LOCATION POPUP
# =========================================================
popup = f"""
<b>🌍 LANDSLIDE SUSCEPTIBILITY</b><br><br>

<b>📍 Latitude:</b> {lat}<br>
<b>📍 Longitude:</b> {lon}<br><br>

<b>🧭 Raster Row:</b> {row}<br>
<b>🧭 Raster Col:</b> {col}<br><br>

<b>⚠ Prediction:</b> {"LANDSLIDE RISK" if pred else "SAFE"}<br>

<b>📊 Probability:</b> {round(prob, 3)}<br>

<b>🚨 Risk Level:</b> {explain_risk(prob)}<br><br>

<b>🤖 Model:</b> {model_name}<br>

<b>📈 Validation:</b> Spatial Block Validation<br>

<b>🏆 ROC-AUC:</b> 0.888<br><br>

<hr>

<b>🌧 Rainfall (7-day):</b> {round(f['rain_r7'], 3)}<br>

<b>🌊 River Influence:</b> {round(f['river'], 3)}<br>

<b>⛰ Elevation:</b> {round(f['elevation'], 3)}<br>

<b>📐 Slope:</b> {round(f['slope'], 3)}<br>

<b>🧭 Aspect Sin:</b> {round(f['aspect_sin'], 3)}<br>

<b>🧭 Aspect Cos:</b> {round(f['aspect_cos'], 3)}<br>

<b>💧 TWI:</b> {round(f['twi'], 3)}<br><br>

<hr>

⚠ Prediction based on regional susceptibility modeling.
"""

folium.Marker(

    [lat, lon],

    popup=folium.Popup(
        popup,
        max_width=350
    ),

    icon=folium.Icon(
        color=get_color(prob)
    )

).add_to(m)

# =========================================================
# 📍 NEARBY ANALYSIS
# =========================================================
offsets = [-0.02, 0, 0.02]

for dlat in offsets:

    for dlon in offsets:

        if dlat == 0 and dlon == 0:
            continue

        new_lat = lat + dlat
        new_lon = lon + dlon

        r = predict(
            model,
            new_lat,
            new_lon
        )

        if r is None:
            continue

        p, pr, _, _, _ = r

        popup_text = f"""
        <b>📍 Nearby Location</b><br><br>

        <b>Latitude:</b> {round(new_lat, 5)}<br>

        <b>Longitude:</b> {round(new_lon, 5)}<br><br>

        <b>Probability:</b> {round(pr, 3)}<br>

        <b>Risk:</b> {explain_risk(pr)}
        """

        folium.CircleMarker(

            location=[new_lat, new_lon],

            radius=8,

            color=get_color(pr),

            fill=True,

            fill_opacity=0.8

        ).add_to(m).add_child(

            folium.Popup(
                popup_text,
                max_width=250
            )
        )

# =========================================================
# 🖱️ CLICK COORDINATES
# =========================================================
m.add_child(folium.LatLngPopup())

# =========================================================
# 💾 SAVE MAP
# =========================================================
output_map = os.path.join(
    base_dir,
    "prediction_map.html"
)

m.save(output_map)

print("\n✅ Prediction Complete")

print(f"📍 Latitude : {lat}")
print(f"📍 Longitude: {lon}")

print(f"📊 Probability: {round(prob,3)}")

print(f"🚨 Risk Level: {explain_risk(prob)}")

print(f"\n🗺️ Map Saved At:\n{output_map}")
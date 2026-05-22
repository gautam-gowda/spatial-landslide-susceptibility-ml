import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# =========================================================
# 📂 LOAD CLEAN DATASET
# =========================================================
data_path = (
    r"C:\Users\gauta\OneDrive\Desktop\landslide susceptibilty\data\processed\final_dataset_real.csv"
)

df = pd.read_csv(data_path)

print("📊 Dataset shape:", df.shape)

# =========================================================
# 🌍 SPATIAL VALIDATION SPLIT
# =========================================================
# Spatial split using geographic column position

split_col = df["col"].median()

train_df = df[df["col"] <= split_col]
test_df  = df[df["col"] > split_col]

print("\n🌍 Spatial Split")
print("Train:", train_df.shape)
print("Test :", test_df.shape)

# =========================================================
# 🧬 FEATURES
# =========================================================
# 🚨 IMPORTANT:
# row/col removed to prevent spatial leakage

feature_cols = [
    "slope",
    "aspect_sin",
    "aspect_cos",
    "rain_r7",
    "elevation",
    "river",
    "twi"
]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

print("\n🧬 Features Used:")
print(feature_cols)

# =========================================================
# 🌲 TRAIN RANDOM FOREST
# =========================================================
print("\n🌲 Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("✅ Random Forest trained successfully")

# =========================================================
# 🧠 CREATE SHAP EXPLAINER
# =========================================================
print("\n🧠 Generating SHAP values...")

explainer = shap.TreeExplainer(rf)

# SHAP values for binary classification
shap_values = explainer.shap_values(X_test)

print("✅ SHAP values generated")

# =========================================================
# 📊 SHAP SUMMARY PLOT
# =========================================================
print("\n📊 Displaying SHAP summary plot...")

# For Random Forest binary classification:
# use class 1 SHAP values
shap.summary_plot(
    shap_values[:, :, 1],
    X_test
)

# =========================================================
# 📈 SHAP FEATURE IMPORTANCE BAR PLOT
# =========================================================
print("\n📈 Displaying SHAP feature importance...")

shap.summary_plot(
    shap_values[:, :, 1],
    X_test,
    plot_type="bar"
)

# =========================================================
# 🌧️ RAINFALL DEPENDENCE PLOT
# =========================================================
print("\n🌧️ Rainfall dependence plot...")

shap.dependence_plot(
    "rain_r7",
    shap_values[:, :, 1],
    X_test
)

# =========================================================
# 🏔️ ELEVATION DEPENDENCE PLOT
# =========================================================
print("\n🏔️ Elevation dependence plot...")

shap.dependence_plot(
    "elevation",
    shap_values[:, :, 1],
    X_test
)

# =========================================================
# 🌊 RIVER DEPENDENCE PLOT
# =========================================================
print("\n🌊 River dependence plot...")

shap.dependence_plot(
    "river",
    shap_values[:, :, 1],
    X_test
)

# =========================================================
# 💧 TWI DEPENDENCE PLOT
# =========================================================
print("\n💧 TWI dependence plot...")

shap.dependence_plot(
    "twi",
    shap_values[:, :, 1],
    X_test
)

print("\n🚀 RANDOM FOREST SHAP ANALYSIS COMPLETE!")
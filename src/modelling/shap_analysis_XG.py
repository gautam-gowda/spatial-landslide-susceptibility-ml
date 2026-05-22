import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

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
# Split geographically using column position

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
# 🔵 TRAIN XGBOOST
# =========================================================
print("\n🔵 Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

print("✅ XGBoost trained successfully")

# =========================================================
# 🧠 CREATE SHAP EXPLAINER
# =========================================================
print("\n🧠 Generating SHAP values...")

explainer = shap.TreeExplainer(xgb)

shap_values = explainer.shap_values(X_test)

print("✅ SHAP values generated")

# =========================================================
# 📊 SHAP SUMMARY PLOT
# =========================================================
print("\n📊 Displaying SHAP summary plot...")

shap.summary_plot(
    shap_values,
    X_test
)

# =========================================================
# 📈 SHAP BAR PLOT
# =========================================================
print("\n📈 Displaying SHAP feature importance...")

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar"
)

# =========================================================
# 🔍 OPTIONAL: SINGLE FEATURE DEPENDENCE PLOTS
# =========================================================

# 🌧️ Rainfall influence
shap.dependence_plot(
    "rain_r7",
    shap_values,
    X_test
)

# 🏔️ Elevation influence
shap.dependence_plot(
    "elevation",
    shap_values,
    X_test
)

# 🌊 River influence
shap.dependence_plot(
    "river",
    shap_values,
    X_test
)

# 💧 TWI influence
shap.dependence_plot(
    "twi",
    shap_values,
    X_test
)

print("\n🚀 SHAP analysis complete!")
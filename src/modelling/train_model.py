import pandas as pd
import os

from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from joblib import dump

# =========================================================
# 📂 BASE DIRECTORY
# =========================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# =========================================================
# 📂 LOAD DATASET
# =========================================================
data_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_dataset_real.csv"
)

df = pd.read_csv(data_path)

print("📊 Dataset shape:", df.shape)

# =========================================================
# 🌍 SPATIAL BLOCK VALIDATION SPLIT
# =========================================================
BLOCK_SIZE = 1000

df["block_x"] = df["col"] // BLOCK_SIZE
df["block_y"] = df["row"] // BLOCK_SIZE

unique_blocks = (
    df[["block_x", "block_y"]]
    .drop_duplicates()
)

unique_blocks = unique_blocks.sample(
    frac=1,
    random_state=42
)

split_idx = int(len(unique_blocks) * 0.7)

train_blocks = unique_blocks.iloc[:split_idx]
test_blocks = unique_blocks.iloc[split_idx:]

train_keys = set(
    zip(
        train_blocks.block_x,
        train_blocks.block_y
    )
)

test_keys = set(
    zip(
        test_blocks.block_x,
        test_blocks.block_y
    )
)

train_mask = df.apply(
    lambda x:
    (x["block_x"], x["block_y"]) in train_keys,
    axis=1
)

test_mask = df.apply(
    lambda x:
    (x["block_x"], x["block_y"]) in test_keys,
    axis=1
)

train_df = df[train_mask]
test_df = df[test_mask]

print("\n🌍 Spatial Block Validation")
print("Train:", train_df.shape)
print("Test :", test_df.shape)

# =========================================================
# 🧬 FEATURES
# =========================================================
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
# 🟡 LOGISTIC REGRESSION
# =========================================================
print("\n🟡 Training Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_preds = lr.predict(X_test)

lr_f1 = f1_score(y_test, lr_preds)

# =========================================================
# 🟢 RANDOM FOREST
# =========================================================
print("🟢 Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

rf_f1 = f1_score(y_test, rf_preds)

# =========================================================
# 🔵 XGBOOST
# =========================================================
print("🔵 Training XGBoost...")

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

xgb_preds = xgb.predict(X_test)

xgb_f1 = f1_score(y_test, xgb_preds)

# =========================================================
# 📊 F1 SCORES
# =========================================================
print("\n📊 F1 SCORES")
print("-" * 40)

print(f"Logistic Regression : {lr_f1:.4f}")
print(f"Random Forest       : {rf_f1:.4f}")
print(f"XGBoost             : {xgb_f1:.4f}")

# =========================================================
# 💾 SAVE MODELS
# =========================================================
model_dir = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(model_dir, exist_ok=True)

dump(
    lr,
    os.path.join(model_dir, "Logistic.pkl")
)

dump(
    rf,
    os.path.join(model_dir, "RandomForest.pkl")
)

dump(
    xgb,
    os.path.join(model_dir, "XGBoost.pkl")
)

print("\n💾 Models Saved Successfully!")

# =========================================================
# 🏆 BEST MODEL (F1)
# =========================================================
results = {
    "Logistic Regression": lr_f1,
    "Random Forest": rf_f1,
    "XGBoost": xgb_f1
}

best_model = max(results, key=results.get)

print("\n🏆 BEST MODEL BASED ON F1-SCORE:")
print(f"{best_model} ({results[best_model]:.4f})")
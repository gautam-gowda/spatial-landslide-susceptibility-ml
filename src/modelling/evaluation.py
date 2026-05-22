import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
# 🌍 SPATIAL BLOCK VALIDATION
# =========================================================

BLOCK_SIZE = 1000

# Create spatial blocks
df["block_x"] = df["col"] // BLOCK_SIZE
df["block_y"] = df["row"] // BLOCK_SIZE

# Unique blocks
unique_blocks = (
    df[["block_x", "block_y"]]
    .drop_duplicates()
)

# Randomize blocks
unique_blocks = unique_blocks.sample(
    frac=1,
    random_state=42
)

# 70% train blocks
split_idx = int(len(unique_blocks) * 0.7)

train_blocks = unique_blocks.iloc[:split_idx]
test_blocks  = unique_blocks.iloc[split_idx:]

# Merge keys
train_keys = set(
    zip(train_blocks.block_x,
        train_blocks.block_y)
)

test_keys = set(
    zip(test_blocks.block_x,
        test_blocks.block_y)
)

# Assign train/test
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
test_df  = df[test_mask]

print("\n🌍 Spatial Block Validation")
print("Train:", train_df.shape)
print("Test:", test_df.shape)

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
# 🟡 LOGISTIC REGRESSION
# =========================================================
print("\n🟡 Training Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

lr_probs = lr.predict_proba(X_test)[:, 1]
lr_preds = lr.predict(X_test)

lr_auc = roc_auc_score(y_test, lr_probs)

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

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

rf_auc = roc_auc_score(y_test, rf_probs)

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

xgb_probs = xgb.predict_proba(X_test)[:, 1]
xgb_preds = xgb.predict(X_test)

xgb_auc = roc_auc_score(y_test, xgb_probs)

# =========================================================
# 🧪 ABLATION STUDY
# =========================================================

print("\n🧪 RUNNING ABLATION STUDY...\n")

ablation_features = [
    "rain_r7",
    "river",
    "elevation",
    "slope"
]

ablation_results = []

# =========================================================
# FULL MODEL BASELINE
# =========================================================

baseline_features = [
    col for col in X_train.columns
]

rf_full = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_full.fit(
    X_train[baseline_features],
    y_train
)

full_probs = rf_full.predict_proba(
    X_test[baseline_features]
)[:, 1]

full_auc = roc_auc_score(
    y_test,
    full_probs
)

ablation_results.append(
    ("Full Model", full_auc)
)

print(f"✅ Full Model AUC: {full_auc:.3f}")

# =========================================================
# FEATURE REMOVAL TESTS
# =========================================================

for feature in ablation_features:

    print(f"\n🚫 Removing: {feature}")

    selected_features = [
        col for col in X_train.columns
        if col != feature
    ]

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(
        X_train[selected_features],
        y_train
    )

    probs = rf.predict_proba(
        X_test[selected_features]
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probs
    )

    ablation_results.append(
        (f"No {feature}", auc)
    )

    print(f"ROC-AUC: {auc:.3f}")

# =========================================================
# 📊 RESULTS TABLE
# =========================================================

ablation_df = pd.DataFrame(
    ablation_results,
    columns=["Experiment", "ROC_AUC"]
)

print("\n📊 Ablation Results:")
print(ablation_df)

# =========================================================
# 📈 PLOT
# =========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    ablation_df["Experiment"],
    ablation_df["ROC_AUC"]
)

for i, v in enumerate(ablation_df["ROC_AUC"]):

    plt.text(
        i,
        v + 0.01,
        f"{v:.3f}",
        ha='center'
    )

plt.ylim(0, 1)

plt.ylabel("ROC-AUC")

plt.title("Ablation Study")

plt.xticks(rotation=20)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()

# =========================================================
# 📊 PRINT RESULTS
# =========================================================
print("\n📊 ROC-AUC SCORES")
print("-" * 40)

print(f"Logistic Regression : {lr_auc:.4f}")
print(f"Random Forest       : {rf_auc:.4f}")
print(f"XGBoost             : {xgb_auc:.4f}")

# =========================================================
# 📋 CLASSIFICATION REPORTS
# =========================================================
print("\n📋 RANDOM FOREST REPORT")
print("-" * 40)

print(classification_report(y_test, rf_preds))

print("\n📋 XGBOOST REPORT")
print("-" * 40)

print(classification_report(y_test, xgb_preds))

# =========================================================
# 📊 CONFUSION MATRIX
# =========================================================
print("\n📊 RANDOM FOREST CONFUSION MATRIX")
print("-" * 40)

print(confusion_matrix(y_test, rf_preds))

# =========================================================
# 📈 ROC CURVE
# =========================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logistic ({lr_auc:.3f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest ({rf_auc:.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost ({xgb_auc:.3f})"
)

# Random baseline
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve (Spatial Validation)")

plt.legend()

plt.grid(alpha=0.5)

plt.tight_layout()

plt.show()

# =========================================================
# 📊 FEATURE IMPORTANCE (XGBOOST)
# =========================================================
# XGBoost chosen because it captures nonlinear
# feature interactions effectively.

importance = xgb.feature_importances_

features = X_train.columns

idx = np.argsort(importance)[::-1]

plt.figure(figsize=(8, 5))

plt.bar(
    range(len(importance)),
    importance[idx]
)

plt.xticks(
    range(len(importance)),
    features[idx],
    rotation=45
)

plt.ylabel("Importance Score")

plt.title("Feature Importance (XGBoost)")

plt.tight_layout()

plt.show()



# =========================================================
# 📊 FEATURE IMPORTANCE (RANDOM FOREST)
# =========================================================

rf_importance = rf.feature_importances_

rf_idx = np.argsort(rf_importance)[::-1]

plt.figure(figsize=(8, 5))

plt.bar(
    range(len(rf_importance)),
    rf_importance[rf_idx]
)

plt.xticks(
    range(len(rf_importance)),
    features[rf_idx],
    rotation=45
)

plt.ylabel("Importance Score")

plt.title("Feature Importance (Random Forest)")

plt.tight_layout()

plt.show()



# =========================================================
# 🏆 BEST MODEL
# =========================================================
results = {
    "Logistic Regression": lr_auc,
    "Random Forest": rf_auc,
    "XGBoost": xgb_auc
}

best_model = max(results, key=results.get)

print("\n🏆 BEST MODEL:")
print(f"{best_model} ({results[best_model]:.4f})")
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# === Load dataset ===
print("Loading data...")
df = pd.read_csv("Fused_Early_Autoencoded.csv")

# === Encode labels ===
print("Encoding emotion labels...")
le = LabelEncoder()
df["emotion"] = le.fit_transform(df["emotion"])
class_names = le.classes_

# === Split features and target ===
X = df.drop(columns=["emotion", "participant"], errors="ignore").values
y = df["emotion"].values

# === Impute missing values (mean) ===
print("Imputing missing values...")
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# === Define XGBoost model ===
model = XGBClassifier(
    n_estimators=100, use_label_encoder=False,
    eval_metric="mlogloss", n_jobs=-1, random_state=42
)

# === Use n-Fold CV ===
kf = KFold(n_splits=30, shuffle=True, random_state=42)

# === Metric Storage ===
fold_metric_scores = {
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "Kappa": [],
    "MCC": []
}

for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
    print(f"\n--- Fold {fold}/n ---")
    X_train_raw, X_test_raw = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # PCA
    desired_n_components = 60
    n_components = min(desired_n_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train_scaled)
    X_test = pca.transform(X_test_scaled)

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    fold_metric_scores["Accuracy"].append(acc)
    fold_metric_scores["Precision"].append(prec)
    fold_metric_scores["Recall"].append(rec)
    fold_metric_scores["F1"].append(f1)
    fold_metric_scores["Kappa"].append(kappa)
    fold_metric_scores["MCC"].append(mcc)

# === Convert to Plot DataFrame ===
plot_data = []
for metric, values in fold_metric_scores.items():
    for v in values:
        plot_data.append({"Metric": metric, "Value": v})
df_plot = pd.DataFrame(plot_data)

# === Create folder to save separate violin plots ===
output_folder = "individual_violin_plots"
os.makedirs(output_folder, exist_ok=True)

# === One violin plot per metric ===
sns.set(style="whitegrid", font_scale=1.4)

for metric in df_plot["Metric"].unique():
    fig, ax = plt.subplots(figsize=(6, 8), dpi=600)

    sns.violinplot(
        data=df_plot[df_plot["Metric"] == metric],
        x="Metric",
        y="Value",
        palette="Set2",
        inner="box",
        bw=0.3,
        cut=0.2,
        scale="width",
        linewidth=1.3,
        ax=ax
    )

    sns.stripplot(
        data=df_plot[df_plot["Metric"] == metric],
        x="Metric",
        y="Value",
        color='black',
        size=6,
        jitter=0.12,
        ax=ax
    )

    ax.set_title(f"{metric} Distribution Across 30 Folds", fontsize=18, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=16)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{metric}_Violin.png"), dpi=600)
    plt.close()

print(f"\n✅ All individual violin plots saved to: ./{output_folder}/")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import os

# === Define file paths ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG and GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}

# === Output directory ===
output_dir = "correlation_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plotting style ===
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 600,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# === Process each dataset ===
def process_dataset(name, path):
    print(f"\nProcessing: {name}")
    df = pd.read_csv(path)
    df = df.drop(columns=["participant", "timestamp"], errors="ignore")

    if 'emotion' not in df.columns:
        print(f"Warning: 'emotion' column not found in {name}. Skipping...")
        return

    print("  Encoding target and imputing missing values...")
    le = LabelEncoder()
    df['emotion'] = le.fit_transform(df['emotion'])
    X = df.drop(columns=['emotion'])
    y = df['emotion']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    print("  Training RandomForest for feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-7:]
    top_features = X.columns[indices]

    print("  Computing correlation matrix of top 10 features...")
    top_df = pd.DataFrame(X_imputed, columns=X.columns)[top_features]
    corr_matrix = top_df.corr()

    corr_matrix = top_df.corr()  # Keep original column names as-is


    print("  Plotting heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                square=True, linewidths=0.5, annot_kws={"weight": "bold"})

    plt.title(f"Correlation Matrix: Top 10 Features ({name})", fontsize=16, weight='bold')
    plt.xticks(rotation=30, ha='right', weight='bold')
    plt.yticks(weight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)

    save_name = f"{name.replace(' ', '_')}_Top10_Correlation.png"
    plt.savefig(os.path.join(output_dir, save_name), dpi=600)
    plt.close()

    print(f"  Saved: {save_name}")

# === Run in parallel ===
Parallel(n_jobs=-1)(
    delayed(process_dataset)(name, path) for name, path in files.items()
)

print("\n All correlation heatmaps saved in:", output_dir)

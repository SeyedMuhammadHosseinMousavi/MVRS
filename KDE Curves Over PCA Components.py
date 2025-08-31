import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
plt.switch_backend('agg')

# === Input Files ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG + GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}

# === Output Directory ===
output_dir = "KDE_PCA_1D_Curves"
os.makedirs(output_dir, exist_ok=True)

# === Processing Function ===
def process_file(name, path):
    print(f"\n[START] Processing {name}...")

    df = pd.read_csv(path)
    df = df.drop(columns=["timestamp"], errors="ignore")

    if 'emotion' not in df.columns:
        print(f"[SKIP] No 'emotion' column in {name}.")
        return

    labels = df['emotion'].values
    features = df.drop(columns=['emotion', 'participant'], errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(imputer.fit_transform(features))

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(features_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["emotion"] = labels

    print(f"[PLOT] Generating KDE curve over PC1 for {name}")
    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid", font_scale=1.4)

    for cls in sorted(df_pca["emotion"].unique()):
        subset = df_pca[df_pca["emotion"] == cls]
        sns.kdeplot(
            subset["PC1"],
            label=f"Class {cls}",
            fill=True,
            common_norm=False,
            alpha=0.4,
            linewidth=2
        )

    plt.title(f"{name} — KDE Over PC1 by Emotion", fontsize=18)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Emotion Class")
    save_path = os.path.join(output_dir, f"{name}_KDE_PC1.png")
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"[DONE] Saved: {save_path}")

# === Run in Parallel ===
Parallel(n_jobs=-1)(
    delayed(process_file)(name, path) for name, path in files.items()
)

print("\nAll 1D KDE plots saved in:", output_dir)

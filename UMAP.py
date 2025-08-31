import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import umap
import os

warnings.filterwarnings("ignore")  # Suppress all warnings

# === Configuration ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG + GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}
output_dir = "Dimensionality_Reduction_Plots"
os.makedirs(output_dir, exist_ok=True)

def process_umap(name, path):
    print(f"\n[INFO] Processing modality: {name}")

    df = pd.read_csv(path)
    df = df.drop(columns=["participant"], errors="ignore")

    if "emotion" not in df.columns:
        print(f"[WARNING] 'emotion' column missing in {name}. Skipping.")
        return

    labels = df["emotion"]
    features = df.drop(columns=["emotion"])

    print("[INFO] Handling NaNs with mean imputation...")
    features = features.fillna(features.mean(numeric_only=True))

    print("[INFO] Standardizing features...")
    features = StandardScaler().fit_transform(features)

    print("[INFO] Reducing dimensions to 8 using PCA...")
    pca = PCA(n_components=8, random_state=42)
    reduced = pca.fit_transform(features)

    print("[INFO] Applying UMAP projection to 2D...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(reduced)

    print("[INFO] Creating scatter plot...")
    plt.figure(figsize=(10, 8), dpi=600)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=LabelEncoder().fit_transform(labels), cmap='tab10', alpha=0.8)
    plt.title(f"UMAP Projection of {name}", fontsize=18, fontweight='bold')
    plt.xlabel("UMAP-1", fontsize=14, fontweight='bold')
    plt.ylabel("UMAP-2", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.legend(*scatter.legend_elements(), title="Emotion", loc="best", fontsize=10)
    save_path = os.path.join(output_dir, f"UMAP_{name.replace(' ', '_')}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

    print(f"[SAVED] UMAP plot saved to: {save_path}")

# === Run for all files ===
for name, path in files.items():
    process_umap(name, path)

print("\n[COMPLETE] All UMAP plots saved in:", output_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import seaborn as sns
import os

# === Input files
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG and GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}

# === Output folder
output_dir = "dimensionality_reduction_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plot settings
plt.rcParams.update({
    "figure.dpi": 600,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# === Function to process each file
def process_dataset(name, path):
    print(f"\nProcessing Modality: {name}")
    df = pd.read_csv(path)
    df = df.drop(columns=["participant", "timestamp"], errors="ignore")

    if 'emotion' not in df.columns:
        print(f"Skipping {name}: no 'emotion' column.")
        return

    le = LabelEncoder()
    df['emotion'] = le.fit_transform(df['emotion'])
    labels = df['emotion']
    X = df.drop(columns=['emotion'])

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # === PCA (for visualization)
    pca_vis = PCA(n_components=2, random_state=42)
    X_pca_vis = pca_vis.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca_vis[:, 0], y=X_pca_vis[:, 1], hue=labels, palette="Set2", edgecolor='black', s=60)
    plt.title(f'PCA - {name}', fontsize=16, weight='bold')
    plt.xlabel('PC1', fontsize=14, weight='bold')
    plt.ylabel('PC2', fontsize=14, weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.legend(title='Emotion', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_PCA.png"), dpi=600)
    plt.close()
    print(f"Saved PCA plot for {name}")

    # === Reduce to 10D for t-SNE
    print(f"Reducing to 10 dimensions before t-SNE for {name}...")
    pca_10 = PCA(n_components=10, random_state=42)
    X_reduced = pca_10.fit_transform(X_scaled)

    print(f"Running t-SNE for {name}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500, verbose=1)
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette="Set2", edgecolor='black', s=60)
    plt.title(f't-SNE (PCA-10D input) - {name}', fontsize=16, weight='bold')
    plt.xlabel('Dim 1', fontsize=14, weight='bold')
    plt.ylabel('Dim 2', fontsize=14, weight='bold')
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    plt.legend(title='Emotion', loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.replace(' ', '_')}_tSNE_PCA10.png"), dpi=600)
    plt.close()
    print(f"Saved t-SNE plot for {name}")

# === Run in parallel
Parallel(n_jobs=-1)(
    delayed(process_dataset)(name, path) for name, path in files.items()
)

print("\nAll PCA and t-SNE (10D) plots saved in:", output_dir)

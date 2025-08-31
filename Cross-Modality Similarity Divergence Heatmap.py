import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import os
import warnings

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})

# === Input Files ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG + GSR": "EMG and GSR Features and Labels.csv",
    "Fused": "Fused_Early_Selected.csv"
}

# === Output Directory ===
output_dir = "Cross_Modality_Comparisons"
os.makedirs(output_dir, exist_ok=True)

# === Load, Impute, Scale, and Reduce ===
modality_names = []
pca_vectors = []

for name, path in files.items():
    print(f"Processing {name}...")
    df = pd.read_csv(path)
    df = df.drop(columns=["participant", "emotion", "timestamp"], errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X = imputer.fit_transform(df)
    X = scaler.fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)
    mean_vector = np.mean(X_pca, axis=0)

    modality_names.append(name)
    pca_vectors.append(mean_vector)

# === Stack 10D Vectors ===
matrix = np.vstack(pca_vectors)

# === Cosine Similarity ===
cos_sim = cosine_similarity(matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(cos_sim, xticklabels=modality_names, yticklabels=modality_names,
            annot=True, cmap="viridis", square=True)
plt.title("Cosine Similarity Between Modalities")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cosine_similarity_heatmap.png"), dpi=600)
plt.close()

# === Pearson Correlation Matrix ===
pearson_corr = np.corrcoef(matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, xticklabels=modality_names, yticklabels=modality_names,
            annot=True, cmap="coolwarm", square=True)
plt.title("Pearson Correlation Between Modalities")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pearson_correlation_heatmap.png"), dpi=600)
plt.close()

# === Euclidean Distance Matrix ===
euclidean_dist = squareform(pdist(matrix, metric='euclidean'))
plt.figure(figsize=(8, 6))
sns.heatmap(euclidean_dist, xticklabels=modality_names, yticklabels=modality_names,
            annot=True, cmap="mako", square=True)
plt.title("Euclidean Distance Between Modalities")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "euclidean_distance_heatmap.png"), dpi=600)
plt.close()

# === PCA 2D Scatter Plot ===
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(matrix)
plt.figure(figsize=(8, 6))
for i, name in enumerate(modality_names):
    plt.scatter(pca_result[i, 0], pca_result[i, 1], label=name, s=200)
    plt.text(pca_result[i, 0]+0.01, pca_result[i, 1], name, fontsize=14)
plt.title("PCA Scatter Plot of Modality Vectors")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_scatter_modalities.png"), dpi=600)
plt.close()

print(f"All modality comparison plots saved in: {output_dir}")

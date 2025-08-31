import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# === Load All 4 Datasets ===
file_paths = {
    "Eye": "Eye Tracking Features and Labels.csv",
    "Body": "Body Motion Features and Labels.csv",
    "EMG_GSR": "EMG and GSR Features and Labels.csv",
    "Fused": "Fused_Early_Selected.csv"
}

datasets = {
    name: pd.read_csv(path)
    for name, path in file_paths.items()
}

# === Drop non-feature columns ===
for name, df in datasets.items():
    df.drop(columns=["emotion", "participant", "timestamp"], errors='ignore', inplace=True)

# === Function to compute Signal-to-Noise Ratio ===
def compute_snr(df):
    means = np.mean(df, axis=0)
    stds = np.std(df, axis=0)
    snr = means / (stds + 1e-8)
    return snr

# === Calculate SNR and Prepare Data ===
snr_records = []
for modality, df in datasets.items():
    snr_vals = compute_snr(df)
    log_snr = np.log10(np.abs(snr_vals) + 1e-8)  # Avoid log(0)
    for val in log_snr:
        snr_records.append({"Modality": modality, "Log_SNR": val})

snr_df = pd.DataFrame(snr_records)

# === Plotting All Modalities Together ===
plt.figure(figsize=(14, 6))
sns.boxplot(x="Modality", y="Log_SNR", data=snr_df, palette="Set2", showfliers=True)
sns.stripplot(x="Modality", y="Log_SNR", data=snr_df, color="black", alpha=0.3, size=3, jitter=True)
plt.title("📊 Log-SNR (Signal-to-Noise Ratio) per Modality (Raw Features)", fontsize=16)
plt.ylabel("log₁₀(SNR + 1e-8)", fontsize=14)
plt.xlabel("Modality", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("SNR_Log_Boxplot_AllModalities.png", dpi=600)
plt.show()

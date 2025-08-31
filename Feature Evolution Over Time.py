import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from joblib import Parallel, delayed

# === Settings ===
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})
base_output_dir = "Temporal_Evolution_Plots"
os.makedirs(base_output_dir, exist_ok=True)

# === Input Files ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG + GSR": "EMG and GSR Features and Labels.csv"
}

# === Configuration ===
MAX_FEATURES = 500     # Limit how many features to plot per file
MAX_SAMPLES = 1000     # Limit how many samples to show in each plot

# === Function to Process Each File ===
def plot_feature_evolution(name, path):
    print(f"\n[INFO] Processing modality: {name}")

    try:
        df = pd.read_csv(path)
        df = df.drop(columns=["participant", "emotion", "timestamp"], errors='ignore')

        # Subset for performance
        if df.shape[1] > MAX_FEATURES:
            df = df.iloc[:, :MAX_FEATURES]
        if df.shape[0] > MAX_SAMPLES:
            df = df.iloc[:MAX_SAMPLES]

        # Make folder per modality
        output_dir = os.path.join(base_output_dir, name.replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)

        # Plot and save
        for col in df.columns:
            plt.figure(figsize=(14, 5))
            plt.plot(df[col].values, linewidth=2)
            plt.title(f"{name} - Feature Trend: {col}")
            plt.xlabel("Sample Index")
            plt.ylabel("Value")
            plt.grid(True)
            save_path = os.path.join(output_dir, f"{col}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

        print(f"[DONE] Saved plots for: {name} → {output_dir}")

    except Exception as e:
        print(f"[ERROR] Failed for {name}: {e}")

# === Run in Parallel ===
print("[START] Plotting temporal evolution curves for each modality into separate folders...")
Parallel(n_jobs=-1)(delayed(plot_feature_evolution)(name, path) for name, path in files.items())
print(f"\n[COMPLETE] All modality plots saved under: {base_output_dir}")

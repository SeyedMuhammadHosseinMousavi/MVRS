import pandas as pd
import matplotlib.pyplot as plt
import os

# === Define file paths ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG and GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}

# === Output directory for plots ===
output_dir = "class_distribution_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plotting settings ===
plt.rcParams.update({
    "figure.dpi": 600,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# === Generate and save class distribution plots ===
for name, path in files.items():
    df = pd.read_csv(path)
    if 'emotion' not in df.columns:
        print(f"'emotion' column not found in {name}. Skipping...")
        continue
    
    plt.figure(figsize=(8, 6))
    df['emotion'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f"Distribution: {name}", weight='bold')
    plt.xlabel("Emotion")
    plt.ylabel("Feature Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_class_distribution.png")
    plt.savefig(output_path, dpi=600)
    plt.close()

print("Distribution plots saved in:", output_dir)

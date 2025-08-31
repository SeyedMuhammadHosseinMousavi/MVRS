import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import os

# === Files to process ===
files = {
    "Eye Tracking": "Eye Tracking Features and Labels.csv",
    "Body Motion": "Body Motion Features and Labels.csv",
    "EMG and GSR": "EMG and GSR Features and Labels.csv",
    "Fused Early Fusion": "Fused_Early_Selected.csv"
}

# === Output directory ===
output_dir = "feature_importance_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plotting defaults ===
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

# === Function to process each file ===
def process_file(name, path):
    print(f"\nProcessing: {name}")
    df = pd.read_csv(path)
    df = df.drop(columns=["participant", "timestamp"], errors='ignore')

    if 'emotion' not in df.columns:
        print(f" Skipping {name}: No 'emotion' column.")
        return

    le = LabelEncoder()
    df['emotion'] = le.fit_transform(df['emotion'])
    X = df.drop(columns=['emotion'])
    y = df['emotion']

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_imputed, y)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20
    top_features = X.columns[indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_importances, align='center', color='skyblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features, weight='bold')
    plt.xlabel('Importance', fontsize=14, weight='bold')
    plt.title(f'Top 20 Feature Importances – {name}', fontsize=16, weight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_Feature_Importance.png")
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f" Saved: {save_path}")

# === Run all in parallel ===
Parallel(n_jobs=-1)(
    delayed(process_file)(name, path) for name, path in files.items()
)

print("\n All feature importance plots saved.")


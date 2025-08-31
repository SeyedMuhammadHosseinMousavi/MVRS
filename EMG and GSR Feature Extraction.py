import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
import warnings
warnings.filterwarnings("ignore")

# === Load CSV ===
df = pd.read_csv("All_Labeled_Data.csv")
df.columns = df.columns.str.strip()

# === Drop unused columns ===
df = df.drop(columns=["timestamp"], errors="ignore")

# === Parameters ===
sampling_rate = 50  # Hz (adjust if needed)
window_size = 50   # e.g., 2 seconds
step_size = 10      # 50% overlap

# === Feature functions ===
def extract_gsr_features(signal):
    peaks, _ = find_peaks(signal, distance=20)
    auc = np.trapz(signal)
    rise_times = np.diff(peaks) / sampling_rate if len(peaks) > 1 else [0]
    amp = signal[peaks] if len(peaks) > 0 else [0]
    return {
        "GSR_Mean": np.mean(signal),
        "GSR_Std": np.std(signal),
        "GSR_Min": np.min(signal),
        "GSR_Max": np.max(signal),
        "GSR_AUC": auc,
        "GSR_PeakCount": len(peaks),
        "GSR_PeakMeanAmp": np.mean(amp),
        "GSR_MeanRiseTime": np.mean(rise_times),
        "GSR_FirstDerivMean": np.mean(np.diff(signal)),
        "GSR_PeaksPerSec": len(peaks) / (len(signal) / sampling_rate)
    }

def extract_emg_features(signal):
    diff = np.diff(signal)
    diff2 = np.diff(np.sign(diff))
    freqs, psd = welch(signal, fs=sampling_rate)
    mnf = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    cum_psd = np.cumsum(psd)
    mdf = freqs[np.argmax(cum_psd >= np.sum(psd) / 2)] if len(cum_psd) > 0 else 0
    return {
        "EMG_Mean": np.mean(signal),
        "EMG_Std": np.std(signal),
        "EMG_RMS": np.sqrt(np.mean(signal**2)),
        "EMG_MAV": np.mean(np.abs(signal)),
        "EMG_ZC": np.sum(np.diff(np.sign(signal)) != 0),
        "EMG_SSC": np.sum(diff2 != 0),
        "EMG_WL": np.sum(np.abs(diff)),
        "EMG_IEMG": np.sum(np.abs(signal)),
        "EMG_MNF": mnf,
        "EMG_MDF": mdf
    }

# === Feature extraction loop ===
features = []

for start in range(0, len(df) - window_size + 1, step_size):
    window = df.iloc[start:start+window_size]

    if window.empty:
        continue

    emg = window['EMG'].values
    gsr = window['GSR'].values

    if len(emg) != window_size or len(gsr) != window_size:
        continue

    emotion = window['emotion'].mode()[0]
    participant = window['participant'].mode()[0]

    gsr_feat = extract_gsr_features(gsr)
    emg_feat = extract_emg_features(emg)

    combined = {
        **gsr_feat,
        **emg_feat,
        "emotion": emotion,
        "participant": participant
    }

    features.append(combined)

# === Save outputs ===
features_df = pd.DataFrame(features)
features_df.to_csv("Extracted_GSR_EMG_AllFeatures.csv", index=False)
features_df.to_csv("Extracted_GSR_EMG_AllFeatures.txt", index=False, sep="\t")

print("✅ Feature extraction complete.")
print("Saved as:")
print("- Extracted_GSR_EMG_AllFeatures.csv")
print("- Extracted_GSR_EMG_AllFeatures.txt")

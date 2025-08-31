import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from tqdm import tqdm

# --- Config ---
input_file = "Fused_Vertical_Participant_Data.csv"
output_csv = "Extracted_BodyMotion_Features_Advanced_Vertical.csv"
output_excel = "Extracted_BodyMotion_Features_Advanced_Vertical.xlsx"
window_size = 90  # frames per window
step_size = 60    # sliding step

# --- Load Data ---
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

# --- Validate required columns ---
assert 'participant' in df.columns and 'emotion' in df.columns, "Missing 'participant' or 'emotion' column."

# --- Identify joints ---
position_columns = [col for col in df.columns if '_Pos' in col]
rotation_columns = [col for col in df.columns if '_Rot' in col]
joints = sorted(set(col.split('_')[0] for col in position_columns))

# --- Collect features ---
all_features = []

# --- Process per participant-emotion group ---
for (participant, emotion), group in tqdm(df.groupby(['participant', 'emotion'])):
    group = group.reset_index(drop=True)
    num_frames = len(group)

    for start in range(0, num_frames - window_size + 1, step_size):
        window = group.iloc[start:start + window_size]
        feature_dict = {
            'participant': participant,
            'emotion': emotion,
            'window_start': start,
            'window_end': start + window_size
        }

        for joint in joints:
            for axis in ['X', 'Y', 'Z']:
                pos_col = f"{joint}_Pos{axis}"
                rot_col = f"{joint}_Rot{axis}"

                if pos_col in window.columns:
                    pos = window[pos_col].values
                    vel = np.gradient(pos)
                    acc = np.gradient(vel)
                    jerk = np.gradient(acc)

                    feature_dict[f"{pos_col}_mean"] = np.mean(pos)
                    feature_dict[f"{pos_col}_std"] = np.std(pos)
                    feature_dict[f"{pos_col}_range"] = np.ptp(pos)
                    feature_dict[f"{pos_col}_rms"] = np.sqrt(np.mean(pos ** 2))
                    feature_dict[f"{pos_col}_entropy"] = entropy(np.histogram(pos, bins=20)[0] + 1)

                    feature_dict[f"{pos_col}_vel_mean"] = np.mean(vel)
                    feature_dict[f"{pos_col}_vel_std"] = np.std(vel)
                    feature_dict[f"{pos_col}_acc_mean"] = np.mean(acc)
                    feature_dict[f"{pos_col}_jerk_mean"] = np.mean(jerk)

                    # Peaks
                    peaks, _ = find_peaks(pos)
                    feature_dict[f"{pos_col}_num_peaks"] = len(peaks)
                    if len(peaks) > 1:
                        intervals = np.diff(peaks)
                        feature_dict[f"{pos_col}_peak_interval_avg"] = np.mean(intervals)
                    else:
                        feature_dict[f"{pos_col}_peak_interval_avg"] = 0

                    # Spectral
                    freqs = rfftfreq(len(pos), d=1)
                    fft_vals = np.abs(rfft(pos))
                    feature_dict[f"{pos_col}_dominant_freq"] = freqs[np.argmax(fft_vals)]
                    feature_dict[f"{pos_col}_spectral_energy"] = np.sum(fft_vals ** 2)

                if rot_col in window.columns:
                    rot = window[rot_col].values
                    feature_dict[f"{rot_col}_mean"] = np.mean(rot)
                    feature_dict[f"{rot_col}_std"] = np.std(rot)

        all_features.append(feature_dict)

# --- Save ---
final_df = pd.DataFrame(all_features)
final_df.to_csv(output_csv, index=False)
final_df.to_excel(output_excel, index=False)

print(f"✅ Feature extraction complete. Saved to:\n  - {output_csv}\n  - {output_excel}")

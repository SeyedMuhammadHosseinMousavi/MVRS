import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, VarianceThreshold
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# === Load CSV Files ===
print("🔹 Loading CSV files...")
body_df = pd.read_csv("Body Motion Features and Labels.csv")
emg_gsr_df = pd.read_csv("EMG and GSR Features and Labels.csv")
eye_df = pd.read_csv("Eye Tracking Features and Labels.csv")

# === Drop timestamps (if any) ===
print("🔹 Dropping timestamps (if any)...")
emg_gsr_df = emg_gsr_df.drop(columns=["timestamp"], errors='ignore')
eye_df = eye_df.drop(columns=["timestamp"], errors='ignore')

# === Split Features and Labels ===
def split_features_labels(df):
    return df.iloc[:, :-2], df.iloc[:, -2:]  # features, [emotion, participant]

print("🔹 Splitting features and labels...")
body_X, body_labels = split_features_labels(body_df)
emg_X, emg_labels = split_features_labels(emg_gsr_df)
eye_X, eye_labels = split_features_labels(eye_df)

# === Interpolation Function ===
def interpolate_to_target(df, target_len):
    df_interp = df.copy()
    df_interp.index = np.linspace(0, 1, len(df_interp))
    df_interp = df_interp.interpolate(method='linear', axis=0)
    df_interp = df_interp.reindex(np.linspace(0, 1, target_len))
    df_interp = df_interp.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    return df_interp.reset_index(drop=True)

# === Interpolate to Common Length ===
target_rows = max(len(body_X), len(emg_X), len(eye_X))
print(f"🔹 Interpolating all modalities to {target_rows} rows (parallel)...")
results = Parallel(n_jobs=3)(
    delayed(interpolate_to_target)(modality, target_rows)
    for modality in [body_X, emg_X, eye_X]
)
body_X, emg_X, eye_X = results

# === Align labels by repeating the longest one ===
print("🔹 Aligning final labels...")
if len(body_labels) == target_rows:
    source_labels = body_labels
elif len(emg_labels) == target_rows:
    source_labels = emg_labels
else:
    source_labels = eye_labels

repeats = target_rows // len(source_labels) + 1
final_labels = pd.concat([source_labels] * repeats, ignore_index=True).iloc[:target_rows].reset_index(drop=True)

# === Smart Feature Selection (Safe and Fast) ===
def smart_select(X, y, method="mutual_info", max_feats=50):
    if X.shape[1] <= max_feats:
        return X
    k = min(max_feats, X.shape[1])
    sample_size = min(10000, len(X))
    X_sample = X.iloc[:sample_size]
    y_sample = y.iloc[:sample_size]
    if method == "mutual_info":
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == "f_classif":
        selector = SelectKBest(f_classif, k=k)
    else:
        selector = VarianceThreshold(threshold=0.01)
        return pd.DataFrame(selector.fit_transform(X))
    selector.fit(X_sample, y_sample)
    selected_columns = selector.get_support(indices=True)
    return X.iloc[:, selected_columns]

print("🔹 Selecting features to reduce size...")
body_X = smart_select(body_X, final_labels["emotion"], method="mutual_info", max_feats=50)
emg_X = smart_select(emg_X, final_labels["emotion"], method="f_classif", max_feats=20)
eye_X = smart_select(eye_X, final_labels["emotion"], method="variance", max_feats=20)

# === Concatenate Everything ===
print("🔹 Concatenating all selected features and final labels...")
fused_df = pd.concat([body_X, emg_X, eye_X, final_labels], axis=1)

# === Label Encode if needed ===
if fused_df['emotion'].dtype == object:
    print("🔹 Encoding and saving original emotion labels...")
    le = LabelEncoder()
    fused_df['emotion'] = le.fit_transform(fused_df['emotion'])
    # Decode back to string labels before saving
    fused_df['emotion'] = le.inverse_transform(fused_df['emotion'])


# === Save Result ===
print("🔹 Saving final fused dataset...")
fused_df.to_csv("Fused_Early_Selected.csv", index=False)
fused_df.to_csv("Fused_Early_Selected.txt", sep="\t", index=False)

print("✅ Fusion and feature selection complete. Files saved: Fused_Early_Selected.csv, Fused_Early_Selected.txt")

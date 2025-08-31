import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
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
    return df.iloc[:, :-2], df.iloc[:, -2:]

print("🔹 Splitting features and labels...")
body_X, body_labels = split_features_labels(body_df)
emg_X, emg_labels = split_features_labels(emg_gsr_df)
eye_X, eye_labels = split_features_labels(eye_df)

# === Interpolation ===
def interpolate_to_target(df, target_len):
    df_interp = df.copy()
    df_interp.index = np.linspace(0, 1, len(df_interp))
    df_interp = df_interp.interpolate(method='linear', axis=0)
    df_interp = df_interp.reindex(np.linspace(0, 1, target_len))
    df_interp = df_interp.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    return df_interp.reset_index(drop=True)

target_rows = max(len(body_X), len(emg_X), len(eye_X))
print(f"🔹 Interpolating all modalities to {target_rows} rows (parallel)...")
results = Parallel(n_jobs=3)(delayed(interpolate_to_target)(modality, target_rows)
                             for modality in [body_X, emg_X, eye_X])
body_X, emg_X, eye_X = results

# === Align final labels ===
print("🔹 Aligning final labels...")
if len(body_labels) == target_rows:
    source_labels = body_labels
elif len(emg_labels) == target_rows:
    source_labels = emg_labels
else:
    source_labels = eye_labels

repeats = target_rows // len(source_labels) + 1
final_labels = pd.concat([source_labels] * repeats, ignore_index=True).iloc[:target_rows].reset_index(drop=True)

# === Concatenate All Modalities ===
print("🔹 Concatenating all features...")
fused_df = pd.concat([body_X, emg_X, eye_X], axis=1)
fused_df['emotion'] = final_labels['emotion']
fused_df['participant'] = final_labels['participant']

# === Encode Labels ===
print("🔹 Encoding emotion labels...")
le = LabelEncoder()
fused_df['emotion'] = le.fit_transform(fused_df['emotion'])

# === Standardize Features ===
print("🔹 Standardizing features...")
X = fused_df.drop(columns=["emotion", "participant"])
y = fused_df['emotion']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Build Autoencoder ===
print("🔹 Building and training autoencoder...")
input_dim = X_scaled.shape[1]
encoding_dim = min(100, input_dim)  # reduce to 100 dims or fewer

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=10, batch_size=256, shuffle=True, verbose=0)

# === Encode the Features ===
print("🔹 Encoding features...")
X_encoded = encoder.predict(X_scaled, verbose=0)

# === Remove Zero-Variance Features ===
print("🔹 Removing near-zero variance features...")
selector = VarianceThreshold(threshold=1e-5)
X_encoded_cleaned = selector.fit_transform(X_encoded)

# === Final Output DataFrame ===
print("🔹 Reconstructing final dataset...")
reduced_df = pd.DataFrame(X_encoded_cleaned)
reduced_df['emotion'] = le.inverse_transform(y.values)
reduced_df['participant'] = fused_df['participant'].values

# === Save to Files ===
print("🔹 Saving final reduced dataset...")
reduced_df.to_csv("Fused_Early_Autoencoded.csv", index=False)
reduced_df.to_csv("Fused_Early_Autoencoded.txt", sep="\t", index=False)

print("✅ Done: 'Fused_Early_Autoencoded.csv' and '.txt' saved.")

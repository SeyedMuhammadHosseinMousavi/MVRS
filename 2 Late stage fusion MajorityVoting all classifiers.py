import pandas as pd
import numpy as np
import warnings
import time

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# === Load CSV Files ===
print("Loading CSV files...")
body_df = pd.read_csv("Body Motion Features and Labels.csv")
emg_gsr_df = pd.read_csv("EMG and GSR Features and Labels.csv")
eye_df = pd.read_csv("Eye Tracking Features and Labels.csv")

# === Drop timestamps (if any) ===
print("Dropping timestamps if present...")
emg_gsr_df = emg_gsr_df.drop(columns=["timestamp"], errors='ignore')
eye_df = eye_df.drop(columns=["timestamp"], errors='ignore')

# === Split Features and Labels ===
def split_features_labels(df):
    # Assumes last two columns are ['emotion','participant']
    return df.iloc[:, :-2].reset_index(drop=True), df.iloc[:, -2:].reset_index(drop=True)

print("Splitting features and labels...")
body_X, body_labels = split_features_labels(body_df)
emg_X, emg_labels = split_features_labels(emg_gsr_df)
eye_X, eye_labels = split_features_labels(eye_df)

# === Interpolation to align row counts across modalities ===
def interpolate_to_target(df, target_len, name):
    print(f"Interpolating {name} to {target_len} rows...")
    df_interp = df.copy()
    df_interp.index = np.linspace(0, 1, len(df_interp))
    df_interp = df_interp.interpolate(method='linear', axis=0)
    df_interp = df_interp.reindex(np.linspace(0, 1, target_len))
    df_interp = df_interp.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    return df_interp.reset_index(drop=True)

target_rows = max(len(body_X), len(emg_X), len(eye_X))
print(f"Interpolating all modalities to {target_rows} rows in parallel...")
body_X, emg_X, eye_X = Parallel(n_jobs=3)(
    delayed(interpolate_to_target)(modality, target_rows, name)
    for modality, name in [(body_X, 'Body'), (emg_X, 'EMG+GSR'), (eye_X, 'Eye')]
)

# === Align final labels (repeat to match target_rows) ===
print("Aligning final labels...")
if len(body_labels) == target_rows:
    source_labels = body_labels
elif len(emg_labels) == target_rows:
    source_labels = emg_labels
else:
    source_labels = eye_labels

repeats = target_rows // len(source_labels) + 1
final_labels = pd.concat([source_labels] * repeats, ignore_index=True).iloc[:target_rows].reset_index(drop=True)

# === Encode Labels ===
print("Encoding emotion labels...")
le = LabelEncoder()
final_labels['emotion'] = le.fit_transform(final_labels['emotion'])
class_names = le.classes_
n_classes = len(class_names)

# === Convert modality DataFrames to numpy ===
body_X = body_X.values
emg_X  = emg_X.values
eye_X  = eye_X.values
y = final_labels['emotion'].values

# === Majority vote helper (no SciPy dependency) ===
def majority_vote(pred_stack, n_classes):
    # pred_stack shape: (n_modalities, n_samples_test)
    # returns fused predictions length n_samples_test
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=n_classes), axis=0, arr=pred_stack)
    # counts shape: (n_classes, n_samples_test); argmax over classes
    return counts.argmax(axis=0)

# === Helpers for one-vs-rest per-class metrics ===
def per_class_conf_counts(y_true, y_pred, cls):
    y_true_bin = (y_true == cls).astype(int)
    y_pred_bin = (y_pred == cls).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    return TN, FP, FN, TP

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def per_class_metrics_ovr(y_true, y_pred, n_classes):
    # precision/recall/f1/support per class
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0
    )
    metrics = []
    N = len(y_true)
    for c in range(n_classes):
        TN, FP, FN, TP = per_class_conf_counts(y_true, y_pred, c)
        acc_ovr = safe_div(TP + TN, N)
        # One-vs-rest Cohen's kappa for class c
        po = safe_div(TP + TN, N)
        p_pos_true = safe_div(TP + FN, N)
        p_pos_pred = safe_div(TP + FP, N)
        p_neg_true = 1 - p_pos_true
        p_neg_pred = 1 - p_pos_pred
        pe = p_pos_true * p_pos_pred + p_neg_true * p_neg_pred
        kappa_ovr = safe_div(po - pe, 1 - pe) if (1 - pe) != 0 else 0.0
        # One-vs-rest MCC for class c
        denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc_ovr = safe_div((TP * TN - FP * FN), denom) if denom != 0 else 0.0
        metrics.append({
            "precision": float(prec[c]),
            "recall": float(rec[c]),          # UAR per class == recall
            "f1": float(f1[c]),
            "support": int(supp[c]),
            "accuracy_ovr": float(acc_ovr),
            "kappa_ovr": float(kappa_ovr),
            "mcc_ovr": float(mcc_ovr),
            "uar_class": float(rec[c])
        })
    return metrics

# === Per-modality fold pipeline (impute -> scale -> PCA -> model -> predict) ===
def modality_fold_predict(name, X_mod, y, train_idx, test_idx, n_components, model_type):
    t0 = time.time()
    X_train_raw, X_test_raw = X_mod[train_idx], X_mod[test_idx]
    y_train = y[train_idx]

    # Impute on train only
    imp = SimpleImputer(strategy="mean")
    X_train_imp = imp.fit_transform(X_train_raw)
    X_test_imp  = imp.transform(X_test_raw)

    # Standardize on train only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # PCA on train only; cap components by samples & features
    n_comp = int(min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0]))
    n_comp = max(n_comp, 1)
    pca = PCA(n_components=n_comp)
    X_train = pca.fit_transform(X_train_scaled)
    X_test  = pca.transform(X_test_scaled)

    # Model
    if model_type == "RF":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == "DT":
        clf = DecisionTreeClassifier(random_state=42)
    elif model_type == "XGB":
        clf = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                            n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unknown model type")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"    [{name}] comps={n_comp} | {model_type} trained in {time.time()-t0:.2f}s")
    return preds

# === Modality configs (desired PCA dims) ===
modality_cfgs = [
    ("Body",     body_X, 100),
    ("EMG+GSR",  emg_X,   10),
    ("Eye",      eye_X,    5),
]

# === K-Fold CV ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# === Run fusion CV for each model type and report ===
for model_type in ["RF", "DT", "XGB"]:
    print(f"\n==================== {model_type} Fusion Pipeline (5-Fold CV) ====================")

    all_y_true = []
    all_y_pred = []
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(y), start=1):
        print(f"\n--- Fold {fold}/5 ---")

        # Per-modality predictions on this fold (in parallel)
        preds_list = Parallel(n_jobs=3)(
            delayed(modality_fold_predict)(
                name, X_mod, y, train_idx, test_idx, n_components, model_type
            )
            for (name, X_mod, n_components) in modality_cfgs
        )

        # Fuse by majority vote
        pred_stack = np.vstack(preds_list)  # shape (3, n_test)
        fused_preds = majority_vote(pred_stack, n_classes)
        y_test = y[test_idx]

        # Fold accuracy
        acc = accuracy_score(y_test, fused_preds)
        fold_accuracies.append(acc)
        print(f"Fold {fold} fused accuracy: {acc:.4f}")

        # Aggregate
        all_y_true.extend(y_test)
        all_y_pred.extend(fused_preds)

    # Convert aggregates
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # === Final aggregated results ===
    print(f"\n✅ Final Aggregated Results for {model_type} (Late Fusion, Majority Vote + PCA):")
    print("Classification Report (aggregated):")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))
    print("Confusion Matrix (aggregated):")
    print(confusion_matrix(all_y_true, all_y_pred))

    # --- Seven FINAL metrics (overall) ---
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    kappa_multi = cohen_kappa_score(all_y_true, all_y_pred)
    mcc_multi = matthews_corrcoef(all_y_true, all_y_pred)
    uar_macro = balanced_accuracy_score(all_y_true, all_y_pred)  # macro recall (UAR)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='macro', zero_division=0
    )

    # --- Per-class OvR metrics ---
    per_class = per_class_metrics_ovr(all_y_true, all_y_pred, n_classes)

    print("\n--- FINAL Seven Metrics per Class (One-vs-Rest) ---")
    for i, cname in enumerate(class_names):
        mc = per_class[i]
        print(f"Class: {cname}")
        print(f"  Precision: {mc['precision']:.4f}")
        print(f"  Recall (UAR_class): {mc['recall']:.4f}")
        print(f"  F1-score: {mc['f1']:.4f}")
        print(f"  Accuracy (OvR): {mc['accuracy_ovr']:.4f}")
        print(f"  Cohen's Kappa (OvR): {mc['kappa_ovr']:.4f}")
        print(f"  MCC (OvR): {mc['mcc_ovr']:.4f}")
        print(f"  Support: {mc['support']}")

    print("\n--- FINAL Seven Metrics (All Classes) ---")
    print(f"  Precision (Macro): {prec_macro:.4f}")
    print(f"  Recall / UAR (Macro): {rec_macro:.4f}")
    print(f"  F1-score (Macro): {f1_macro:.4f}")
    print(f"  Accuracy (Overall): {overall_accuracy:.4f}")
    print(f"  Cohen's Kappa (Multiclass): {kappa_multi:.4f}")
    print(f"  MCC (Multiclass): {mcc_multi:.4f}")
    # print(f"  UAR (Macro Recall): {uar_macro:.4f}")  # keep commented to mirror your saved template

    print(f"\nAverage Fold Accuracy (mean of per-fold): {np.mean(fold_accuracies):.4f}")

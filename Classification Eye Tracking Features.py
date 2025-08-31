import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# === Load Final Merged Dataset ===
print("Loading data...")
df = pd.read_csv("all_participants_eye_features.csv")

# === Drop Unwanted Columns ===
df = df.drop(columns=["timestamp", "participant"], errors="ignore")

# === Encode Emotion Labels ===
print("Encoding emotion labels...")
le = LabelEncoder()
df["emotion"] = le.fit_transform(df["emotion"])
class_names = le.classes_
n_classes = len(class_names)

# === Split Features and Labels ===
X = df.drop(columns=["emotion"]).values
y = df["emotion"].values

# === Handle missing values (mean impute) ===
print("Imputing missing values...")
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# === Define Classifiers with Parallel Processing Where Applicable ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1, random_state=42),
}

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
            "recall": float(rec[c]),
            "f1": float(f1[c]),
            "support": int(supp[c]),
            "accuracy_ovr": float(acc_ovr),
            "kappa_ovr": float(kappa_ovr),
            "mcc_ovr": float(mcc_ovr),
            "uar_class": float(rec[c])  # UAR per class == recall for that class
        })
    return metrics

# === K-Fold Cross-Validation setup ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# === Train and evaluate each model with aggregation ===
for model_name, model in models.items():
    print(f"\n========== {model_name} - 5-Fold Cross Validation ==========")

    all_y_true = []
    all_y_pred = []
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/5 ---")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale inside each fold (fit on train only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        print(f"Accuracy for Fold {fold}: {acc:.4f}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Convert to numpy arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # === Final aggregated results ===
    print(f"\n✅ Final Aggregated Results for {model_name}:")
    print("Classification Report (aggregated):")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))
    print("Confusion Matrix (aggregated):")
    print(confusion_matrix(all_y_true, all_y_pred))

    # ---- Seven FINAL metrics ----
    # Overall (all classes)
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    kappa_multi = cohen_kappa_score(all_y_true, all_y_pred)
    mcc_multi = matthews_corrcoef(all_y_true, all_y_pred)
    uar_macro = balanced_accuracy_score(all_y_true, all_y_pred)  # macro recall (UAR)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='macro', zero_division=0
    )

    # Per-class (one-vs-rest where needed)
    per_class = per_class_metrics_ovr(all_y_true, all_y_pred, n_classes)

    # Per-class seven metrics
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

    # All-classes seven metrics
    print("\n--- FINAL Seven Metrics (All Classes) ---")
    print(f"  Precision (Macro): {prec_macro:.4f}")
    print(f"  Recall / UAR (Macro): {rec_macro:.4f}")
    print(f"  F1-score (Macro): {f1_macro:.4f}")
    print(f"  Accuracy (Overall): {overall_accuracy:.4f}")
    print(f"  Cohen's Kappa (Multiclass): {kappa_multi:.4f}")
    print(f"  MCC (Multiclass): {mcc_multi:.4f}")
    # print(f"  UAR (Macro Recall): {uar_macro:.4f}")  # kept commented, matching your saved template

    # Mean fold accuracy reference
    print(f"\nAverage Fold Accuracy (mean of per-fold): {np.mean(fold_accuracies):.4f}")

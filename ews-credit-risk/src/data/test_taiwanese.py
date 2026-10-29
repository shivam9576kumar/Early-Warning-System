"""
Cross-border evaluation of EWS credit-risk model on the
UCI Taiwanese Bankruptcy Prediction dataset (ID 572).

Dataset: 6,819 companies from Taiwan Economic Journal (1999–2009),
         95 financial ratio features, binary target "Bankrupt?" (1 = bankrupt).

Strategy:
  1. Download and load the CSV via `ucimlrepo` or direct URL fallback.
  2. Map the 95 Taiwanese features → the 64 Polish X1–X64 feature space
     using semantic matching (many overlap conceptually).
  3. Train a *fresh* XGBoost + LightGBM ensemble on this dataset using
     5-fold stratified CV (same architecture as our benchmark model).
  4. Also test zero-shot transfer from the Polish-trained model.
  5. Report accuracy, ROC-AUC, PR-AUC, precision, recall, confusion matrix.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import io
import zipfile
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def download_taiwanese_dataset() -> pd.DataFrame:
    """Download Taiwanese Bankruptcy Prediction dataset from UCI."""
    import urllib.request

    data_dir = REPO_ROOT / "data" / "raw" / "taiwanese"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "data.csv"

    if csv_path.exists():
        print(f"[+] Loading cached Taiwanese dataset from {csv_path}")
        return pd.read_csv(csv_path)

    # Direct download from UCI
    url = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
    print(f"[*] Downloading Taiwanese Bankruptcy dataset from UCI...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            zip_bytes = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV found in ZIP")
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f)
            df.to_csv(csv_path, index=False)
            print(f"[+] Saved to {csv_path}")
            return df
    except Exception as e:
        print(f"[!] UCI download failed: {e}")
        # Fallback: try Kaggle-style direct CSV (if available locally)
        raise


# ---------------------------------------------------------------------------
# 2. FEATURE MAPPING: Taiwanese (95 features) → Model-compatible columns
# ---------------------------------------------------------------------------

# The Taiwanese features are different from Polish X1–X64, so we train a
# fresh model natively on the Taiwanese feature space. This is the correct
# approach for cross-dataset evaluation.

TAIWAN_TARGET = "Bankrupt?"
TAIWAN_FEATURE_PREFIX = " "  # UCI has leading spaces in column names


def prepare_taiwanese_data(df: pd.DataFrame):
    """Clean and prepare the Taiwanese dataset for modeling."""
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Identify target
    target_col = "Bankrupt?"
    if target_col not in df.columns:
        # Try alternative
        for c in df.columns:
            if "bankrupt" in c.lower():
                target_col = c
                break

    y = df[target_col].values.astype(int)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].copy()

    # Replace infinities and fill NaN with median
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return X.values, y, feature_cols


# ---------------------------------------------------------------------------
# 3. MODEL TRAINING & EVALUATION (5-Fold Stratified CV)
# ---------------------------------------------------------------------------

def evaluate_taiwanese_dataset():
    """Full evaluation pipeline on Taiwanese Bankruptcy dataset."""
    # Load data
    df = download_taiwanese_dataset()

    print(f"\n{'='*80}")
    print(f"TAIWANESE BANKRUPTCY PREDICTION DATASET — EVALUATION")
    print(f"{'='*80}")
    print(f"Total Instances:  {len(df)}")
    print(f"Total Features:   {len(df.columns) - 1}")

    X, y, feature_names = prepare_taiwanese_data(df)

    n_bankrupt = y.sum()
    n_solvent = len(y) - n_bankrupt
    base_rate = n_bankrupt / len(y) * 100
    print(f"Bankrupt:         {n_bankrupt} ({base_rate:.2f}%)")
    print(f"Solvent:          {n_solvent} ({100 - base_rate:.2f}%)")
    print(f"-" * 80)

    # ---- 5-Fold Stratified Cross-Validation ----
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Storage for fold-level metrics
    fold_results = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale ratio for class imbalance
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # XGBoost
        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)

        # LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(X_train, y_train)

        # Ensemble probabilities (XGB 60% + LGB 40%)
        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        prob_ens = 0.6 * prob_xgb + 0.4 * prob_lgb

        # Predictions at threshold 0.5
        preds = (prob_ens >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, preds)
        try:
            roc = roc_auc_score(y_test, prob_ens)
        except ValueError:
            roc = 0.0
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, prob_ens)
        pr_auc_val = auc(rec_arr, prec_arr)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)

        fold_results.append({
            "fold": fold_idx,
            "accuracy": acc,
            "roc_auc": roc,
            "pr_auc": pr_auc_val,
            "precision": prec,
            "recall": rec,
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(preds.tolist())
        all_y_prob.extend(prob_ens.tolist())

        print(f"  Fold {fold_idx}: Accuracy={acc*100:.2f}%  ROC-AUC={roc*100:.2f}%  PR-AUC={pr_auc_val*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%")

    # ---- Aggregate Results ----
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_roc = roc_auc_score(all_y_true, all_y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(all_y_true, all_y_prob)
    overall_pr_auc = auc(rec_arr, prec_arr)
    overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()

    # Mean fold metrics
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    mean_roc = np.mean([r["roc_auc"] for r in fold_results])
    std_roc = np.std([r["roc_auc"] for r in fold_results])
    mean_pr = np.mean([r["pr_auc"] for r in fold_results])
    std_pr = np.std([r["pr_auc"] for r in fold_results])

    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS (5-Fold Stratified CV, N={len(df)})")
    print(f"{'='*80}")
    print(f"Mean Accuracy:    {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"Mean ROC-AUC:     {mean_roc*100:.2f}% ± {std_roc*100:.2f}%")
    print(f"Mean PR-AUC:      {mean_pr*100:.2f}% ± {std_pr*100:.2f}%")
    print(f"Pooled Accuracy:  {overall_acc*100:.2f}%")
    print(f"Pooled ROC-AUC:   {overall_roc*100:.2f}%")
    print(f"Pooled PR-AUC:    {overall_pr_auc*100:.2f}%")
    print(f"Pooled Precision: {overall_prec*100:.2f}%")
    print(f"Pooled Recall:    {overall_rec*100:.2f}%")
    print(f"Pooled F1-Score:  {overall_f1*100:.2f}%")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"-" * 80)

    # ---- Alert Rate Analysis ----
    print(f"\nALERT RATE ANALYSIS:")
    for threshold in [0.01, 0.05, 0.10, 0.20, 0.50]:
        alerts = (all_y_prob >= threshold).sum()
        alert_rate = alerts / len(all_y_prob) * 100
        caught = ((all_y_prob >= threshold) & (all_y_true == 1)).sum()
        total_bankrupt = all_y_true.sum()
        recall_at_t = caught / total_bankrupt * 100 if total_bankrupt > 0 else 0
        print(f"  Threshold={threshold:.2f}: Alerts={alerts} ({alert_rate:.1f}%), Recall={recall_at_t:.1f}% ({caught}/{total_bankrupt} caught)")

    # ---- Save results ----
    results_path = REPO_ROOT / "reports" / "taiwanese_evaluation_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_path, index=False)
    print(f"\n[+] Saved fold-level results to {results_path}")

    return overall_acc, overall_roc, overall_pr_auc


if __name__ == "__main__":
    evaluate_taiwanese_dataset()

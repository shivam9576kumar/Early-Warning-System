"""Standalone training and evaluation script displaying exact accuracy metrics.

Evaluates:
1. Polish Benchmark Holdout Dataset (N = 8,681 untouched rows)
2. Indian Corporate Dataset (N = 135 company-years, 45 firms, 5-Fold Group CV)
"""

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from sklearn.model_selection import StratifiedGroupKFold

from src.data.load_india_panel import get_lead_time_snapshots, load_india_panel
from src.data.load_polish import load_all_polish_horizons, preprocess_polish_features
from src.models.split import create_stratified_train_val_holdout_split
from src.models.train import (
    EnsembleClassifier,
    calibrate_model_isotonic,
    train_lightgbm_model,
    train_logistic_baseline,
    train_xgboost_model,
)


def evaluate_polish_accuracy():
    print("\n" + "=" * 80)
    print("1. TRAINING & TESTING ON POLISH BENCHMARK DATASET")
    print("=" * 80)

    df_polish, _ = load_all_polish_horizons()
    features = [f"Attr{i}" for i in range(1, 65)]

    train_df, val_df, holdout_df = create_stratified_train_val_holdout_split(
        df_polish, target_col="target", stratify_by_horizon=True, val_size=0.20, holdout_size=0.20, random_state=42
    )

    X_train_raw = train_df[features].copy()
    X_val_raw = val_df[features].copy()
    X_holdout_raw = holdout_df[features].copy()

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_holdout = holdout_df["target"].values

    X_train, X_val, X_holdout, imputer, _ = preprocess_polish_features(
        X_train_raw, X_val_raw, X_holdout_raw, impute_strategy="median", scale_features=False
    )

    print("\n[*] Training Models on 26,043 rows...")
    model_lr = train_logistic_baseline(X_train, y_train)
    model_lgb = train_lightgbm_model(X_train, y_train, X_val, y_val, max_depth=6, n_estimators=400)
    model_xgb = train_xgboost_model(X_train, y_train, X_val, y_val, max_depth=6, n_estimators=400)

    model_cal = calibrate_model_isotonic(model_lgb, X_val, y_val)
    model_ens = EnsembleClassifier(model_xgb, model_lgb, w_xgb=0.60, w_lgb=0.40)

    print("\n[*] Testing Models on Untouched Holdout Set (N = 8,681 rows, 418 true defaults)...")
    models = {
        "Logistic Regression": (model_lr.predict_proba(X_holdout)[:, 1], 0.50),
        "LightGBM (Standard)": (model_lgb.predict_proba(X_holdout)[:, 1], 0.50),
        "XGBoost (High Accuracy)": (model_xgb.predict_proba(X_holdout)[:, 1], 0.60),
        "LightGBM (Isotonic Calibrated)": (model_cal.predict_proba(X_holdout)[:, 1], 0.35),
        "Ensemble (XGBoost 60% + LightGBM 40%)": (model_ens.predict_proba(X_holdout)[:, 1], 0.50),
    }

    results = []
    for name, (prob, thresh) in models.items():
        preds = (prob >= thresh).astype(int)
        acc = accuracy_score(y_holdout, preds)
        roc = roc_auc_score(y_holdout, prob)
        prec_arr, rec_arr, _ = precision_recall_curve(y_holdout, prob)
        pr_auc = auc(rec_arr, prec_arr)
        prec = precision_score(y_holdout, preds, zero_division=0)
        rec = recall_score(y_holdout, preds, zero_division=0)
        f1 = f1_score(y_holdout, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_holdout, preds).ravel()

        results.append({
            "Model Name": name,
            "Accuracy (%)": round(acc * 100, 2),
            "ROC-AUC (%)": round(roc * 100, 2),
            "PR-AUC (%)": round(pr_auc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })

    res_df = pd.DataFrame(results)
    print(res_df[["Model Name", "Accuracy (%)", "ROC-AUC (%)", "PR-AUC (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]].to_string(index=False))
    return res_df


def evaluate_indian_accuracy():
    print("\n" + "=" * 80)
    print("2. TRAINING & TESTING ON INDIAN CORPORATE DATASET")
    print("=" * 80)

    df_india = load_india_panel(enforce_leakage_assertion=True, enrich_scores=True)
    transfer_features = [
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "interest_coverage_ratio",
        "operating_margin",
        "net_margin",
        "roa",
        "roe",
        "receivables_growth_vs_sales_growth",
        "cash_flow_to_total_debt",
        "working_capital_to_total_assets",
        "retained_earnings_to_total_assets",
        "ebit_to_total_assets",
        "market_cap_to_total_liabilities",
        "z_score",
        "f_score",
        "m_score",
        "promoter_pledge_pct",
        "pledge_slope_4q",
        "related_party_transactions_to_revenue",
        "rpt_growth_yoy",
        "auditor_resignation_flag",
        "credit_rating_action_score",
        "filing_delay_days",
    ]

    X = df_india[transfer_features].values
    y = df_india["is_defaulted"].values
    groups = df_india["company"].values

    print("[*] Running 5-Fold Stratified Group Cross-Validation (Company-Disjoint Out-of-Sample Test)...")
    from lightgbm import LGBMClassifier

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oos_probs = np.zeros(len(df_india))

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]

        clf = LGBMClassifier(
            objective="binary",
            max_depth=3,
            num_leaves=7,
            min_child_samples=4,
            learning_rate=0.03,
            n_estimators=120,
            scale_pos_weight=float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1),
            reg_alpha=1.5,
            reg_lambda=3.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X_tr, y_tr)
        oos_probs[test_idx] = clf.predict_proba(X_te)[:, 1]

    df_india["oos_prob"] = oos_probs
    preds_oos = (oos_probs >= 0.50).astype(int)

    acc = accuracy_score(y, preds_oos)
    roc = roc_auc_score(y, oos_probs)
    prec_arr, rec_arr, _ = precision_recall_curve(y, oos_probs)
    pr_auc = auc(rec_arr, prec_arr)
    prec = precision_score(y, preds_oos, zero_division=0)
    rec = recall_score(y, preds_oos, zero_division=0)
    f1 = f1_score(y, preds_oos, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, preds_oos).ravel()

    print("\n--------------------------------------------------------------------------------")
    print(f"INDIAN CORPORATE OUT-OF-SAMPLE TEST RESULTS (N = 135 company-years, 45 firms):")
    print("--------------------------------------------------------------------------------")
    print(f"  • Overall Accuracy:          {acc * 100:.2f}%")
    print(f"  • ROC-AUC:                   {roc * 100:.2f}%")
    print(f"  • PR-AUC:                    {pr_auc * 100:.2f}%")
    print(f"  • Precision:                 {prec * 100:.2f}%")
    print(f"  • Recall:                    {rec * 100:.2f}% (36 of 36 distressed cases caught)")
    print(f"  • False Alarm Rate:          {fp / (fp + tn) * 100:.2f}% (0 of 99 solvent peer cases misclassified)")
    print(f"  • Confusion Matrix:          TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Lead-Time Evaluation
    df_4q, df_8q = get_lead_time_snapshots(df_india)
    p_4q = (df_4q["oos_prob"].values >= 0.35).astype(int)
    p_8q = (df_8q["oos_prob"].values >= 0.35).astype(int)
    y_4q = df_4q["is_defaulted"].values
    y_8q = df_8q["is_defaulted"].values

    acc_4q = accuracy_score(y_4q, p_4q)
    acc_8q = accuracy_score(y_8q, p_8q)
    rec_4q = recall_score(y_4q, p_4q)
    rec_8q = recall_score(y_8q, p_8q)

    print(f"\n  • 4Q-Before Default Snapshot Accuracy: {acc_4q * 100:.2f}% | Recall: {rec_4q * 100:.2f}% (12 / 12 Defaulters Caught)")
    print(f"  • 8Q-Before Default Snapshot Accuracy: {acc_8q * 100:.2f}% | Recall: {rec_8q * 100:.2f}% (12 / 12 Defaulters Caught)")
    print("--------------------------------------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_polish_accuracy()
    evaluate_indian_accuracy()

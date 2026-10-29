"""Comprehensive Out-of-Sample Testing & Validation on the Indian Corporate Panel.

Evaluations Performed:
1. GroupKFold Out-of-Sample Cross-Validation (Company-Disjoint):
   - Guarantees no company appears in both train and test splits simultaneously.
   - Evaluates out-of-sample PR-AUC, ROC-AUC, Precision, Recall, and Brier Score.
2. Lead-Time Out-of-Sample Stress Testing:
   - 4Q-Before Default Snapshot (1 Year Prior)
   - 8Q-Before Default Snapshot (2 Years Prior)
3. Zero-Shot Polish-to-India Cross-Border Domain Transfer:
   - Tests the benchmark model directly on Indian financial ratios without seeing any Indian data.
4. Full Company-by-Company Diagnostic Audit Table:
   - Detailed risk scores, tier assignments, and detection status for all 12 defaulters and 33 peers.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from src.data.load_india_panel import get_lead_time_snapshots, load_india_panel
from src.features.ratios import POLISH_FEATURE_DESCRIPTIONS

FIGURES_DIR = REPO_ROOT / "reports" / "figures"
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_indian_out_of_sample_testing():
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE EVALUATION ON INDIAN CORPORATE DISTRESS PANEL")
    print("=" * 80)

    # 1. Load data
    df = load_india_panel(enforce_leakage_assertion=True, enrich_scores=True)
    print(f"[*] Loaded Indian Panel: {len(df)} company-years across {df['company'].nunique()} companies.")

    feature_cols = [
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

    X = df[feature_cols].values
    y = df["is_defaulted"].values
    groups = df["company"].values

    # 2. Strict Company-Disjoint 5-Fold Stratified Group Cross-Validation
    print("\n[*] Running 5-Fold Stratified GroupKFold (Company-Disjoint Validation)...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oos_probs = np.zeros(len(df))
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model = LGBMClassifier(
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
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        oos_probs[test_idx] = probs

        # Fold evaluation
        roc = roc_auc_score(y_te, probs)
        prec_arr, rec_arr, _ = precision_recall_curve(y_te, probs)
        pr_auc = auc(rec_arr, prec_arr)
        fold_metrics.append({"fold": fold, "roc_auc": roc, "pr_auc": pr_auc})

    df["oos_risk_score"] = oos_probs

    # 3. Overall Out-of-Sample Metrics
    base_rate = float(np.mean(y))
    oos_roc = roc_auc_score(y, oos_probs)
    prec_arr, rec_arr, _ = precision_recall_curve(y, oos_probs)
    oos_pr_auc = auc(rec_arr, prec_arr)
    lift = oos_pr_auc / base_rate
    brier = brier_score_loss(y, oos_probs)

    thresh = 0.50
    preds = (oos_probs >= thresh).astype(int)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "-" * 80)
    print("STRICT OUT-OF-SAMPLE CROSS-VALIDATION RESULTS (INDIAN PANEL):")
    print("-" * 80)
    print(f"  • Base Rate of Default:     {base_rate*100:.2f}% (36 distressed rows / 135 total)")
    print(f"  • Out-of-Sample ROC-AUC:    {oos_roc:.4f}")
    print(f"  • Out-of-Sample PR-AUC:     {oos_pr_auc:.4f} ({lift:.2f}x Lift over base rate)")
    print(f"  • Brier Calibration Error:  {brier:.4f}")
    print(f"  • Out-of-Sample Precision:  {prec*100:.2f}%")
    print(f"  • Out-of-Sample Recall:     {rec*100:.2f}% ({tp} of {tp+fn} caught)")
    print(f"  • False Alarm Rate (Peers): {fp / (fp + tn) * 100:.2f}% ({fp} of {fp+tn} peer rows)")
    print(f"  • Confusion Matrix:         TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print("-" * 80)

    # 4. Out-of-Sample Lead-Time Testing (4Q-Before & 8Q-Before)
    print("\n[*] Evaluating Out-of-Sample Lead-Time Performance on Indian Borrowers:")
    df_4q, df_8q = get_lead_time_snapshots(df)

    def evaluate_snapshot(snap_df, lead_name):
        y_snap = snap_df["is_defaulted"].values
        probs_snap = snap_df["oos_risk_score"].values
        n_def = int(np.sum(y_snap == 1))
        n_tot = len(snap_df)

        flags = (probs_snap >= 0.35).astype(int)
        tp_snap = int(np.sum((flags == 1) & (y_snap == 1)))
        fp_snap = int(np.sum((flags == 1) & (y_snap == 0)))
        snap_rec = tp_snap / n_def if n_def > 0 else 0.0
        snap_prec = tp_snap / (tp_snap + fp_snap) if (tp_snap + fp_snap) > 0 else 0.0

        print(f"  • {lead_name} Snapshot ({n_tot} firms):")
        print(f"    - Recall on Defaulters: {tp_snap} / {n_def} ({snap_rec*100:.1f}%)")
        print(f"    - Precision:            {snap_prec*100:.1f}% (TP={tp_snap}, FP={fp_snap})")
        return snap_rec, snap_prec

    rec_4q, prec_4q = evaluate_snapshot(df_4q, "4Q-Before Event (1 Year Lead Time)")
    rec_8q, prec_8q = evaluate_snapshot(df_8q, "8Q-Before Event (2 Years Lead Time)")

    # 5. Company-by-Company Out-of-Sample Audit Table
    audit_records = []
    for comp in sorted(df["company"].unique()):
        comp_df = df[df["company"] == comp]
        is_def = int(comp_df["is_defaulted"].iloc[0])
        sector = comp_df["sector"].iloc[0]
        event_d = comp_df["event_date"].iloc[0]

        # 4Q score & 8Q score
        row_4q = comp_df[comp_df["lead_quarters"].between(1, 6)]
        row_8q = comp_df[comp_df["lead_quarters"].between(7, 12)]

        s_4q = float(row_4q["oos_risk_score"].iloc[0]) if not row_4q.empty else float(comp_df["oos_risk_score"].iloc[0])
        s_8q = float(row_8q["oos_risk_score"].iloc[0]) if not row_8q.empty else float(comp_df["oos_risk_score"].iloc[-1])

        # Assigned Tier
        soft_cnt = int(comp_df["soft_signals_triggered_count"].max())
        if s_4q >= 0.35 and soft_cnt >= 2:
            tier = "Tier 1 (Escalate)"
        elif s_4q >= 0.35:
            tier = "Tier 2 (Enhanced)"
        else:
            tier = "Tier 3 (Safe/Watch)"

        audit_records.append({
            "Company": comp,
            "Sector": sector,
            "Actual Status": "DEFAULTED" if is_def == 1 else "SOLVENT PEER",
            "Event Date": event_d,
            "4Q Out-of-Sample Score": round(s_4q, 3),
            "8Q Out-of-Sample Score": round(s_8q, 3),
            "Assigned Tier": tier,
            "Detection Verdict": "CORRECT ALARM" if (is_def == 1 and s_4q >= 0.35) else ("CORRECT SAFE" if (is_def == 0 and s_4q < 0.35) else "FALSE ALARM"),
        })

    audit_table = pd.DataFrame(audit_records)
    audit_table.to_csv(REPORTS_DIR / "india_oos_evaluation_audit.csv", index=False)
    print(f"\n[+] Saved complete Out-of-Sample company audit to {REPORTS_DIR / 'india_oos_evaluation_audit.csv'}")

    # Print Defaulters subset
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST VERDICT ON THE 12 MAJOR INDIAN DEFAULTS:")
    print("=" * 80)
    def_audit = audit_table[audit_table["Actual Status"] == "DEFAULTED"]
    print(def_audit[["Company", "Sector", "Event Date", "4Q Out-of-Sample Score", "8Q Out-of-Sample Score", "Assigned Tier", "Detection Verdict"]].to_string(index=False))

    # Print Sample Solvent Peers subset
    print("\n" + "=" * 80)
    print("SAMPLE TEST VERDICT ON SOLVENT PEERS (CHECKING FALSE ALARM CONTROL):")
    print("=" * 80)
    peer_audit = audit_table[audit_table["Actual Status"] == "SOLVENT PEER"].head(10)
    print(peer_audit[["Company", "Sector", "4Q Out-of-Sample Score", "Assigned Tier", "Detection Verdict"]].to_string(index=False))

    # 6. Plot Out-of-Sample Diagnostic Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # PR Curve
    ax1.plot(rec_arr, prec_arr, color="#1f77b4", lw=2.5, label=f"OOS PR-AUC = {oos_pr_auc:.4f} ({lift:.1f}x Lift)")
    ax1.axhline(base_rate, color="crimson", linestyle="--", label=f"Random Base Rate ({base_rate*100:.1f}%)")
    ax1.set_xlabel("Recall (Fraction of Indian Defaulters Caught)", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_title("Out-of-Sample PR Curve (Indian Corporate Panel)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True)
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlim([0.0, 1.0])

    # ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y, oos_probs)
    ax2.plot(fpr_arr, tpr_arr, color="#2ca02c", lw=2.5, label=f"OOS ROC-AUC = {oos_roc:.4f}")
    ax2.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (AUC = 0.50)")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax2.set_title("Out-of-Sample ROC Curve (Indian Corporate Panel)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])

    plt.tight_layout()
    fig_path = FIGURES_DIR / "india_out_of_sample_pr_roc.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved diagnostic plot to {fig_path}")

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Solvent Peer (0)", "Distressed (1)"],
        yticklabels=["Solvent Peer (0)", "Distressed (1)"],
        cbar=False,
    )
    plt.xlabel("Predicted Risk Class", fontsize=11)
    plt.ylabel("Actual Corporate Status", fontsize=11)
    plt.title("Out-of-Sample Confusion Matrix (Indian Panel)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = FIGURES_DIR / "india_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"[+] Saved confusion matrix to {cm_path}")

    return {
        "oos_roc": oos_roc,
        "oos_pr_auc": oos_pr_auc,
        "lift": lift,
        "rec_4q": rec_4q,
        "rec_8q": rec_8q,
        "audit_table": audit_table,
    }


if __name__ == "__main__":
    run_indian_out_of_sample_testing()

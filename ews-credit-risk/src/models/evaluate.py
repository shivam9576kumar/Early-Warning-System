"""Evaluation metrics, precision/recall at k alert rates, and visualization plots for credit risk.

Metrics implemented:
- Precision-Recall AUC (PR-AUC) with explicit lift over base rate
- ROC-AUC (Secondary metric)
- Alert-rate cutoffs (1%, 5%, 10%, 20%) with Precision@k and Recall@k
- Cost-based evaluation at operating threshold
- High-resolution publication plots saved to reports/figures/
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
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

# Style configuration for clean publication figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculates PR-AUC, ROC-AUC, Base Rate, Lift, Brier Score, and threshold metrics."""
    base_rate = float(np.mean(y_true))
    roc_auc = float(roc_auc_score(y_true, y_prob))

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(recalls, precisions))
    lift = pr_auc / base_rate if base_rate > 0 else 0.0

    brier = float(brier_score_loss(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) > 0 else 0.0

    # Alert rate: % of monitored population flagged
    alert_rate = float(np.mean(y_pred))

    # Fixed hackathon cost: 20 * FN + 1 * FP
    total_cost = 20 * fn + 1 * fp

    return {
        "model_name": model_name,
        "base_rate": base_rate,
        "accuracy": accuracy,
        "pr_auc": pr_auc,
        "lift": lift,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "threshold": threshold,
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "cost": int(total_cost),
    }


def compute_metrics_at_alert_rates(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alert_rates: List[float] = [0.01, 0.05, 0.10, 0.20],
) -> pd.DataFrame:
    """Computes Precision, Recall, and Lift at fixed capacity-constrained alert rate cutoffs."""
    base_rate = float(np.mean(y_true))
    rows = []

    n = len(y_prob)
    sorted_indices = np.argsort(y_prob)[::-1]

    for rate in alert_rates:
        k = max(1, int(np.round(rate * n)))
        top_k_indices = sorted_indices[:k]

        y_pred = np.zeros(n, dtype=int)
        y_pred[top_k_indices] = 1

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        prec = tp / k if k > 0 else 0.0
        rec = tp / np.sum(y_true) if np.sum(y_true) > 0 else 0.0
        lift = prec / base_rate if base_rate > 0 else 0.0
        cutoff_threshold = float(y_prob[sorted_indices[k - 1]])

        rows.append({
            "alert_rate_target": f"{int(rate*100)}%",
            "top_k_flagged": k,
            "threshold": cutoff_threshold,
            "precision": prec,
            "recall": rec,
            "lift_over_base": lift,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "cost_20fn_1fp": int(20 * fn + 1 * fp),
        })

    return pd.DataFrame(rows)


def plot_pr_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    base_rate: float,
    output_path: Path,
) -> None:
    """Plots PR curves for multiple models against the horizontal base-rate line."""
    plt.figure(figsize=(8, 6), dpi=300)

    for name, (y_true, y_prob) in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        score = auc(rec, prec)
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {score:.4f})")

    plt.axhline(
        y=base_rate,
        color="crimson",
        linestyle="--",
        label=f"Random Base Rate ({base_rate*100:.2f}%)",
    )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve on Untouched Polish Holdout Set", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.ylim([0.0, 1.02])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plots ROC curves for multiple models with random baseline diagonal."""
    plt.figure(figsize=(8, 6), dpi=300)

    for name, (y_true, y_prob) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        score = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {score:.4f})")

    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Classifier (AUC = 0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_calibration_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Plots reliability diagrams (calibration curves) before and after isotonic calibration."""
    plt.figure(figsize=(8, 6), dpi=300)

    for name, (y_true, y_prob) in models_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=name)

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Fraction of Defaults", fontsize=12)
    plt.title("Probability Calibration Curve (Reliability Diagram)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plots clean heatmap of confusion matrix with labeled rates."""
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"],
        cbar=False,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Actual Label", fontsize=12)
    plt.title(f"{title} (Threshold = {threshold:.3f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

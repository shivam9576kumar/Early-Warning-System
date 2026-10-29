"""Cost-minimizing threshold search and risk policy curve generation.

Asymmetric Cost Matrix:
- Cost = 20 * False_Negatives + 1 * False_Positives
- Rationale: Missing a catastrophic default (e.g. IL&FS default of ~INR 91,000 Crore)
  incurs immense direct write-offs, capital depletion, and legal recovery haircuts,
  whereas a False Alarm triggers incremental analyst review hours and clarification
  calls with borrower CFOs (~20x cost asymmetry ratio).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 20.0,
    cost_fp: float = 1.0,
    n_thresholds: int = 101,
) -> pd.DataFrame:
    """Sweeps decision thresholds from 0.0 to 1.0 and calculates full confusion matrix and asymmetric cost."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    records = []

    total_pos = np.sum(y_true == 1)
    total_neg = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        # Handle edge cases
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        cost = cost_fn * fn + cost_fp * fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        alert_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0

        records.append({
            "threshold": float(thresh),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "alert_rate": float(alert_rate),
            "cost": float(cost),
        })

    df_cost = pd.DataFrame(records)
    return df_cost


def find_optimal_threshold(
    cost_df: pd.DataFrame,
) -> Tuple[float, float, Dict[str, Any]]:
    """Identifies the cost-minimizing threshold and summary operating metrics."""
    min_idx = cost_df["cost"].idxmin()
    best_row = cost_df.loc[min_idx].to_dict()
    best_thresh = float(best_row["threshold"])
    min_cost = float(best_row["cost"])
    return best_thresh, min_cost, best_row


def plot_cost_vs_threshold(
    cost_df: pd.DataFrame,
    optimal_thresh: float,
    output_path: Path,
) -> None:
    """Plots asymmetric cost curve as a function of classification threshold."""
    plt.figure(figsize=(9, 5.5), dpi=300)

    plt.plot(
        cost_df["threshold"],
        cost_df["cost"],
        color="#1f77b4",
        lw=2.5,
        label="Expected Policy Cost (20*FN + 1*FP)",
    )

    opt_row = cost_df.loc[cost_df["threshold"] == optimal_thresh]
    if not opt_row.empty:
        opt_cost = opt_row.iloc[0]["cost"]
        plt.scatter(
            [optimal_thresh],
            [opt_cost],
            color="crimson",
            s=120,
            zorder=5,
            label=f"Cost-Optimal Threshold = {optimal_thresh:.3f} (Cost = {int(opt_cost):,})",
        )
        plt.axvline(x=optimal_thresh, color="crimson", linestyle=":", alpha=0.7)

    plt.xlabel("Decision Threshold (Probability of Default)", fontsize=12)
    plt.ylabel("Total Expected Cost (Arbitrary Units)", fontsize=12)
    plt.title(
        "Operating Cost vs. Decision Threshold (20x Asymmetric Loss Ratio)",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

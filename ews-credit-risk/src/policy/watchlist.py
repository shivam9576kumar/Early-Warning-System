"""Quarterly capacity-constrained watchlist selection, ranking, and tiered escalation.

Policy Architecture:
- Capacity Constraint: Exactly 40 firms selected out of an 800-firm portfolio (~5.0% alert capacity),
  matching realistic risk analyst committee review bandwidth.
- Tiering Rules:
  * Tier 1: Calibrated Risk Score > Cost-Optimal Threshold AND >= 2 Soft Signals Triggered
    -> Immediate Credit Committee Escalation / Exposure Reduction.
  * Tier 2: Calibrated Risk Score > Cost-Optimal Threshold AND 0-1 Soft Signals Triggered
    -> Enhanced Surveillance / Request CFO Clarification / Covenant Review.
  * Tier 3: Top-40 by Rank but Risk Score <= Cost-Optimal Threshold
    -> Standard Quarterly Watch / Rescore Next Quarter.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


def determine_tier_and_action(
    risk_score: float,
    soft_signals_count: int,
    cost_optimal_threshold: float,
) -> Tuple[str, str]:
    """Applies policy tiering and assigns prescribed credit actions."""
    if risk_score > cost_optimal_threshold:
        if soft_signals_count >= 2:
            tier = "Tier 1"
            action = (
                "Immediate Credit Committee Escalation: Freeze incremental facility limits, "
                "demand supplementary collateral, inspect related-party exposure, and audit encumbered shares."
            )
        else:
            tier = "Tier 2"
            action = (
                "Enhanced Monitoring: Request CFO clarification regarding cashflow stretch, "
                "tighten quarterly debt-service covenants, and monitor rating outlook."
            )
    else:
        tier = "Tier 3"
        action = (
            "Standard Surveillance: Maintain facility under observation, review scheduled quarterly filings, "
            "and re-score at next covenant reporting date."
        )

    return tier, action


def generate_quarterly_watchlist(
    scored_universe_df: pd.DataFrame,
    cost_optimal_threshold: float,
    capacity_limit: int = 40,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Ranks monitored portfolio and generates top-40 capacity-constrained watchlist.

    Required columns in scored_universe_df:
    - company: Company name
    - sector: Industry sector
    - risk_score: Calibrated probability of default / distress
    - soft_signals_triggered_count: Count of active soft red flags
    - top_driver_1, top_driver_2, top_driver_3: SHAP narrative text strings
    """
    df = scored_universe_df.copy()

    # Rank by risk_score descending
    df = df.sort_values(by="risk_score", ascending=False).reset_index(drop=True)

    # Select top-k firms (capacity-constrained)
    watchlist = df.head(capacity_limit).copy()
    watchlist["rank"] = range(1, len(watchlist) + 1)

    tiers = []
    actions = []

    for _, row in watchlist.iterrows():
        score = float(row["risk_score"])
        soft_cnt = int(row.get("soft_signals_triggered_count", 0))
        tier, action = determine_tier_and_action(score, soft_cnt, cost_optimal_threshold)
        tiers.append(tier)
        actions.append(action)

    watchlist["tier"] = tiers
    watchlist["recommended_action"] = actions

    # Format output columns per specification
    final_cols = [
        "rank",
        "company",
        "sector",
        "risk_score",
        "tier",
        "top_driver_1",
        "top_driver_2",
        "top_driver_3",
        "recommended_action",
    ]

    # Ensure all required driver columns exist
    for c in ["top_driver_1", "top_driver_2", "top_driver_3"]:
        if c not in watchlist.columns:
            watchlist[c] = "Financial ratio deterioration"

    watchlist_export = watchlist[final_cols].copy()

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        watchlist_export.to_csv(output_csv, index=False)
        print(f"[+] Saved quarterly watchlist ({len(watchlist_export)} companies) to {output_csv}")

    return watchlist_export

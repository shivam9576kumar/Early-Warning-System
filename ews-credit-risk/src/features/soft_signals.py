"""Soft warning signals encoder for corporate credit distress and governance tampering.

Signals implemented:
1. auditor_resignation_flag (binary) + days_before_event
2. promoter_pledge_pct + 4-quarter rolling slope (QoQ trend / pledge_slope_4q)
3. related_party_transactions_to_revenue + YoY growth rate (rpt_growth_yoy)
4. credit_rating_action_score: ordinal encoding:
   (+1 upgrade, 0 affirm, -1 negative watch, -2 downgrade, -3 default)
5. filing_delay_days: delay in days beyond the statutory 180-day deadline
"""

from typing import Dict, Union
import numpy as np
import pandas as pd

RATING_ACTION_MAP = {
    "upgrade": 1,
    "positive_watch": 1,
    "affirm": 0,
    "reaffirm": 0,
    "stable": 0,
    "negative_watch": -1,
    "watch_negative": -1,
    "downgrade": -2,
    "default": -3,
    "d": -3,
}


def encode_rating_action(action: Union[str, int, float]) -> int:
    """Encodes credit rating actions into an ordinal risk metric.

    +1: Upgrade
     0: Affirmation / Stable
    -1: Negative Watch
    -2: Downgrade
    -3: Default / Severe Impairment
    """
    if pd.isna(action):
        return 0
    if isinstance(action, (int, float)):
        return int(np.clip(action, -3, 1))
    clean_str = str(action).strip().lower().replace(" ", "_")
    return RATING_ACTION_MAP.get(clean_str, 0)


def compute_soft_signals_composite(
    auditor_resignation_flag: Union[int, pd.Series],
    promoter_pledge_pct: Union[float, pd.Series],
    pledge_slope_4q: Union[float, pd.Series],
    rpt_to_revenue: Union[float, pd.Series],
    rpt_growth_yoy: Union[float, pd.Series],
    rating_action_score: Union[int, pd.Series],
    filing_delay_days: Union[int, pd.Series] = 0,
) -> Union[int, pd.Series]:
    """Computes total count of active red-flag soft signals.

    A signal triggers if:
    - Auditor abruptly resigned
    - Promoter pledge is high (>30%) or accelerating (+10% over 4 quarters)
    - High Related Party Transactions (>15% revenue or growing >25% YoY)
    - Negative Rating action (downgrade or negative watch)
    - Statutory filing delayed by > 15 days
    """
    c1 = (auditor_resignation_flag > 0).astype(int)
    c2 = ((promoter_pledge_pct > 30.0) | (pledge_slope_4q > 5.0)).astype(int)
    c3 = ((rpt_to_revenue > 0.15) | (rpt_growth_yoy > 0.25)).astype(int)
    c4 = (rating_action_score < 0).astype(int)
    c5 = (filing_delay_days > 15).astype(int)

    return c1 + c2 + c3 + c4 + c5


def add_soft_signal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes and adds soft-signal features to a DataFrame."""
    res = df.copy()

    # Ensure required columns exist with sensible defaults
    if "auditor_resignation_flag" not in res.columns:
        res["auditor_resignation_flag"] = 0
    if "auditor_resignation_days_before" not in res.columns:
        res["auditor_resignation_days_before"] = 0

    if "promoter_pledge_pct" not in res.columns:
        res["promoter_pledge_pct"] = 0.0
    if "pledge_slope_4q" not in res.columns:
        res["pledge_slope_4q"] = 0.0

    if "related_party_transactions_to_revenue" not in res.columns:
        res["related_party_transactions_to_revenue"] = 0.0
    if "rpt_growth_yoy" not in res.columns:
        res["rpt_growth_yoy"] = 0.0

    if "credit_rating_action" in res.columns:
        res["credit_rating_action_score"] = res["credit_rating_action"].apply(encode_rating_action)
    elif "credit_rating_action_score" not in res.columns:
        res["credit_rating_action_score"] = 0

    if "filing_delay_days" not in res.columns:
        res["filing_delay_days"] = 0

    res["soft_signals_triggered_count"] = compute_soft_signals_composite(
        auditor_resignation_flag=res["auditor_resignation_flag"],
        promoter_pledge_pct=res["promoter_pledge_pct"],
        pledge_slope_4q=res["pledge_slope_4q"],
        rpt_to_revenue=res["related_party_transactions_to_revenue"],
        rpt_growth_yoy=res["rpt_growth_yoy"],
        rating_action_score=res["credit_rating_action_score"],
        filing_delay_days=res["filing_delay_days"],
    )

    return res

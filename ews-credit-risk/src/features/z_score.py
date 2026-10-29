"""Altman Z-Score feature calculation and distress classification.

Formula:
Z = 1.2 * (WC / TA) + 1.4 * (RE / TA) + 3.3 * (EBIT / TA) + 0.6 * (MVE / TL) + 1.0 * (Sales / TA)

Distress Thresholds:
- Z < 1.81: Distress Zone (high probability of bankruptcy)
- 1.81 <= Z <= 2.99: Grey Zone (moderate risk)
- Z > 2.99: Safe Zone (low default risk)
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_altman_z_score(
    wc_to_ta: Union[float, pd.Series],
    re_to_ta: Union[float, pd.Series],
    ebit_to_ta: Union[float, pd.Series],
    mve_to_tl: Union[float, pd.Series],
    sales_to_ta: Union[float, pd.Series],
) -> Union[float, pd.Series]:
    """Computes the original Altman Z-Score (1968).

    Args:
        wc_to_ta: Working capital / Total assets
        re_to_ta: Retained earnings / Total assets
        ebit_to_ta: EBIT / Total assets
        mve_to_tl: Market value of equity (or M-Cap) / Total liabilities
        sales_to_ta: Sales / Total assets

    Returns:
        Altman Z-Score value or series.
    """
    z = (
        1.2 * wc_to_ta
        + 1.4 * re_to_ta
        + 3.3 * ebit_to_ta
        + 0.6 * mve_to_tl
        + 1.0 * sales_to_ta
    )
    return z


def classify_z_zone(z_score: float) -> str:
    """Classifies a Z-score into Altman risk zones."""
    if pd.isna(z_score):
        return "Unknown"
    if z_score < 1.81:
        return "Distress"
    elif z_score <= 2.99:
        return "Grey"
    return "Safe"


def add_z_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Altman Z-Score and category to DataFrame."""
    res = df.copy()

    # Map column names if present
    wc_ta = res.get("working_capital_to_total_assets", res.get("Attr3", 0.0))
    re_ta = res.get("retained_earnings_to_total_assets", res.get("Attr6", 0.0))
    ebit_ta = res.get("ebit_to_total_assets", res.get("Attr7", 0.0))
    mve_tl = res.get(
        "market_cap_to_total_liabilities",
        res.get("Attr8", res.get("debt_to_equity", 1.0).map(lambda x: 1.0 / max(x, 0.01))),
    )
    sales_ta = res.get("sales_to_total_assets", res.get("Attr9", 0.0))

    res["z_score"] = compute_altman_z_score(wc_ta, re_ta, ebit_ta, mve_tl, sales_ta)
    res["z_distress_flag"] = (res["z_score"] < 1.81).astype(int)
    res["z_zone"] = res["z_score"].apply(classify_z_zone)
    return res

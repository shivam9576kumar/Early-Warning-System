"""Piotroski F-Score (0-9 scale) implementation for financial health screening.

The Piotroski F-Score evaluates 9 fundamental binary signals across three categories:
1. Profitability:
   - ROA > 0 (1 point)
   - CFO (Cash Flow from Operations) > 0 (1 point)
   - Delta ROA > 0 (YoY ROA improvement) (1 point)
   - Accrual: CFO > Net Income / ROA (Quality of earnings) (1 point)

2. Leverage, Liquidity, and Source of Funds:
   - Delta Leverage <= 0 (Debt-to-Assets/Equity decreased YoY) (1 point)
   - Delta Current Ratio > 0 (Current Ratio improved YoY) (1 point)
   - No Equity Dilution / Share issuance <= 0 (1 point)

3. Operating Efficiency:
   - Delta Gross Margin > 0 (Gross Margin improved YoY) (1 point)
   - Delta Asset Turnover > 0 (Asset Turnover improved YoY) (1 point)

Total Score: 0 to 9.
- 0 to 3: Weak / High Distress Risk
- 4 to 6: Neutral / Average
- 7 to 9: Strong Fundamentals
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def compute_piotroski_f_score_row(
    roa: float,
    cfo: float,
    delta_roa: float,
    cfo_gt_ni: bool,
    delta_leverage: float,
    delta_current_ratio: float,
    new_shares_issued: bool,
    delta_margin: float,
    delta_turnover: float,
) -> int:
    """Computes Piotroski F-score (0-9) for a single observation."""
    score = 0
    # Profitability
    if roa > 0:
        score += 1
    if cfo > 0:
        score += 1
    if delta_roa > 0:
        score += 1
    if cfo_gt_ni:
        score += 1

    # Leverage & Liquidity
    if delta_leverage <= 0:
        score += 1
    if delta_current_ratio > 0:
        score += 1
    if not new_shares_issued:
        score += 1

    # Operating Efficiency
    if delta_margin > 0:
        score += 1
    if delta_turnover > 0:
        score += 1

    return score


def add_f_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized / batch Piotroski F-score feature addition for panel data."""
    res = df.copy()

    roa = res.get("roa", res.get("Attr1", pd.Series(0.0, index=res.index)))
    cfo = res.get("cash_flow_to_total_debt", res.get("Attr5", pd.Series(0.0, index=res.index)))
    delta_roa = res.get("delta_roa", pd.Series(0.0, index=res.index))
    cfo_gt_ni = (cfo > roa) if "cash_flow_to_total_debt" in res and "roa" in res else pd.Series(True, index=res.index)
    delta_lev = res.get("delta_leverage", pd.Series(0.0, index=res.index))
    delta_cr = res.get("delta_current_ratio", pd.Series(0.0, index=res.index))
    dilution = res.get("new_shares_issued", pd.Series(False, index=res.index))
    delta_margin = res.get("delta_margin", pd.Series(0.0, index=res.index))
    delta_turnover = res.get("delta_turnover", pd.Series(0.0, index=res.index))

    scores = []
    for idx in res.index:
        s = compute_piotroski_f_score_row(
            roa=float(roa.loc[idx]) if pd.notna(roa.loc[idx]) else 0.0,
            cfo=float(cfo.loc[idx]) if pd.notna(cfo.loc[idx]) else 0.0,
            delta_roa=float(delta_roa.loc[idx]) if pd.notna(delta_roa.loc[idx]) else 0.0,
            cfo_gt_ni=bool(cfo_gt_ni.loc[idx]) if pd.notna(cfo_gt_ni.loc[idx]) else False,
            delta_leverage=float(delta_lev.loc[idx]) if pd.notna(delta_lev.loc[idx]) else 0.0,
            delta_current_ratio=float(delta_cr.loc[idx]) if pd.notna(delta_cr.loc[idx]) else 0.0,
            new_shares_issued=bool(dilution.loc[idx]) if pd.notna(dilution.loc[idx]) else False,
            delta_margin=float(delta_margin.loc[idx]) if pd.notna(delta_margin.loc[idx]) else 0.0,
            delta_turnover=float(delta_turnover.loc[idx]) if pd.notna(delta_turnover.loc[idx]) else 0.0,
        )
        scores.append(s)

    res["f_score"] = scores
    res["f_weak_flag"] = (res["f_score"] <= 3).astype(int)
    return res

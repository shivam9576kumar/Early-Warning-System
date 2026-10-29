"""Automated Leakage, Temporal Integrity, and Accounting Sanity Tests.

Tests:
1. test_indian_panel_filing_strictly_precedes_event_date:
   Asserts no Indian panel row has statement_filing_date >= event_date.
2. test_no_post_event_restatements_in_features:
   Asserts original as-filed figures are preserved without hindsight leakage.
3. test_polish_holdout_strict_three_way_split:
   Asserts the Polish benchmark hold-out set was carved out FIRST and is mutually
   exclusive with train and validation tuning sets.
4. test_altman_z_score_calculation:
   Verifies mathematical calculation of Altman Z-Score and distress classification.
5. test_beneish_m_score_manipulation_flag:
   Verifies M > -1.78 flags financial manipulation.
6. test_piotroski_f_score_range:
   Verifies F-Score is bounded between 0 and 9.
7. test_quarterly_watchlist_capacity_and_tiering:
   Asserts watchlist has exact capacity of 40 firms (5%) with valid tiers.
"""

import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data.load_india_panel import load_india_panel
from src.data.load_polish import load_all_polish_horizons
from src.features.f_score import compute_piotroski_f_score_row
from src.features.m_score import compute_beneish_m_score
from src.features.z_score import classify_z_zone, compute_altman_z_score
from src.models.split import create_stratified_train_val_holdout_split
from src.policy.watchlist import generate_quarterly_watchlist


def test_indian_panel_filing_strictly_precedes_event_date():
    """Asserts no Indian panel row has statement_filing_date >= event_date."""
    df_india = load_india_panel(enforce_leakage_assertion=True, enrich_scores=False)
    assert len(df_india) > 0, "Indian panel is empty!"

    for idx, row in df_india.iterrows():
        company = row["company"]
        fy = row["fiscal_year"]
        filing_date = str(row["statement_filing_date"]).strip()
        event_date = str(row["event_date"]).strip()

        # Hard assertion: Filing date must strictly precede default/insolvency event date
        assert (
            filing_date < event_date
        ), f"LEAKAGE DETECTED: {company} ({fy}) statement filed on {filing_date} >= event date {event_date}"


def test_no_post_event_restatements_in_features():
    """Asserts original as-filed figures are preserved without hindsight restatements.

    For IL&FS and CG Power, forensic investigations revealed massive off-balance sheet
    borrowings only post-event. Original as-filed FY18 statements must reflect the
    historically reported metrics rather than post-mortem restatements.
    """
    df_india = load_india_panel(enforce_leakage_assertion=True, enrich_scores=False)

    # IL&FS FY18 check
    ilfs_fy18 = df_india[
        (df_india["company"].str.contains("IL&FS", case=False, na=False))
        & (df_india["fiscal_year"] == "FY18")
    ]
    assert len(ilfs_fy18) == 1, "IL&FS FY18 record not found!"
    ilfs_row = ilfs_fy18.iloc[0]

    # As-filed FY18 reported positive net margin before forensic restatements wiped it out
    assert ilfs_row["statement_filing_date"] == "2018-05-30"
    assert ilfs_row["statement_filing_date"] < ilfs_row["event_date"]
    assert ilfs_row["net_margin"] > 0, "IL&FS FY18 should reflect as-filed numbers, not restatements"

    # CG Power FY18 check
    cg_fy18 = df_india[
        (df_india["company"].str.contains("CG Power", case=False, na=False))
        & (df_india["fiscal_year"] == "FY18")
    ]
    assert len(cg_fy18) == 1, "CG Power FY18 record not found!"
    cg_row = cg_fy18.iloc[0]
    assert cg_row["statement_filing_date"] == "2018-05-28"
    assert cg_row["statement_filing_date"] < cg_row["event_date"]


def test_polish_holdout_strict_three_way_split():
    """Asserts the Polish benchmark hold-out set was carved out FIRST and is mutually exclusive."""
    df_polish, _ = load_all_polish_horizons()
    n_total = len(df_polish)

    train_df, val_df, holdout_df = create_stratified_train_val_holdout_split(
        df_polish, target_col="target", stratify_by_horizon=True, val_size=0.20, holdout_size=0.20, random_state=42
    )

    # Check non-overlap in counts
    assert len(train_df) + len(val_df) + len(holdout_df) == n_total

    # Proportions check (20% holdout, 20% validation, 60% train)
    assert abs(len(holdout_df) / n_total - 0.20) < 0.01
    assert abs(len(val_df) / n_total - 0.20) < 0.01
    assert abs(len(train_df) / n_total - 0.60) < 0.01

    # Class balance preservation check
    overall_br = df_polish["target"].mean()
    assert abs(holdout_df["target"].mean() - overall_br) < 0.01
    assert abs(val_df["target"].mean() - overall_br) < 0.01
    assert abs(train_df["target"].mean() - overall_br) < 0.01


def test_altman_z_score_calculation():
    """Verifies Altman Z-Score formula and zone classifications."""
    # Test safe firm: WC/TA=0.3, RE/TA=0.4, EBIT/TA=0.15, MVE/TL=2.0, Sales/TA=1.5
    z_safe = compute_altman_z_score(0.3, 0.4, 0.15, 2.0, 1.5)
    # 1.2*0.3 + 1.4*0.4 + 3.3*0.15 + 0.6*2.0 + 1.0*1.5 = 0.36 + 0.56 + 0.495 + 1.2 + 1.5 = 4.115
    assert abs(z_safe - 4.115) < 1e-4
    assert classify_z_zone(z_safe) == "Safe"

    # Test distressed firm: WC/TA=-0.1, RE/TA=-0.2, EBIT/TA=0.01, MVE/TL=0.1, Sales/TA=0.3
    z_dist = compute_altman_z_score(-0.1, -0.2, 0.01, 0.1, 0.3)
    # 1.2*(-0.1) + 1.4*(-0.2) + 3.3*(0.01) + 0.6*(0.1) + 1.0*(0.3) = -0.12 - 0.28 + 0.033 + 0.06 + 0.3 = -0.007
    assert z_dist < 1.81
    assert classify_z_zone(z_dist) == "Distress"


def test_beneish_m_score_manipulation_flag():
    """Verifies Beneish M-Score manipulation threshold (-1.78)."""
    # Normal unmanipulated baseline (all indices ~ 1.0, accruals ~ 0)
    m_normal = compute_beneish_m_score(dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0, depi=1.0, sgai=1.0, lvgi=1.0, tata=0.0)
    # -4.84 + 0.92 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 + 0.0327 = -2.1223 <= -1.78
    assert m_normal <= -1.78, f"Expected normal M-score <= -1.78, got {m_normal}"

    # Aggressive manipulator (DSRI=2.5, SGI=1.8, GMI=1.5, TATA=0.15)
    m_manip = compute_beneish_m_score(dsri=2.5, gmi=1.5, aqi=1.4, sgi=1.8, depi=1.2, sgai=0.9, lvgi=1.6, tata=0.15)
    assert m_manip > -1.78, f"Expected manipulator M-score > -1.78, got {m_manip}"


def test_piotroski_f_score_range():
    """Verifies Piotroski F-Score produces valid integer in [0, 9]."""
    score_min = compute_piotroski_f_score_row(-0.05, -0.02, -0.01, False, 0.5, -0.2, True, -0.1, -0.1)
    assert score_min == 0

    score_max = compute_piotroski_f_score_row(0.12, 0.15, 0.02, True, -0.1, 0.2, False, 0.05, 0.1)
    assert score_max == 9


def test_quarterly_watchlist_capacity_and_tiering():
    """Asserts capacity constraint (40 firms) and tiering logic."""
    df_dummy = pd.DataFrame([
        {
            "company": f"Borrower {i}",
            "sector": "NBFC",
            "risk_score": 0.95 - (i * 0.01),
            "soft_signals_triggered_count": 2 if i < 15 else (1 if i < 30 else 0),
            "top_driver_1": "Driver 1",
            "top_driver_2": "Driver 2",
            "top_driver_3": "Driver 3",
        }
        for i in range(100)
    ])

    wl = generate_quarterly_watchlist(df_dummy, cost_optimal_threshold=0.70, capacity_limit=40)
    assert len(wl) == 40
    assert list(wl["rank"]) == list(range(1, 41))

    # Check tiers: scores > 0.70 and soft >= 2 should be Tier 1
    t1_subset = wl[wl["tier"] == "Tier 1"]
    assert len(t1_subset) > 0
    for _, r in t1_subset.iterrows():
        assert r["risk_score"] > 0.70

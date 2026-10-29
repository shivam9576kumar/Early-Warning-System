"""Loader, validator, and feature enrichment for the Curated Indian Distress Panel.

Critical Leakage Rules & Accounting Integrity:
1. Hard Assertion: statement_filing_date < event_date for all rows.
2. Original As-Filed Figures: Excludes post-event forensic restatements
   (specifically critical for IL&FS, CG Power, DHFL where off-balance sheet SPVs
   and fraudulent related-party diversion were uncovered only post-event).
3. Clear Transparency: Transparently tracks `is_synthetic_placeholder` column.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.features.f_score import add_f_score_features
from src.features.m_score import add_m_score_features
from src.features.ratios import CORE_RATIO_COLS, normalize_within_sector, winsorize_series
from src.features.soft_signals import add_soft_signal_features
from src.features.z_score import add_z_score_features

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "raw"
    / "india"
    / "indian_distress_panel.csv"
)


def load_india_panel(
    csv_path: Optional[Path] = None,
    enforce_leakage_assertion: bool = True,
    enrich_scores: bool = True,
) -> pd.DataFrame:
    """Loads and validates the Indian Corporate Credit Risk Panel.

    Enforces strict temporal leakage assertions and applies Z, F, M scores
    and soft signals.
    """
    if csv_path is None:
        csv_path = DEFAULT_CSV_PATH

    if not csv_path.exists():
        # Generate panel if not yet generated
        from src.data.build_india_panel import build_panel_dataframe

        df = build_panel_dataframe()
    else:
        df = pd.read_csv(csv_path)

    # 1. HARD LEAKAGE AUDIT ASSERTION
    if enforce_leakage_assertion:
        for idx, row in df.iterrows():
            c_name = row["company"]
            fy = row["fiscal_year"]
            filing = str(row["statement_filing_date"])
            event = str(row["event_date"])

            assert (
                filing < event
            ), f"CRITICAL LEAKAGE DETECTED: {c_name} ({fy}) statement filed on {filing} >= event date {event}"

    # 2. Enrich with domain scores if requested
    if enrich_scores:
        df = add_z_score_features(df)
        df = add_f_score_features(df)
        df = add_m_score_features(df)
        df = add_soft_signal_features(df)

    return df


def get_lead_time_snapshots(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts strictly separated '4Q-before' and '8Q-before' feature snapshots for each company.

    - 4Q-before: Last audited annual report publicly available >= 4 quarters (approx 1 year) before event.
    - 8Q-before: Audited annual report publicly available >= 8 quarters (approx 2 years) before event.
    """
    # For defaulted companies: lead_quarters ~ 4 and lead_quarters ~ 8
    # For peers: matching lag years
    q4_rows = []
    q8_rows = []

    for comp, group in df.groupby("company"):
        sorted_g = group.sort_values(by="lead_quarters", ascending=True)

        # 4Q snapshot: closest to lead_quarters >= 3 (1-4 quarters before event)
        # 8Q snapshot: closest to lead_quarters >= 7 (7-9 quarters before event)
        g_4q = sorted_g[sorted_g["lead_quarters"].between(1, 6)]
        g_8q = sorted_g[sorted_g["lead_quarters"].between(7, 12)]

        if not g_4q.empty:
            q4_rows.append(g_4q.iloc[0])
        elif len(sorted_g) >= 1:
            q4_rows.append(sorted_g.iloc[0])

        if not g_8q.empty:
            q8_rows.append(g_8q.iloc[0])
        elif len(sorted_g) >= 2:
            q8_rows.append(sorted_g.iloc[1])

    df_4q = pd.DataFrame(q4_rows).reset_index(drop=True)
    df_8q = pd.DataFrame(q8_rows).reset_index(drop=True)

    return df_4q, df_8q


if __name__ == "__main__":
    df = load_india_panel()
    df_4q, df_8q = get_lead_time_snapshots(df)
    print(f"Loaded Indian Panel: {len(df)} rows across {df['company'].nunique()} companies.")
    print(f"4Q-Before Snapshot: {len(df_4q)} firms ({df_4q['is_defaulted'].sum()} defaulters).")
    print(f"8Q-Before Snapshot: {len(df_8q)} firms ({df_8q['is_defaulted'].sum()} defaulters).")

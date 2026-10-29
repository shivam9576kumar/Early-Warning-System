"""Time-aware and Group-aware dataset splitting for corporate credit risk.

Leakage Controls & Methodology:
1. True Holdout Set: Carved out FIRST and strictly preserved until final evaluation.
   No hyperparameter tuning, feature selection, or threshold calibration touches this set.
2. 3-Way Split:
   - Train (60%): Model learning
   - Validation (20%): Early stopping, calibration, threshold optimization
   - Holdout (20%): Untouched evaluation
3. Lead-Time Evaluation Split:
   Allows testing models trained on longer horizons (e.g. Horizons 4-5) against short horizons (Horizons 1-2).
4. Anonymization / Grouping Caveat:
   In the UCI Polish dataset (Zieba et al.), firm IDs were anonymized by the authors.
   Consequently, identical firms cannot be explicitly tracked across annual ARFF files.
   This structural limitation is explicitly addressed and documented.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def create_stratified_train_val_holdout_split(
    df: pd.DataFrame,
    target_col: str = "target",
    stratify_by_horizon: bool = True,
    val_size: float = 0.20,
    holdout_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset into 3 strictly separated sets: Train, Validation, and Holdout.

    The holdout set is carved out first to avoid any leakage during hyperparameter tuning
    or calibration. Stratification is performed jointly on target and horizon to preserve
    class imbalance and temporal horizon proportions.
    """
    if stratify_by_horizon and "horizon" in df.columns:
        # Create composite stratify key: e.g. "h1_c0", "h1_c1", etc.
        strat_key = df["horizon"].astype(str) + "_" + df[target_col].astype(str)
    else:
        strat_key = df[target_col].astype(str)

    # First split: Holdout (untouched) vs Dev (train + val)
    sss_holdout = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_size, random_state=random_state
    )
    dev_idx, holdout_idx = next(sss_holdout.split(df, strat_key))

    dev_df = df.iloc[dev_idx].copy().reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].copy().reset_index(drop=True)

    # Second split: Train vs Validation from Dev set
    # Adjusted val_size relative to dev set: val_size / (1.0 - holdout_size)
    rel_val_size = val_size / (1.0 - holdout_size)
    dev_strat_key = (
        (dev_df["horizon"].astype(str) + "_" + dev_df[target_col].astype(str))
        if stratify_by_horizon and "horizon" in dev_df.columns
        else dev_df[target_col].astype(str)
    )

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=rel_val_size, random_state=random_state + 1
    )
    train_idx, val_idx = next(sss_val.split(dev_df, dev_strat_key))

    train_df = dev_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = dev_df.iloc[val_idx].copy().reset_index(drop=True)

    print(
        f"[*] Split Complete -> Train: {len(train_df):,} rows ({train_df[target_col].mean()*100:.2f}% bankrupt) | "
        f"Val: {len(val_df):,} rows ({val_df[target_col].mean()*100:.2f}% bankrupt) | "
        f"Holdout: {len(holdout_df):,} rows ({holdout_df[target_col].mean()*100:.2f}% bankrupt)"
    )

    return train_df, val_df, holdout_df


def create_horizon_lead_time_split(
    df: pd.DataFrame,
    early_horizons: Tuple[int, ...] = (4, 5),
    late_horizons: Tuple[int, ...] = (1, 2),
    target_col: str = "target",
    val_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data temporally by horizon:

    Early horizons (e.g. 4 and 5 years prior to bankruptcy) are used for training and
    validating earlier warning signals; late horizons (e.g. 1 and 2 years prior) serve as test.
    """
    early_df = df[df["horizon"].isin(early_horizons)].copy().reset_index(drop=True)
    late_df = df[df["horizon"].isin(late_horizons)].copy().reset_index(drop=True)

    strat_key = early_df["horizon"].astype(str) + "_" + early_df[target_col].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    tr_idx, val_idx = next(sss.split(early_df, strat_key))

    train_df = early_df.iloc[tr_idx].copy().reset_index(drop=True)
    val_df = early_df.iloc[val_idx].copy().reset_index(drop=True)

    return train_df, val_df, late_df

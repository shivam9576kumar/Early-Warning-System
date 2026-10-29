"""ARFF loader and preprocessor for UCI Polish Companies Bankruptcy Dataset.

Dataset Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
Contains 5 ARFF files corresponding to 1 to 5 years prior to bankruptcy:
- 1year.arff: 1 year prior to bankruptcy
- 2year.arff: 2 years prior to bankruptcy
- 3year.arff: 3 years prior to bankruptcy
- 4year.arff: 4 years prior to bankruptcy
- 5year.arff: 5 years prior to bankruptcy

Attributes: 64 financial ratios (Attr1 to Attr64) + class ('0' = active, '1' = bankrupt).
Missing values encoded as '?'.
"""

import io
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

UCI_URL = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"


def ensure_polish_data_downloaded(raw_dir: Path) -> List[Path]:
    """Ensures all 5 ARFF files exist in raw_dir, downloading and unzipping if needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_files = [raw_dir / f"{h}year.arff" for h in range(1, 6)]

    if all(f.exists() and f.stat().st_size > 0 for f in expected_files):
        return expected_files

    print(f"[*] Downloading Polish Bankruptcy Dataset from UCI: {UCI_URL} ...")
    zip_path = raw_dir / "polish_data.zip"
    urllib.request.urlretrieve(UCI_URL, zip_path)

    print(f"[*] Extracting {zip_path.name} to {raw_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    # Some zip archives package another zip or direct files
    nested_zips = list(raw_dir.glob("*.zip"))
    for nz in nested_zips:
        if nz != zip_path:
            with zipfile.ZipFile(nz, "r") as z:
                z.extractall(raw_dir)

    for f in expected_files:
        if not f.exists():
            raise FileNotFoundError(
                f"Expected file {f} not found after extraction. Files in directory: {list(raw_dir.glob('*'))}"
            )

    print("[+] All 5 Polish ARFF files successfully downloaded and verified.")
    return expected_files


def parse_arff_file(file_path: Path, horizon: int) -> pd.DataFrame:
    """Parses a single Polish bankruptcy ARFF file, replacing '?' with NaN."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start = False
    attributes = []
    data_rows = []

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        if line_s.lower().startswith("@attribute"):
            parts = line_s.split()
            attributes.append(parts[1])
        elif line_s.lower().startswith("@data"):
            data_start = True
            continue
        elif data_start:
            # Data row
            data_rows.append(line_s)

    # If attributes weren't listed with @attribute, generate Attr1..Attr64 + class
    if len(attributes) < 65:
        attributes = [f"Attr{i}" for i in range(1, 65)] + ["class"]

    # Parse rows via CSV buffer
    csv_text = "\n".join(data_rows)
    df = pd.read_csv(
        io.StringIO(csv_text),
        names=attributes,
        header=None,
        na_values=["?", "nan", "NaN", ""],
        dtype={attributes[-1]: str},
    )

    # Clean target class ('0' / '1' or b'0' / b'1')
    target_col = attributes[-1]
    df["target"] = (
        df[target_col].astype(str).str.strip().str.replace("'", "").str.replace('"', "")
    )
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    if target_col != "target":
        df = df.drop(columns=[target_col])

    # Ensure all Attr1..Attr64 are numeric floats
    for i in range(1, 65):
        col = f"Attr{i}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["horizon"] = horizon
    return df


def load_all_polish_horizons(
    raw_dir: Optional[Path] = None,
    rename_to_x: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """Loads all 5 Polish ARFF files, tags horizon, and calculates class imbalance metrics.

    Args:
        raw_dir: Path to raw ARFF directory.
        rename_to_x: If True, renames Attr1..Attr64 to X1..X64 to match UCI web documentation.

    Returns:
        combined_df: DataFrame with all rows, Attr1..Attr64 (or X1..X64), target, horizon.
        horizon_stats: Dict of statistics per horizon (rows, features, missing %, bankrupt %).
    """
    if raw_dir is None:
        raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "polish"

    ensure_polish_data_downloaded(raw_dir)

    dfs = []
    horizon_stats = {}

    print("\n" + "=" * 70)
    print("POLISH COMPANIES BANKRUPTCY DATASET — HORIZON-BY-HORIZON REPORT")
    print("=" * 70)
    print(
        f"{'Horizon':<10} | {'Total Rows':<12} | {'Features':<10} | {'Missing %':<12} | {'Bankrupt %':<12} | {'Bankrupt Count':<14}"
    )
    print("-" * 70)

    for h in range(1, 6):
        arff_path = raw_dir / f"{h}year.arff"
        hdf = parse_arff_file(arff_path, horizon=h)
        dfs.append(hdf)

        total_rows = len(hdf)
        feature_cols = [c for c in hdf.columns if c.startswith("Attr")]
        total_cells = total_rows * len(feature_cols)
        missing_cells = hdf[feature_cols].isna().sum().sum()
        missing_pct = (missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

        bankrupt_cnt = int(hdf["target"].sum())
        bankrupt_pct = (bankrupt_cnt / total_rows) * 100.0 if total_rows > 0 else 0.0

        horizon_stats[h] = {
            "total_rows": total_rows,
            "feature_count": len(feature_cols),
            "missing_pct": missing_pct,
            "bankrupt_count": bankrupt_cnt,
            "bankrupt_pct": bankrupt_pct,
        }

        print(
            f"Horizon-{h:<2} | {total_rows:<12} | {len(feature_cols):<10} | {missing_pct:>9.2f}% | {bankrupt_pct:>9.2f}% | {bankrupt_cnt:<14}"
        )

    print("-" * 70)
    combined_df = pd.concat(dfs, ignore_index=True)
    total_all = len(combined_df)
    bankrupt_all = int(combined_df["target"].sum())
    bankrupt_all_pct = (bankrupt_all / total_all) * 100.0
    print(
        f"{'COMBINED':<10} | {total_all:<12} | {64:<10} | "
        f"{((combined_df[[f'Attr{i}' for i in range(1, 65)]].isna().sum().sum() / (total_all * 64)) * 100.0):>9.2f}% | "
        f"{bankrupt_all_pct:>9.2f}% | {bankrupt_all:<14}"
    )
    print("=" * 70 + "\n")

    if rename_to_x:
        x_map = {f"Attr{i}": f"X{i}" for i in range(1, 65)}
        combined_df = combined_df.rename(columns=x_map)

    return combined_df, horizon_stats


def preprocess_polish_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    impute_strategy: str = "median",
    scale_features: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], SimpleImputer, Optional[StandardScaler]]:
    """Preprocesses features using train-fit-only imputation and optional scaling.

    Rationale for Median Imputation:
    Financial ratios contain extreme outliers (e.g. division by near-zero equity or liabilities).
    Mean imputation is heavily skewed by such spikes, whereas median imputation provides an
    unbiased, robust measure of central tendency that preserves tree partition boundaries
    without leaking test distribution information.
    """
    imputer = SimpleImputer(strategy=impute_strategy)
    X_train_imp = imputer.fit_transform(X_train)

    X_val_imp = imputer.transform(X_val) if X_val is not None else None
    X_test_imp = imputer.transform(X_test) if X_test is not None else None

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train_imp = scaler.fit_transform(X_train_imp)
        if X_val_imp is not None:
            X_val_imp = scaler.transform(X_val_imp)
        if X_test_imp is not None:
            X_test_imp = scaler.transform(X_test_imp)

    return X_train_imp, X_val_imp, X_test_imp, imputer, scaler


if __name__ == "__main__":
    df, stats = load_all_polish_horizons()
    print(f"Successfully loaded {len(df)} total rows across 5 horizons.")

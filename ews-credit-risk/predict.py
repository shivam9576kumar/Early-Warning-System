"""Standalone Prediction & Risk Scoring Script for Early-Warning Corporate Credit Risk (EWS).

This script allows any user to test their own company dataset on the EWS credit risk model:
1. Auto-detects dataset column format:
   - Polish Bankruptcy Format (Attr1-Attr64 or X1-X64)
   - Taiwanese Bankruptcy Format (95 named financial ratios like 'ROA(C)...')
   - Custom Format (any numeric financial columns — automatically mapped to feature space)
2. Trains or loads the calibrated XGBoost (60%) + LightGBM (40%) ensemble model on the
   UCI Polish Bankruptcy benchmark dataset (auto-downloads if not cached).
3. Applies median imputation (fit on training data, transform on user data).
4. Generates per-row distress risk scores (0.0 to 1.0) and early-warning alerts (ALERT vs SAFE).
5. Computes ground-truth accuracy, ROC-AUC, PR-AUC, and confusion matrix if a target column is found.
6. Exports predictions to a new CSV file and displays a terminal report with box-drawing UI.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Path Configuration & Root Setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe stdout UTF-8 encoding configuration for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import model & dataset dependencies from repository
from src.data.load_polish import load_all_polish_horizons
from src.models.train import EnsembleClassifier, LGBMClassifier, XGBClassifier

# Terminal formatting width
BOX_WIDTH = 80

# Known target column candidate names (case-insensitive)
TARGET_COLUMN_CANDIDATES = [
    "target",
    "bankrupt?",
    "bankrupt",
    "is_defaulted",
    "label",
    "class",
    "default",
    "y",
    "distress",
    "status",
]

# Signature features from Taiwanese Bankruptcy dataset (95 features)
DISTINCTIVE_TAIWANESE_FEATURES = [
    "ROA(C) before interest and depreciation before interest",
    "ROA(A) before interest and % after tax",
    "ROA(B) before interest and depreciation after tax",
    "Operating Gross Margin",
    "Realized Sales Gross Margin",
    "Operating Profit Rate",
    "Pre-tax net Interest Rate",
    "After-tax net Interest Rate",
    "Cash flow rate",
    "Persistent EPS in the Last Four Seasons",
    "Current Ratio",
    "Quick Ratio",
    "Total debt/Total net worth",
    "Debt ratio %",
    "Net worth/Assets",
    "Total Asset Turnover",
    "Accounts Receivable Turnover",
    "Average Collection Days",
    "Inventory Turnover Rate (times)",
    "Fixed Assets Turnover Frequency",
    "Working Capital to Total Assets",
    "Cash/Current Liability",
    "Current Liability to Assets",
    "Retained Earnings to Total Assets",
    "Cash Flow to Total Assets",
    "Net Income to Total Assets",
    "Gross Profit to Sales",
    "Liability to Equity",
    "Interest Coverage Ratio (Interest expense to EBIT)",
    "Equity to Liability",
]

# Semantic mapping: Taiwanese named features -> Polish Attr1-Attr64 features
TAIWAN_TO_POLISH_MAP = {
    "Net Income to Total Assets": "Attr1",
    "ROA(A) before interest and % after tax": "Attr1",
    "Debt ratio %": "Attr2",
    "Working Capital to Total Assets": "Attr3",
    "Current Ratio": "Attr4",
    "Retained Earnings to Total Assets": "Attr6",
    "ROA(C) before interest and depreciation before interest": "Attr7",
    "Equity to Liability": "Attr8",
    "Total Asset Turnover": "Attr9",
    "Net worth/Assets": "Attr10",
    "Cash flow rate": "Attr16",
    "Realized Sales Gross Margin": "Attr19",
    "Gross Profit to Sales": "Attr19",
    "Operating Profit Rate": "Attr22",
    "Continuous interest rate (after tax)": "Attr23",
    "After-tax net Interest Rate": "Attr23",
    "Interest Coverage Ratio (Interest expense to EBIT)": "Attr27",
    "Cash/Current Liability": "Attr40",
    "Average Collection Days": "Attr44",
    "Quick Ratio": "Attr46",
    "Current Liability to Assets": "Attr51",
    "Operating Gross Margin": "Attr56",
    "Liability to Equity": "Attr59",
    "Total debt/Total net worth": "Attr59",
    "Inventory Turnover Rate (times)": "Attr60",
    "Accounts Receivable Turnover": "Attr61",
    "Fixed Assets Turnover Frequency": "Attr64",
}

# Semantic aliases for common financial ratios in custom CSVs
CUSTOM_NAME_TO_POLISH = {
    "net_margin": "Attr1",
    "roa": "Attr1",
    "net_income_to_total_assets": "Attr1",
    "debt_ratio": "Attr2",
    "total_liabilities_to_total_assets": "Attr2",
    "working_capital_to_total_assets": "Attr3",
    "wc_ta": "Attr3",
    "current_ratio": "Attr4",
    "cr": "Attr4",
    "retained_earnings_to_total_assets": "Attr6",
    "re_ta": "Attr6",
    "ebit_to_total_assets": "Attr7",
    "ebit_ta": "Attr7",
    "equity_to_liability": "Attr8",
    "asset_turnover": "Attr9",
    "sales_to_total_assets": "Attr9",
    "sales_ta": "Attr9",
    "equity_to_total_assets": "Attr10",
    "operating_margin": "Attr22",
    "opm": "Attr22",
    "interest_coverage_ratio": "Attr27",
    "icr": "Attr27",
    "cash_to_current_liabilities": "Attr40",
    "quick_ratio": "Attr46",
    "qr": "Attr46",
    "short_term_debt_to_total_assets": "Attr51",
    "gross_margin": "Attr56",
    "debt_to_equity": "Attr59",
    "dte": "Attr59",
    "inventory_turnover": "Attr60",
    "receivables_turnover": "Attr61",
    "fixed_asset_turnover": "Attr64",
}

# ---------------------------------------------------------------------------
# Indian / Ind-AS EWS Feature Definitions & Multi-Alias Dictionaries
# ---------------------------------------------------------------------------
INDIAN_EWS_FEATURES = [
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "interest_coverage_ratio",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "receivables_growth_vs_sales_growth",
    "cash_flow_to_total_debt",
    "working_capital_to_total_assets",
    "retained_earnings_to_total_assets",
    "ebit_to_total_assets",
    "market_cap_to_total_liabilities",
    "z_score",
    "f_score",
    "m_score",
    "promoter_pledge_pct",
    "pledge_slope_4q",
    "related_party_transactions_to_revenue",
    "rpt_growth_yoy",
    "auditor_resignation_flag",
    "credit_rating_action_score",
    "filing_delay_days",
]

INDIAN_FEATURE_ALIASES = {
    "cr": "current_ratio",
    "current_ratio": "current_ratio",
    "qr": "quick_ratio",
    "quick_ratio": "quick_ratio",
    "dte": "debt_to_equity",
    "debt_to_equity": "debt_to_equity",
    "debt_equity": "debt_to_equity",
    "icr": "interest_coverage_ratio",
    "interest_coverage": "interest_coverage_ratio",
    "interest_coverage_ratio": "interest_coverage_ratio",
    "opm": "operating_margin",
    "operating_margin": "operating_margin",
    "npm": "net_margin",
    "net_margin": "net_margin",
    "roa": "roa",
    "roe": "roe",
    "ato": "sales_to_total_assets",
    "asset_turnover": "sales_to_total_assets",
    "rec_growth": "receivables_growth_vs_sales_growth",
    "receivables_growth_vs_sales_growth": "receivables_growth_vs_sales_growth",
    "cfo_debt": "cash_flow_to_total_debt",
    "cash_flow_to_total_debt": "cash_flow_to_total_debt",
    "wc_ta": "working_capital_to_total_assets",
    "working_capital_to_total_assets": "working_capital_to_total_assets",
    "re_ta": "retained_earnings_to_total_assets",
    "retained_earnings_to_total_assets": "retained_earnings_to_total_assets",
    "ebit_ta": "ebit_to_total_assets",
    "ebit_to_total_assets": "ebit_to_total_assets",
    "sales_ta": "sales_to_total_assets",
    "mcap_tl": "market_cap_to_total_liabilities",
    "market_cap_to_total_liabilities": "market_cap_to_total_liabilities",
    "z_score": "z_score",
    "f_score": "f_score",
    "m_score": "m_score",
    "pledge": "promoter_pledge_pct",
    "promoter_pledge_pct": "promoter_pledge_pct",
    "pledge_slope": "pledge_slope_4q",
    "pledge_slope_4q": "pledge_slope_4q",
    "rpt_rev": "related_party_transactions_to_revenue",
    "related_party_transactions_to_revenue": "related_party_transactions_to_revenue",
    "rpt_growth": "rpt_growth_yoy",
    "rpt_growth_yoy": "rpt_growth_yoy",
    "auditor_resig": "auditor_resignation_flag",
    "auditor_resignation_flag": "auditor_resignation_flag",
    "rating_score": "credit_rating_action_score",
    "credit_rating_action_score": "credit_rating_action_score",
    "delay": "filing_delay_days",
    "filing_delay_days": "filing_delay_days",
}


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    """Reads input dataset across diverse file extensions (CSV, Excel, TSV, Parquet) and delimiters."""
    suffix = input_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(input_path)
    elif suffix in [".parquet"]:
        return pd.read_parquet(input_path)
    elif suffix in [".tsv"]:
        return pd.read_csv(input_path, sep="\t")
    else:
        # Standard CSV, auto-detect delimiter and encoding
        try:
            return pd.read_csv(input_path)
        except Exception:
            try:
                return pd.read_csv(input_path, sep=None, engine="python", encoding="utf-8")
            except Exception:
                return pd.read_csv(input_path, sep=None, engine="python", encoding="latin1")


# ---------------------------------------------------------------------------
# UI & Box-Drawing Utilities
# ---------------------------------------------------------------------------

def print_banner() -> None:
    """Prints the application banner with double-lined box styling."""
    title1 = "EARLY-WARNING CREDIT RISK SYSTEM (EWS) — PREDICTOR"
    title2 = "XGBoost (60%) + LightGBM (40%) Ensemble Model Engine"
    w = BOX_WIDTH
    print("\n" + "╔" + "═" * (w - 2) + "╗")
    print("║" + f"{title1:^{w - 2}}" + "║")
    print("║" + f"{title2:^{w - 2}}" + "║")
    print("╚" + "═" * (w - 2) + "╝")


def print_error_box(title: str, message: str) -> None:
    """Prints an error message framed in a box."""
    w = BOX_WIDTH
    print("\n" + "┌" + "─" * (w - 2) + "┐")
    print("│" + f" ERROR: {title.upper():<{w - 10}}" + "│")
    print("├" + "─" * (w - 2) + "┤")
    # Wrap message if long
    import textwrap
    wrapped = textwrap.wrap(message, width=w - 6)
    for line in wrapped:
        print("│" + f"  {line:<{w - 4}}" + "│")
    print("└" + "─" * (w - 2) + "┘\n")


def print_section_box(title: str, rows: List[Tuple[str, str]]) -> None:
    """Prints a styled section card with key-value entries."""
    w = BOX_WIDTH
    print("\n┌" + "─" * (w - 2) + "┐")
    print("│" + f" {title.upper():<{w - 3}}" + "│")
    print("├" + "─" * (w - 2) + "┤")
    for key, val in rows:
        line_content = f"  {key:<24} : {val}"
        if len(line_content) > w - 4:
            line_content = line_content[: w - 7] + "..."
        print("│" + f"{line_content:<{w - 2}}" + "│")
    print("└" + "─" * (w - 2) + "┘")


# ---------------------------------------------------------------------------
# Column & Target Detection Functions
# ---------------------------------------------------------------------------

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """Detects target/distress label column name (case-insensitive)."""
    # 1. Exact match on clean string
    for col in df.columns:
        clean = str(col).strip().lower()
        if clean in TARGET_COLUMN_CANDIDATES:
            return col

    # 2. Substring match
    for col in df.columns:
        clean = str(col).strip().lower()
        if "bankrupt" in clean or "default" in clean or clean.endswith("_target"):
            return col

    return None


def parse_target_series(series: pd.Series) -> pd.Series:
    """Standardizes heterogeneous target column representations into binary integers (0 or 1)."""
    s_clean = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("'", "")
        .str.replace('"', "")
    )
    mapping = {
        "0": 0,
        "0.0": 0,
        "false": 0,
        "active": 0,
        "solvent": 0,
        "healthy": 0,
        "safe": 0,
        "no": 0,
        "1": 1,
        "1.0": 1,
        "true": 1,
        "bankrupt": 1,
        "distressed": 1,
        "default": 1,
        "defaulted": 1,
        "alert": 1,
        "yes": 1,
    }
    mapped = s_clean.map(mapping)
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.fillna(0).astype(int)
    return mapped.astype(int)


def detect_identifier_column(df: pd.DataFrame, target_col: Optional[str]) -> Optional[str]:
    """Finds firm identifier column (e.g. company name, symbol, ticker) for readable display."""
    id_candidates = ["company", "company_name", "firm", "borrower", "name", "ticker", "id", "cin", "symbol"]
    for col in df.columns:
        if col == target_col:
            continue
        clean = str(col).strip().lower()
        if clean in id_candidates:
            return col

    for col in df.columns:
        if col == target_col:
            continue
        clean = str(col).strip().lower()
        if ("name" in clean or "company" in clean) and df[col].dtype == object:
            return col

    return None


# ---------------------------------------------------------------------------
# Format Detection & Feature Preparation
# ---------------------------------------------------------------------------

def detect_and_prepare_features(
    df: pd.DataFrame,
    target_col: Optional[str],
    preferred_model: str = "auto",
) -> Tuple[str, pd.DataFrame, List[str], str, str]:
    """Detects dataset format and maps input features into model feature space.

    Returns:
        format_name: Human-readable name of the detected format.
        X_df: Clean DataFrame containing mapped feature columns.
        matched_feature_cols: Names of original features utilized.
        format_description: Narrative details about the mapping applied.
        model_family: Model family to run ('polish' or 'india').
    """
    feature_candidates = [c for c in df.columns if c != target_col]
    clean_feature_map = {str(c).strip(): c for c in feature_candidates}
    stripped_cols = list(clean_feature_map.keys())
    stripped_lower_map = {str(c).strip().lower(): c for c in feature_candidates}

    polish_attr = [c for c in stripped_cols if c.startswith("Attr") and c[4:].isdigit()]
    polish_x = [c for c in stripped_cols if c.startswith("X") and c[1:].isdigit()]

    # 1. Polish Format Check (Attr1-Attr64 or X1-X64)
    if (len(polish_attr) >= 5 or (len(stripped_cols) > 0 and len(polish_attr) / len(stripped_cols) > 0.3)) and preferred_model != "india":
        format_name = "Polish Bankruptcy Benchmark Format (Attr1-Attr64)"
        desc = f"Identified {len(polish_attr)} Polish Attr features matching model feature space."
        X_df = pd.DataFrame(index=df.index)
        for i in range(1, 65):
            col_name = f"Attr{i}"
            if col_name in clean_feature_map:
                orig_col = clean_feature_map[col_name]
                X_df[col_name] = pd.to_numeric(df[orig_col], errors="coerce")
            else:
                X_df[col_name] = np.nan
        return format_name, X_df, polish_attr, desc, "polish"

    if (len(polish_x) >= 5 or (len(stripped_cols) > 0 and len(polish_x) / len(stripped_cols) > 0.3)) and preferred_model != "india":
        format_name = "Polish Bankruptcy Benchmark Format (X1-X64 notation)"
        desc = f"Identified {len(polish_x)} Polish X features; mapped directly to Attr1-Attr64."
        X_df = pd.DataFrame(index=df.index)
        for i in range(1, 65):
            x_col = f"X{i}"
            attr_col = f"Attr{i}"
            if x_col in clean_feature_map:
                orig_col = clean_feature_map[x_col]
                X_df[attr_col] = pd.to_numeric(df[orig_col], errors="coerce")
            else:
                X_df[attr_col] = np.nan
        return format_name, X_df, polish_x, desc, "polish"

    # 2. Indian Corporate & Forensic EWS Format Check
    indian_matches = []
    for sc in stripped_cols:
        norm = sc.strip().lower()
        if norm in INDIAN_FEATURE_ALIASES:
            indian_matches.append(sc)

    has_forensic = any(
        k in stripped_lower_map
        for k in ["z_score", "m_score", "f_score", "pledge", "promoter_pledge_pct", "auditor_resig", "auditor_resignation_flag", "delay", "filing_delay_days"]
    )
    is_indian = (preferred_model == "india") or (
        len(indian_matches) >= 3 and (has_forensic or len(indian_matches) >= 6)
    )

    if is_indian and preferred_model != "polish":
        format_name = "Indian Corporate & Forensic EWS Format (24 Key Financial & Governance Indicators)"
        desc = f"Identified {len(indian_matches)} corporate financial & forensic indicators; routed to Indian EWS model."
        X_df = pd.DataFrame(index=df.index)
        for target_feat in INDIAN_EWS_FEATURES:
            X_df[target_feat] = np.nan

        assigned_indian = set()
        for sc in stripped_cols:
            norm = sc.strip().lower()
            if norm in INDIAN_FEATURE_ALIASES:
                target_feat = INDIAN_FEATURE_ALIASES[norm]
                if target_feat not in assigned_indian:
                    orig_col = clean_feature_map[sc]
                    X_df[target_feat] = pd.to_numeric(df[orig_col], errors="coerce")
                    assigned_indian.add(target_feat)

        return format_name, X_df, indian_matches, desc, "india"

    # 3. Taiwanese Bankruptcy Format Check (95 named features)
    taiwan_lower_map = {c.lower(): c for c in DISTINCTIVE_TAIWANESE_FEATURES}
    taiwan_matches = [c for c in stripped_cols if c.lower() in taiwan_lower_map]

    if (len(taiwan_matches) >= 3 or (
        any("roa(c)" in c.lower() for c in stripped_cols)
        and any("net worth/assets" in c.lower() for c in stripped_cols)
    )) and preferred_model != "india":
        format_name = "Taiwanese Bankruptcy Format (95 Named Financial Ratios)"
        desc = f"Detected {len(taiwan_matches)} Taiwanese financial metrics; semantically mapped to Attr1-Attr64."
        X_df = pd.DataFrame(index=df.index)
        for i in range(1, 65):
            X_df[f"Attr{i}"] = np.nan

        for taiwan_col, attr_col in TAIWAN_TO_POLISH_MAP.items():
            for sc in stripped_cols:
                if sc.lower() == taiwan_col.lower():
                    orig_col = clean_feature_map[sc]
                    X_df[attr_col] = pd.to_numeric(df[orig_col], errors="coerce")
                    break

        return format_name, X_df, taiwan_matches, desc, "polish"

    # 4. Custom Format (Any numeric columns)
    format_name = "Custom Corporate Financial Format (Numeric Feature Alignment)"
    numeric_cols = []
    for c in feature_candidates:
        num_s = pd.to_numeric(df[c], errors="coerce")
        if num_s.notna().sum() > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(
            f"No numeric columns could be identified among the {len(feature_candidates)} features in the CSV."
        )

    desc = f"Extracted {len(numeric_cols)} numeric columns; aligned into 64-ratio model space."
    X_df = pd.DataFrame(index=df.index)
    for i in range(1, 65):
        X_df[f"Attr{i}"] = np.nan

    assigned_attrs = set()
    used_numeric_cols = []

    for num_col in numeric_cols:
        clean_key = str(num_col).strip().lower().replace(" ", "_").replace("-", "_")
        if clean_key in CUSTOM_NAME_TO_POLISH:
            target_attr = CUSTOM_NAME_TO_POLISH[clean_key]
            if target_attr not in assigned_attrs:
                X_df[target_attr] = pd.to_numeric(df[num_col], errors="coerce")
                assigned_attrs.add(target_attr)
                used_numeric_cols.append(num_col)

    attr_idx = 1
    for num_col in numeric_cols:
        if num_col in used_numeric_cols:
            continue
        while attr_idx <= 64 and f"Attr{attr_idx}" in assigned_attrs:
            attr_idx += 1
        if attr_idx <= 64:
            target_attr = f"Attr{attr_idx}"
            X_df[target_attr] = pd.to_numeric(df[num_col], errors="coerce")
            assigned_attrs.add(target_attr)
            used_numeric_cols.append(num_col)
            attr_idx += 1
        else:
            break

    return format_name, X_df, used_numeric_cols, desc, "polish"


# ---------------------------------------------------------------------------
# Model Training & Caching Engine
# ---------------------------------------------------------------------------

def load_or_train_ensemble(
    force_retrain: bool = False,
    cache_path: Optional[Path] = None,
    quiet: bool = False,
) -> Tuple[EnsembleClassifier, SimpleImputer, float]:
    """Loads cached ensemble model or trains fresh XGBoost + LightGBM on UCI Polish dataset.

    Parameters:
        force_retrain: If True, ignores existing cache and refits from scratch.
        cache_path: File path to save/load model cache joblib.
        quiet: If True, suppresses non-critical progress logging.

    Returns:
        ensemble: Fitted EnsembleClassifier combining XGBoost (60%) and LightGBM (40%).
        imputer: Fitted SimpleImputer (median strategy on Polish training set).
        scale_pos_weight: Class-imbalance weight ratio.
    """
    if cache_path is None:
        cache_path = REPO_ROOT / "data" / "processed" / "polish_ensemble_model.joblib"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt to load from disk cache
    if cache_path.exists() and not force_retrain:
        try:
            cached_pkg = joblib.load(cache_path)
            if (
                isinstance(cached_pkg, dict)
                and "model" in cached_pkg
                and "imputer" in cached_pkg
            ):
                if not quiet:
                    print(f"  [+] Loaded cached pre-trained ensemble model: {cache_path.name}")
                spw = cached_pkg.get("scale_pos_weight", 20.73)
                return cached_pkg["model"], cached_pkg["imputer"], spw
        except Exception as e:
            if not quiet:
                print(f"  [!] Cache load failed ({e}), training fresh model...")

    if not quiet:
        print("\n[*] Training Benchmark EWS Ensemble on UCI Polish Companies Dataset...")
        print("    (Downloading ARFF files from UCI repository if not already cached)")

    # Load all 5 horizons of the Polish bankruptcy dataset
    df_polish, _ = load_all_polish_horizons()
    features = [f"Attr{i}" for i in range(1, 65)]

    X_train_raw = df_polish[features].copy()
    y_train = df_polish["target"].values.astype(int)

    # Calculate class imbalance weighting
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(max(np.sum(y_train == 1), 1))
    scale_pos_weight = float(neg_count) / float(pos_count)

    if not quiet:
        print(f"    • Total Training Records: {len(df_polish):,} company-years across 5 horizons")
        print(f"    • Bankrupt Instances:     {pos_count:,} ({pos_count / len(df_polish) * 100:.2f}%)")
        print(f"    • Imbalance Compensation: scale_pos_weight = {scale_pos_weight:.2f}")
        print("    • Preprocessing:          Fitting Robust Median Imputer...")

    # Fit median imputer on training set only (prevent leakage)
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)

    if not quiet:
        print("    • Training Component 1:   XGBoost (500 trees, depth 6, lr 0.05, colsample 0.8)...")

    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_train_imp, y_train)

    if not quiet:
        print("    • Training Component 2:   LightGBM (500 trees, depth 6, lr 0.05, colsample 0.8)...")

    lgb_model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train_imp, y_train)

    if not quiet:
        print("    • Assembling Ensemble:    60% XGBoost + 40% LightGBM...")

    ensemble = EnsembleClassifier(xgb_model, lgb_model, w_xgb=0.60, w_lgb=0.40)

    # Cache fitted models and imputer to disk
    try:
        joblib.dump(
            {
                "model": ensemble,
                "imputer": imputer,
                "scale_pos_weight": scale_pos_weight,
                "features": features,
            },
            cache_path,
        )
        if not quiet:
            print(f"  [+] Ensemble model successfully serialized to: {cache_path}")
    except Exception as e:
        if not quiet:
            print(f"  [!] Notice: Could not save model cache ({e})")

    return ensemble, imputer, scale_pos_weight


def load_or_train_india_model(
    force_retrain: bool = False,
    cache_path: Optional[Path] = None,
    quiet: bool = False,
) -> Tuple[Any, SimpleImputer, float]:
    """Loads cached Indian EWS model or trains LightGBM on the curated Indian distress panel."""
    if cache_path is None:
        cache_path = REPO_ROOT / "data" / "processed" / "india_ews_model.joblib"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_retrain:
        try:
            cached = joblib.load(cache_path)
            if isinstance(cached, dict) and "model" in cached and "imputer" in cached:
                if not quiet:
                    print(f"  [+] Loaded cached Indian Corporate EWS model: {cache_path.name}")
                return cached["model"], cached["imputer"], cached.get("threshold", 0.35)
        except Exception as e:
            if not quiet:
                print(f"  [!] Indian model cache load failed ({e}), training fresh...")

    if not quiet:
        print("\n[*] Training Indian Corporate Early-Warning System (EWS) Model...")

    from src.data.load_india_panel import load_india_panel
    df_india = load_india_panel(enforce_leakage_assertion=True, enrich_scores=True)
    features = INDIAN_EWS_FEATURES

    X_train_raw = df_india[features].copy()
    y_train = df_india["is_defaulted"].values.astype(int)

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(max(np.sum(y_train == 1), 1))
    scale_pos_weight = float(neg_count) / float(pos_count)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)

    model = LGBMClassifier(
        objective="binary",
        max_depth=4,
        num_leaves=12,
        min_child_samples=3,
        learning_rate=0.03,
        n_estimators=150,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train_imp, y_train)

    try:
        joblib.dump(
            {
                "model": model,
                "imputer": imputer,
                "threshold": 0.35,
                "features": features,
            },
            cache_path,
        )
        if not quiet:
            print(f"  [+] Indian EWS model successfully serialized to: {cache_path}")
    except Exception as e:
        if not quiet:
            print(f"  [!] Notice: Could not save Indian model cache ({e})")

    return model, imputer, 0.35


# ---------------------------------------------------------------------------
# Evaluation & Metric Computation
# ---------------------------------------------------------------------------

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Computes standard credit risk and supervisory alert evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) > 1

    if has_both_classes:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = None

        try:
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(rec_arr, prec_arr)
        except Exception:
            pr_auc = None

        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    else:
        roc_auc = None
        pr_auc = None
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    n_alerts = int(np.sum(y_pred == 1))

    return {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_alerts": n_alerts,
        "alert_rate": (n_alerts / n_total * 100.0) if n_total > 0 else 0.0,
        "has_both_classes": has_both_classes,
    }


def print_predictions_table(
    df: pd.DataFrame,
    id_col: Optional[str],
    target_col: Optional[str],
    max_preview: int = 10,
) -> None:
    """Renders a clean tabular preview of scored instances using box-drawing characters."""
    w = BOX_WIDTH
    print("\n┌" + "─" * (w - 2) + "┐")
    preview_title = f" PREDICTIONS PREVIEW (Top {min(len(df), max_preview)} of {len(df)} Rows)"
    print("│" + f"{preview_title:<{w - 2}}" + "│")
    print("├" + "─" * (w - 2) + "┤")

    has_target = target_col is not None

    if has_target:
        header = f"│ {'#':<3} │ {'Company / Identifier':<24} │ {'Risk Score':<10} │ {'Predicted':<9} │ {'Actual':<8} │ {'Status':<11} │"
        sep    = f"├─────┼──────────────────────────┼────────────┼───────────┼──────────┼─────────────┤"
    else:
        header = f"│ {'#':<3} │ {'Company / Identifier':<38} │ {'Risk Score':<14} │ {'Classification':<12} │"
        sep    = f"├─────┼────────────────────────────────────────┼────────────────┼──────────────┤"

    print(header)
    print(sep)

    preview_df = df.head(max_preview)
    for idx, (_, row) in enumerate(preview_df.iterrows(), 1):
        if id_col and id_col in row and pd.notna(row[id_col]):
            name = str(row[id_col])
        else:
            name = f"Company #{idx}"

        if has_target:
            name_trunc = name[:24]
            r_score = f"{float(row['risk_score']):.4f}"
            pred_lbl = str(row["alert_classification"])
            actual_val = int(row["ground_truth"])
            act_lbl = "Distress" if actual_val == 1 else "Healthy"

            is_correct = bool(row["is_correct"])
            if is_correct:
                status_lbl = "CORRECT"
            elif pred_lbl == "ALERT" and actual_val == 0:
                status_lbl = "FALSE ALARM"
            else:
                status_lbl = "MISSED DEF"

            row_line = (
                f"│ {idx:<3} │ {name_trunc:<24} │ {r_score:<10} │ {pred_lbl:<9} │ {act_lbl:<8} │ {status_lbl:<11} │"
            )
        else:
            name_trunc = name[:38]
            r_score = f"{float(row['risk_score']):.4f}"
            pred_lbl = str(row["alert_classification"])
            row_line = f"│ {idx:<3} │ {name_trunc:<38} │ {r_score:<14} │ {pred_lbl:<12} │"

        print(row_line)

    if len(df) > max_preview:
        remaining = len(df) - max_preview
        print(f"│  ... and {remaining:,} additional rows exported to destination CSV file. ...{' ' * (w - 75)}│")

    print("└" + "─" * (w - 2) + "┘")


# ---------------------------------------------------------------------------
# Main Orchestration Routine
# ---------------------------------------------------------------------------

def run_prediction(
    input_csv_path: str,
    threshold: Optional[float] = None,
    preferred_model: str = "auto",
    output_path: Optional[str] = None,
    force_retrain: bool = False,
    max_preview: int = 10,
    quiet: bool = False,
) -> pd.DataFrame:
    """Executes the full early-warning prediction pipeline on the user's input dataset."""
    t_start = time.time()
    print_banner()

    input_file = Path(input_csv_path).resolve()
    if not input_file.exists():
        print_error_box(
            "File Not Found",
            f"The input dataset file does not exist:\n'{input_file}'\nPlease check the path and try again.",
        )
        sys.exit(1)

    # 1. Read input dataset across formats (CSV, Excel, TSV, Parquet)
    try:
        user_df = load_input_dataframe(input_file)
    except Exception as e:
        print_error_box("Dataset Read Failure", f"Unable to parse '{input_file.name}': {e}")
        sys.exit(1)

    if len(user_df) == 0:
        print_error_box("Empty Dataset", f"The file '{input_file.name}' contains 0 rows.")
        sys.exit(1)

    # Replace infinities with NaN
    user_df = user_df.replace([np.inf, -np.inf], np.nan)

    # 2. Target Column Detection
    target_col = detect_target_column(user_df)
    id_col = detect_identifier_column(user_df, target_col)

    # 3. Format Detection & Feature Preparation
    try:
        format_name, X_features, used_cols, format_desc, model_family = detect_and_prepare_features(
            user_df, target_col, preferred_model=preferred_model
        )
    except Exception as e:
        print_error_box("Feature Extraction Error", str(e))
        sys.exit(1)

    # Determine default alert threshold if not explicitly overridden by user
    if threshold is None:
        threshold = 0.35 if model_family == "india" else 0.05

    # 4. Load or Train Model Engine
    if model_family == "india":
        model, imputer, default_th = load_or_train_india_model(
            force_retrain=force_retrain,
            quiet=quiet,
        )
        model_name = "Indian Corporate & Forensic EWS Model (LightGBM 24-Metric Engine)"
        active_features = INDIAN_EWS_FEATURES
        features_mapped_desc = f"{len(used_cols)} input columns -> 24 Indian EWS ratios"
    else:
        ensemble, imputer, spw = load_or_train_ensemble(
            force_retrain=force_retrain,
            quiet=quiet,
        )
        model = ensemble
        model_name = "Polish Benchmark Ensemble (60% XGBoost + 40% LightGBM Engine)"
        active_features = [f"Attr{i}" for i in range(1, 65)]
        features_mapped_desc = f"{len(used_cols)} input columns -> 64 Attr ratios"

    # Section 1: Dataset Inspection
    target_display = f"'{target_col}' (Ground truth will be evaluated)" if target_col else "None (Unlabeled test data)"
    inspection_rows = [
        ("Input File", input_file.name),
        ("Total Records", f"{len(user_df):,} rows"),
        ("Detected Format", format_name),
        ("Features Mapped", features_mapped_desc),
        ("Active Model Engine", model_name),
        ("Target Column", target_display),
        ("Alert Threshold", f"{threshold:.3f} (Early-warning alert cutoff)"),
    ]
    print_section_box("Dataset Inspection & Format Auto-Detection", inspection_rows)

    # 5. Apply Imputation & Prediction
    X_imputed = imputer.transform(X_features[active_features])

    raw_probs = model.predict_proba(X_imputed)[:, 1]
    risk_scores = np.clip(raw_probs, 0.0, 1.0)
    binary_alerts = (risk_scores >= threshold).astype(int)
    alert_labels = ["ALERT" if a == 1 else "SAFE" for a in binary_alerts]

    # Assemble output DataFrame
    output_df = user_df.copy()
    output_df["risk_score"] = np.round(risk_scores, 4)
    output_df["alert_classification"] = alert_labels
    output_df["alert_threshold"] = threshold

    y_true = None
    if target_col is not None:
        y_true = parse_target_series(user_df[target_col]).values
        output_df["ground_truth"] = y_true
        output_df["is_correct"] = (binary_alerts == y_true)

    # 6. Render Terminal Preview Table
    print_predictions_table(output_df, id_col, target_col, max_preview=max_preview)

    # 7. Summary Metrics
    if target_col is not None and y_true is not None:
        metrics = compute_evaluation_metrics(y_true, risk_scores, threshold)
        metric_rows = [
            ("Accuracy", f"{metrics['accuracy'] * 100.0:.2f}%"),
            (
                "ROC-AUC",
                f"{metrics['roc_auc'] * 100.0:.2f}%" if metrics["roc_auc"] is not None else "N/A (Single class)",
            ),
            (
                "PR-AUC",
                f"{metrics['pr_auc'] * 100.0:.2f}%" if metrics["pr_auc"] is not None else "N/A (Single class)",
            ),
            ("Precision", f"{metrics['precision'] * 100.0:.2f}%"),
            (
                "Recall (Distress Caught)",
                f"{metrics['recall'] * 100.0:.2f}% ({metrics['tp']} of {metrics['n_pos']} true defaults)",
            ),
            (
                "False Alarm Rate",
                f"{(metrics['fp'] / max(metrics['n_neg'], 1)) * 100.0:.2f}% ({metrics['fp']} of {metrics['n_neg']} healthy firms)",
            ),
            ("F1-Score", f"{metrics['f1'] * 100.0:.2f}%"),
            (
                "Confusion Matrix",
                f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}",
            ),
            ("Alert Volume", f"{metrics['n_alerts']:,} / {metrics['n_total']:,} rows ({metrics['alert_rate']:.1f}%)"),
        ]
        print_section_box("Model Evaluation Summary (Ground-Truth Verified)", metric_rows)
    else:
        n_alerts = int(np.sum(binary_alerts == 1))
        n_safe = len(user_df) - n_alerts
        summary_rows = [
            ("Total Companies Scored", f"{len(user_df):,}"),
            ("Alert Threshold", f"{threshold:.3f}"),
            ("Triggered ALERTS", f"{n_alerts:,} ({n_alerts / len(user_df) * 100.0:.1f}%) -> Escalated for Review"),
            ("Classified SAFE", f"{n_safe:,} ({n_safe / len(user_df) * 100.0:.1f}%)"),
            ("Mean Risk Score", f"{float(np.mean(risk_scores)):.4f}"),
            ("Median Risk Score", f"{float(np.median(risk_scores)):.4f}"),
            ("Max Risk Score", f"{float(np.max(risk_scores)):.4f}"),
            ("Min Risk Score", f"{float(np.min(risk_scores)):.4f}"),
        ]
        print_section_box("Portfolio Risk Summary (Unlabeled Test Set)", summary_rows)

    # 8. Export Output CSV
    if output_path:
        out_file = Path(output_path).resolve()
    else:
        out_file = input_file.parent / f"{input_file.stem}_predictions.csv"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_file, index=False)

    elapsed = time.time() - t_start
    save_rows = [
        ("Destination Path", str(out_file)),
        ("Total Rows Exported", f"{len(output_df):,}"),
        ("New Columns Added", "risk_score, alert_classification, alert_threshold" + (", ground_truth, is_correct" if target_col else "")),
        ("Execution Time", f"{elapsed:.2f} seconds"),
    ]
    print_section_box("Prediction Results Exported", save_rows)

    print("\n[+] Prediction pipeline completed successfully.\n")
    return output_df


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Builds user-friendly command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "🛡️ EARLY-WARNING SYSTEM (EWS) FOR CORPORATE CREDIT RISK — PREDICTOR\n\n"
            "Evaluates financial datasets for insolvency risk using an ensemble\n"
            "of XGBoost (60%) and LightGBM (40%) trained on corporate financial ratio benchmarks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # 1. Run predictions on sample dummy data
  python predict.py sample_input.csv

  # 2. Specify a custom alert threshold (e.g., 10% risk threshold)
  python predict.py sample_input.csv --threshold 0.10

  # 3. Specify an explicit output CSV destination
  python predict.py sample_input.csv --output custom_predictions.csv

  # 4. Evaluate on the 6,819-row UCI Taiwanese Bankruptcy dataset
  python predict.py data/raw/taiwanese/data.csv --output reports/taiwan_preds.csv

  # 5. Force re-training the benchmark model from source data
  python predict.py sample_input.csv --retrain

Supported Input Formats:
  • Polish Bankruptcy Format: 64 financial ratios named Attr1-Attr64 or X1-X64
  • Taiwanese Bankruptcy Format: 95 named financial ratios (e.g. 'ROA(C)...')
  • Custom Corporate Format: Any CSV with numeric accounting / financial metrics
""",
    )

    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file containing company financial ratio data.",
    )

    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help=(
            "Early-warning alert threshold between 0.0 and 1.0. "
            "Defaults to optimal calibration: 0.35 for Indian Corporate EWS, 0.05 for Polish Benchmark. "
            "Borrowers with risk scores >= threshold are classified as ALERT."
        ),
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=["auto", "polish", "india"],
        default="auto",
        help="Model engine to apply: 'auto' (intelligent auto-detection), 'india' (Indian EWS 24-metric model), or 'polish' (benchmark ensemble).",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Custom path to save the output CSV (default: <input_stem>_predictions.csv).",
    )

    parser.add_argument(
        "--retrain",
        "-r",
        action="store_true",
        help="Force re-training of the XGBoost + LightGBM ensemble on the Polish dataset instead of using cache.",
    )

    parser.add_argument(
        "--max-preview",
        type=int,
        default=10,
        help="Maximum number of rows to display in the terminal preview table (default: 10).",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress model training progress and non-critical logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.threshold is not None and (args.threshold <= 0.0 or args.threshold >= 1.0):
        print_error_box(
            "Invalid Threshold",
            f"Threshold must be strictly between 0.0 and 1.0 (received: {args.threshold}).",
        )
        sys.exit(1)

    run_prediction(
        input_csv_path=args.input_csv,
        threshold=args.threshold,
        preferred_model=args.model,
        output_path=args.output,
        force_retrain=args.retrain,
        max_preview=args.max_preview,
        quiet=args.quiet,
    )

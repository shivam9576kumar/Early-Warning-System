"""Financial ratio definitions, calculators, winsorization, and cross-dataset mappings.

Standard Ratios:
1. current_ratio: Current Assets / Current Liabilities
2. quick_ratio: (Cash + Short-term Investments + Receivables) / Current Liabilities
3. debt_to_equity: Total Debt / Shareholder Equity
4. interest_coverage_ratio: EBIT / Interest Expense
5. operating_margin: EBIT / Revenue
6. net_margin: Net Income / Revenue
7. roa: Net Income / Total Assets
8. roe: Net Income / Shareholder Equity
9. asset_turnover: Revenue / Total Assets
10. receivables_growth_vs_sales_growth: (Receivables_t / Receivables_{t-1}) / (Sales_t / Sales_{t-1})
11. cash_flow_to_total_debt: Cash Flow from Operations / Total Debt
12. working_capital_to_total_assets: (Current Assets - Current Liabilities) / Total Assets
13. retained_earnings_to_total_assets: Retained Earnings / Total Assets
14. ebit_to_total_assets: EBIT / Total Assets
15. sales_to_total_assets: Sales / Total Assets
16. market_cap_to_total_liabilities: Market Value of Equity / Total Liabilities
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Canonical feature list
CORE_RATIO_COLS = [
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    "interest_coverage_ratio",
    "operating_margin",
    "net_margin",
    "roa",
    "roe",
    "asset_turnover",
    "receivables_growth_vs_sales_growth",
    "cash_flow_to_total_debt",
    "working_capital_to_total_assets",
    "retained_earnings_to_total_assets",
    "ebit_to_total_assets",
    "sales_to_total_assets",
    "market_cap_to_total_liabilities",
]

# Mapping Polish dataset attributes to financial concepts
POLISH_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "Attr1": "Net profit / total assets",
    "Attr2": "Total liabilities / total assets",
    "Attr3": "Working capital / total assets",
    "Attr4": "Current assets / short-term liabilities",
    "Attr5": "[(Cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
    "Attr6": "Retained earnings / total assets",
    "Attr7": "EBIT / total assets",
    "Attr8": "Book value of equity / total liabilities",
    "Attr9": "Sales / total assets",
    "Attr10": "Equity / total assets",
    "Attr11": "(Gross profit + extraordinary items + financial expenses) / total assets",
    "Attr12": "Gross profit / short-term liabilities",
    "Attr13": "(Gross profit + depreciation) / sales",
    "Attr14": "(Gross profit + interest) / total assets",
    "Attr15": "(Total liabilities * 365) / (gross profit + depreciation)",
    "Attr16": "(Gross profit + depreciation) / total liabilities",
    "Attr17": "Total assets / short-term liabilities",
    "Attr18": "Gross profit / total assets",
    "Attr19": "Gross profit / sales",
    "Attr20": "(Inventory * 365) / sales",
    "Attr21": "Sales (n) / sales (n-1)",
    "Attr22": "Operating profit / total assets",
    "Attr23": "Net profit / sales",
    "Attr24": "Gross profit (in 3 years) / total assets",
    "Attr25": "(Equity - share capital) / total assets",
    "Attr26": "(Net profit + depreciation) / total liabilities",
    "Attr27": "Profit on operating activities / financial expenses",
    "Attr28": "Working capital / fixed assets",
    "Attr29": "Logarithm of total assets",
    "Attr30": "(Total liabilities - cash) / sales",
    "Attr31": "(Gross profit + interest) / sales",
    "Attr32": "(Current liabilities * 365) / cost of products sold",
    "Attr33": "Operating expenses / short-term liabilities",
    "Attr34": "Operating expenses / total liabilities",
    "Attr35": "Profit on sales / total assets",
    "Attr36": "Total sales / total assets",
    "Attr37": "(Current assets - inventories) / long-term liabilities",
    "Attr38": "Constant capital / total assets",
    "Attr39": "Profit on sales / sales",
    "Attr40": "(Current assets - inventory - receivables) / short-term liabilities",
    "Attr41": "Total liabilities / ((profit on operating activities + depreciation) * (12/365))",
    "Attr42": "Profit on operating activities / sales",
    "Attr43": "Rotation receivables + inventory turnover in days",
    "Attr44": "(Receivables * 365) / sales",
    "Attr45": "Net profit / inventory",
    "Attr46": "(Current assets - inventory) / short-term liabilities",
    "Attr47": "(Inventory * 365) / cost of products sold",
    "Attr48": "EBITDA / total assets",
    "Attr49": "EBITDA / sales",
    "Attr50": "Current assets / total liabilities",
    "Attr51": "Short-term liabilities / total assets",
    "Attr52": "(Short-term liabilities * 365) / cost of products sold",
    "Attr53": "Equity / fixed assets",
    "Attr54": "Constant capital / fixed assets",
    "Attr55": "Working capital",
    "Attr56": "(Sales - cost of products sold) / sales",
    "Attr57": "(Current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    "Attr58": "Total costs / total sales",
    "Attr59": "Long-term liabilities / equity",
    "Attr60": "Sales / inventory",
    "Attr61": "Sales / receivables",
    "Attr62": "(Short-term liabilities * 365) / sales",
    "Attr63": "Sales / short-term liabilities",
    "Attr64": "Sales / fixed assets",
}

# Add X1..X64 aliases to POLISH_FEATURE_DESCRIPTIONS
for i in range(1, 65):
    POLISH_FEATURE_DESCRIPTIONS[f"X{i}"] = POLISH_FEATURE_DESCRIPTIONS[f"Attr{i}"]


def map_attr_to_x(name: str) -> str:
    """Converts Attr{i} to X{i}."""
    if name.startswith("Attr") and name[4:].isdigit():
        return f"X{name[4:]}"
    return name


def map_x_to_attr(name: str) -> str:
    """Converts X{i} to Attr{i}."""
    if name.startswith("X") and name[1:].isdigit():
        return f"Attr{name[1:]}"
    return name


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Clips extreme outliers beyond lower and upper quantiles."""
    valid = s.dropna()
    if len(valid) == 0:
        return s
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)


def normalize_within_sector(df: pd.DataFrame, feature_cols: List[str], sector_col: str = "sector") -> pd.DataFrame:
    """Standardizes features (z-score) within each industry sector to remove sector bias."""
    res = df.copy()
    if sector_col not in res.columns:
        return res

    for col in feature_cols:
        if col in res.columns and pd.api.types.is_numeric_dtype(res[col]):
            means = res.groupby(sector_col)[col].transform("mean")
            stds = res.groupby(sector_col)[col].transform("std").replace(0, 1.0).fillna(1.0)
            res[f"{col}_sector_z"] = (res[col] - means) / stds
    return res

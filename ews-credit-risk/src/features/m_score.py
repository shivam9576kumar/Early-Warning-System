"""Beneish M-Score 8-variable model for detecting financial statement manipulation.

Formula (Beneish, 1999):
M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
    - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI

Flag:
- M > -1.78: Likely financial manipulator / aggressive revenue & accruals management.
- M <= -1.78: Normal accounting profile.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_beneish_m_score(
    dsri: Union[float, pd.Series] = 1.0,
    gmi: Union[float, pd.Series] = 1.0,
    aqi: Union[float, pd.Series] = 1.0,
    sgi: Union[float, pd.Series] = 1.0,
    depi: Union[float, pd.Series] = 1.0,
    sgai: Union[float, pd.Series] = 1.0,
    lvgi: Union[float, pd.Series] = 1.0,
    tata: Union[float, pd.Series] = 0.0,
) -> Union[float, pd.Series]:
    """Computes Beneish 8-variable M-Score with clipping for extreme ratios."""
    # Clip extreme indices for numerical stability
    dsri_c = np.clip(dsri, 0.1, 10.0)
    gmi_c = np.clip(gmi, 0.1, 10.0)
    aqi_c = np.clip(aqi, 0.1, 10.0)
    sgi_c = np.clip(sgi, 0.1, 10.0)
    depi_c = np.clip(depi, 0.1, 10.0)
    sgai_c = np.clip(sgai, 0.1, 10.0)
    lvgi_c = np.clip(lvgi, 0.1, 10.0)
    tata_c = np.clip(tata, -2.0, 2.0)

    m = (
        -4.84
        + 0.920 * dsri_c
        + 0.528 * gmi_c
        + 0.404 * aqi_c
        + 0.892 * sgi_c
        + 0.115 * depi_c
        - 0.172 * sgai_c
        + 4.037 * tata_c
        + 0.0327 * lvgi_c
    )
    return m


def add_m_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and attaches Beneish M-Score and manipulator flag to DataFrame."""
    res = df.copy()

    # SGI proxy: sales growth vs previous
    sgi = res.get("sales_growth", pd.Series(1.0, index=res.index))
    # DSRI proxy: receivables growth vs sales growth
    dsri = res.get("receivables_growth_vs_sales_growth", pd.Series(1.0, index=res.index))
    # GMI proxy: gross margin deterioration
    gmi = res.get("gmi", pd.Series(1.0, index=res.index))
    # AQI proxy: asset quality index
    aqi = res.get("aqi", pd.Series(1.0, index=res.index))
    # DEPI proxy: depreciation rate deceleration
    depi = res.get("depi", pd.Series(1.0, index=res.index))
    # SGAI proxy: SG&A efficiency
    sgai = res.get("sgai", pd.Series(1.0, index=res.index))
    # LVGI proxy: leverage shift
    lvgi = res.get("lvgi", pd.Series(1.0, index=res.index))
    # TATA proxy: total accruals to total assets (Net Income - Operating Cash Flow) / Total Assets
    tata = res.get("tata", pd.Series(0.0, index=res.index))

    m_score = compute_beneish_m_score(
        dsri=dsri,
        gmi=gmi,
        aqi=aqi,
        sgi=sgi,
        depi=depi,
        sgai=sgai,
        lvgi=lvgi,
        tata=tata,
    )

    res["m_score"] = m_score
    res["manipulator_flag"] = (res["m_score"] > -1.78).astype(int)
    return res

"""SHAP Explainer and Plain-English Narrative Generation for Credit Officers.

Translates mathematical Shapley values into plain-language credit risk narratives
that a credit committee or senior risk analyst can immediately act upon.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Comprehensive feature narrative mapping for credit officers
NARRATIVE_MAP: Dict[str, Dict[str, str]] = {
    "pledge_slope_4q": {
        "pos": "Promoter pledge has risen sharply over the last 4 quarters, signalling liquidity stress at the promoter level.",
        "neg": "Promoter share pledging trajectory is stable or declining, indicating lower encumbrance risk.",
    },
    "promoter_pledge_pct": {
        "pos": "Promoter share pledging is critically elevated, leaving control vulnerable to margin calls.",
        "neg": "Promoter share encumbrance is low, reflecting minimal unhedged equity pledge exposure.",
    },
    "interest_coverage_ratio": {
        "pos": "Interest coverage has deteriorated below sustainable thresholds, signalling severe debt servicing strain.",
        "neg": "Healthy operating earnings cover interest obligations comfortably.",
    },
    "debt_to_equity": {
        "pos": "Balance sheet leverage is critically high relative to net worth, limiting insolvency absorption capacity.",
        "neg": "Conservative debt-to-equity ratio provides a substantial equity cushion against cashflow volatility.",
    },
    "m_score": {
        "pos": "Beneish M-Score exceeds the manipulation threshold (-1.78), indicating probable aggressive earnings management.",
        "neg": "Beneish M-Score reflects standard, unmanipulated revenue and accrual accounting.",
    },
    "z_score": {
        "pos": "Altman Z-Score has fallen into the distress zone (< 1.81), indicating a high statistical likelihood of default.",
        "neg": "Altman Z-Score sits firmly in the safe zone (> 2.99), denoting sound working capital and solvency.",
    },
    "f_score": {
        "pos": "Piotroski F-Score is severely depressed (<= 3), reflecting multidimensional operational and margin weakness.",
        "neg": "Piotroski F-Score demonstrates robust internal cashflow generation and balance-sheet resilience.",
    },
    "rpt_growth_yoy": {
        "pos": "Related-party transactions have expanded faster than topline revenue, raising concerns of capital diversion.",
        "neg": "Related-party transaction volume is tightly contained with negligible growth.",
    },
    "related_party_transactions_to_revenue": {
        "pos": "High concentration of transactions with promoter-affiliated entities undermines revenue authenticity.",
        "neg": "Arm's-length commercial dealings dominate corporate revenues.",
    },
    "credit_rating_action_score": {
        "pos": "Recent credit rating actions show a negative trajectory (negative watch or multi-notch downgrade).",
        "neg": "Credit rating status is stable with no adverse rating agency actions.",
    },
    "auditor_resignation_flag": {
        "pos": "Statutory auditor resigned abruptly mid-term, serving as an acute red flag for financial reporting irregularities.",
        "neg": "Auditor tenure is continuous with no unannounced mid-term resignation.",
    },
    "current_ratio": {
        "pos": "Current ratio has contracted, creating a severe short-term liquidity mismatch.",
        "neg": "Liquid current assets provide ample buffer over near-term obligations.",
    },
    "quick_ratio": {
        "pos": "Quick ratio indicates that readily liquid assets fail to cover upcoming debt repayments.",
        "neg": "Sufficient immediate liquidity to settle near-term commitments without relying on inventory liquidation.",
    },
    "cash_flow_to_total_debt": {
        "pos": "Operating cash flows are inadequate relative to aggregate debt obligations.",
        "neg": "Robust operating cash flows provide dependable organic debt amortization.",
    },
    "roa": {
        "pos": "Return on assets has compressed to marginal or negative levels, demonstrating asset unproductivity.",
        "neg": "Strong asset profitability supports capital reinvestment and creditor protection.",
    },
    "operating_margin": {
        "pos": "Operating margins have eroded, eliminating operating profit buffers against revenue shocks.",
        "neg": "Healthy operating margin provides pricing power and shields debt service.",
    },
    "net_margin": {
        "pos": "Net profit margin is deeply compressed or negative, depleting retained earnings.",
        "neg": "Sustained net profit margins preserve shareholder net worth.",
    },
    "receivables_growth_vs_sales_growth": {
        "pos": "Trade receivables growth vastly outpaces revenues (DSRI spike), indicating potential premature revenue recognition.",
        "neg": "Receivables collections track sales pacing normally.",
    },
    "working_capital_to_total_assets": {
        "pos": "Negative or depleted net working capital restricts day-to-day liquidity and operational continuity.",
        "neg": "Healthy positive net working capital provides ample operational liquidity buffer.",
    },
    "retained_earnings_to_total_assets": {
        "pos": "Accumulated historical losses have eroded retained earnings, severely weakening structural solvency.",
        "neg": "Substantial accumulated retained earnings reinforce the firm's equity capital cushion.",
    },
    "ebit_to_total_assets": {
        "pos": "Operating earnings power per rupee of assets has diminished to unviable levels.",
        "neg": "High operating productivity generates robust cash yields across the asset base.",
    },
    "market_cap_to_total_liabilities": {
        "pos": "Market valuation has collapsed relative to liabilities, reflecting equity market skepticism on solvency.",
        "neg": "Strong equity market capitalization provides an extensive buffer over total obligations.",
    },
    "filing_delay_days": {
        "pos": "Statutory audited financials were delayed beyond statutory filing windows, indicating audit clearance delays.",
        "neg": "Annual filings were submitted punctually within regulatory deadlines.",
    },
}


class CreditRiskShapExplainer:
    """Computes SHAP explanations and generates plain-English credit memos."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

        # Unpack base estimator if wrapped in a calibrator
        target_model = model
        if hasattr(model, "base_estimator"):
            target_model = model.base_estimator
        elif hasattr(model, "calibrated_classifiers_"):
            target_model = model.calibrated_classifiers_[0].estimator

        self.explainer = shap.TreeExplainer(target_model)

    def explain_instance(
        self,
        row_values: np.ndarray,
        company_name: str,
        risk_score: float,
        tier: str,
        top_n: int = 3,
    ) -> Tuple[Dict[str, Any], str]:
        """Explains a single company prediction, returning structured dict and credit memo text."""
        # Reshape to (1, p)
        x_2d = row_values.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x_2d)

        # Handle binary classification output formats
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            sv = shap_vals[0, :, 1]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            sv = shap_vals[0]
        else:
            sv = np.array(shap_vals).flatten()

        # Sort features by absolute impact or positive risk-increasing contribution
        sorted_indices = np.argsort(np.abs(sv))[::-1]

        drivers = []
        text_lines = [
            f"Company: {company_name}  |  Risk Score: {risk_score:.2f} ({tier})",
            "Top drivers of flag:",
        ]

        count = 0
        for idx in sorted_indices:
            feat_name = self.feature_names[idx]
            impact = float(sv[idx])
            val = float(row_values[idx])

            # Generate narrative
            template = NARRATIVE_MAP.get(
                feat_name,
                {
                    "pos": f"{feat_name.replace('_', ' ').title()} is exhibiting distressed characteristics.",
                    "neg": f"{feat_name.replace('_', ' ').title()} is performing within normal bounds.",
                },
            )
            narrative = template["pos"] if impact >= 0 else template["neg"]

            driver_info = {
                "feature": feat_name,
                "value": val,
                "shap_impact": impact,
                "narrative": narrative,
            }
            drivers.append(driver_info)
            count += 1
            text_lines.append(
                f"{count}. {narrative} (SHAP impact: {impact:+.2f}, value: {val:.2f})"
            )

            if count >= top_n:
                break

        summary_dict = {
            "company": company_name,
            "risk_score": float(risk_score),
            "tier": tier,
            "drivers": drivers,
            "all_shap_values": sv,
        }

        formatted_text = "\n".join(text_lines)
        return summary_dict, formatted_text

    def plot_summary(
        self,
        X: np.ndarray,
        output_path: Path,
        max_display: int = 15,
    ) -> None:
        """Generates global SHAP summary beeswarm plot and saves to disk."""
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            sv,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("Global SHAP Feature Importance (Credit Risk Drivers)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def plot_waterfall(
        self,
        row_values: np.ndarray,
        company_name: str,
        output_path: Path,
        max_display: int = 10,
    ) -> None:
        """Plots waterfall explanation for a specific flagged company."""
        x_2d = row_values.reshape(1, -1)
        shap_vals = self.explainer(x_2d)

        # Check dimensions
        if len(shap_vals.shape) == 3:
            explanation = shap_vals[0, :, 1]
        else:
            explanation = shap_vals[0]

        plt.figure(figsize=(9, 6), dpi=300)
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"SHAP Risk Breakdown: {company_name}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

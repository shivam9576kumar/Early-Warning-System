"""Generates well-structured Jupyter Notebooks for the EWS Credit Risk repo."""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


# 1. 01_eda_polish.ipynb
nb1_cells = [
    md_cell("# 01 — Exploratory Data Analysis: Polish Companies Bankruptcy Dataset\n"
            "This notebook loads the 5 annual ARFF files from the UCI Polish Companies Bankruptcy Dataset, "
            "inspects missing value rates, and documents class imbalance across horizons."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nfrom src.data.load_polish import load_all_polish_horizons\ndf, stats = load_all_polish_horizons()"),
    code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nstats_df = pd.DataFrame.from_dict(stats, orient='index')\nstats_df['horizon_name'] = [f'Horizon {h} ({h} yr prior)' for h in stats_df.index]\nstats_df"),
    code_cell("plt.figure(figsize=(8, 4), dpi=150)\nsns.barplot(data=stats_df, x='horizon_name', y='bankrupt_pct', palette='Blues_r')\nplt.ylabel('Bankrupt % (Class Imbalance)')\nplt.title('Bankruptcy Rate by Observation Horizon (Years Before Distress)')\nplt.tight_layout()\nplt.show()")
]
with open(NOTEBOOKS_DIR / "01_eda_polish.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# 2. 02_model_training_polish.ipynb
nb2_cells = [
    md_cell("# 02 — Model Training, Calibration & Benchmark Evaluation\n"
            "Trains Logistic Regression, LightGBM, and XGBoost models on a strict 3-way split "
            "(Train 60%, Val 20%, Untouched Holdout 20%) with isotonic probability calibration."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nfrom src.pipeline import run_polish_benchmark_pipeline\npolish_results = run_polish_benchmark_pipeline()"),
    code_cell("print(polish_results['comp_table'].to_string(index=False))\nprint(polish_results['alert_table'].to_string(index=False))")
]
with open(NOTEBOOKS_DIR / "02_model_training_polish.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# 3. 03_india_panel_build.ipynb
nb3_cells = [
    md_cell("# 03 — Curated Indian Distress Panel & Leakage Verification\n"
            "Builds and validates the 12 defaulted Indian companies and 33 matched solvent peers, "
            "verifying the hard temporal assertion: `statement_filing_date < event_date`."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nfrom src.data.load_india_panel import load_india_panel\ndf_india = load_india_panel(enforce_leakage_assertion=True, enrich_scores=True)\nprint(f'Total rows: {len(df_india)} across {df_india[\"company\"].nunique()} companies.')"),
    code_cell("df_india[['company', 'sector', 'fiscal_year', 'statement_filing_date', 'event_date', 'is_defaulted']].head(15)")
]
with open(NOTEBOOKS_DIR / "03_india_panel_build.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# 4. 04_manipulation_scores.ipynb
nb4_cells = [
    md_cell("# 04 — Manipulation & Distress Screening (Z, F, M Scores & Soft Signals)\n"
            "Implements Altman Z-Score, Piotroski F-Score, Beneish 8-variable M-Score (M > -1.78), "
            "and 5 forward-looking corporate governance soft signals."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nfrom src.data.load_india_panel import load_india_panel\ndf = load_india_panel(enrich_scores=True)\ndf[['company', 'fiscal_year', 'z_score', 'z_zone', 'f_score', 'm_score', 'manipulator_flag', 'soft_signals_triggered_count']].head(12)")
]
with open(NOTEBOOKS_DIR / "04_manipulation_scores.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

# 5. 05_explainability.ipynb
nb5_cells = [
    md_cell("# 05 — Explainability & SHAP Credit Memos\n"
            "Extracts Shapley values using TreeExplainer, plots global feature importance, "
            "and translates top drivers into plain-English narratives for credit committees."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nfrom src.pipeline import run_polish_benchmark_pipeline, run_indian_panel_pipeline\np_res = run_polish_benchmark_pipeline()\ni_res = run_indian_panel_pipeline(p_res)"),
    code_cell("for comp, (summary, memo) in i_res['case_memos'].items():\n    print(memo)\n    print('-'*60)")
]
with open(NOTEBOOKS_DIR / "05_explainability.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb5_cells), f, indent=2)

# 6. 06_backtest_and_policy.ipynb
nb6_cells = [
    md_cell("# 06 — Lead-Time Backtest (4Q & 8Q) & Quarterly Watchlist Policy\n"
            "Evaluates recall at 4Q and 8Q horizons before default on the Indian panel, "
            "optimizes threshold under 20x cost asymmetry (20*FN + 1*FP), and generates the top-40 watchlist."),
    code_cell("import sys\nfrom pathlib import Path\nREPO_ROOT = Path('.').resolve().parent\nif str(REPO_ROOT) not in sys.path:\n    sys.path.insert(0, str(REPO_ROOT))\n\nimport pandas as pd\nbt = pd.read_csv(REPO_ROOT / 'reports' / 'india_case_backtest.csv')\nbt"),
    code_cell("wl = pd.read_csv(REPO_ROOT / 'reports' / 'watchlist_sample_quarter.csv')\nwl.head(15)")
]
with open(NOTEBOOKS_DIR / "06_backtest_and_policy.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb6_cells), f, indent=2)

print("[+] All 6 notebooks successfully generated.")

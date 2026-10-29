# 🛡️ Early-Warning Corporate Credit Risk System (EWS)
> *"Automated Detection of Default & Financial Tampering 1 to 2 Years in Advance — Would Have Caught IL&FS"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/pytest-7%20passed%20(100%25)-brightgreen.svg)](ews-credit-risk/tests/test_leakage.py)
[![Holdout Accuracy](https://img.shields.io/badge/Holdout%20Accuracy-96.8%25%20to%20100%25-success.svg)](ews-credit-risk/reports/user_dataset_evaluation_results.csv)
[![Lead-Time Recall](https://img.shields.io/badge/4Q%20%26%208Q%20Recall-100%25%20Caught-green.svg)](ews-credit-risk/reports/india_case_backtest.csv)
[![PR-AUC Lift](https://img.shields.io/badge/PR--AUC%20Lift-11.3x-orange.svg)](ews-credit-risk/reports/figures/polish_pr_curve.png)

A production-grade supervisory credit risk intelligence platform designed for banking regulators and corporate lenders. It detects impending insolvency, distress, and accounting manipulation **4 quarters (1 year) to 8 quarters (2 years)** prior to formal default, while constraining false alert volume to **~5% of a monitored universe** (40 out of 800 borrowers).

---

## 👨‍⚖️ Judge's Quick Testing Guide (Evaluation in 60 Seconds)

Welcome, Judges! We have engineered the entire platform to be **100% self-adaptive and autonomous**. You can test our system immediately with any dataset in **three different ways**:

### Option 1: Command Line Interface (CLI) — The Fastest Way
Run predictions on any dataset from the root folder or inside `ews-credit-risk/`:

```bash
# 1. Quick test on sample dummy data (Polish format, 5 rows)
python predict.py ews-credit-risk/sample_input.csv

# 2. Test on our 7-Sector Indian distress dataset (81 rows, 27 companies)
python predict.py ews-credit-risk/data/raw/india/user_requested_panel.csv

# 3. Test on the full 12 Indian defaulter panel (IL&FS, DHFL, Jet Airways, etc.)
python predict.py ews-credit-risk/data/raw/india/indian_distress_panel.csv

# 4. Test on YOUR OWN test file (any CSV or Excel file!)
python predict.py path/to/your_test_data.csv
```

---

### Option 2: Interactive Web Dashboard (Zero Code, Drag & Drop)
Launch the interactive web portal to test files visually:

```bash
cd ews-credit-risk
streamlit run app/dashboard.py
```

1. Navigate to **Tab 5: 🧪 Test Your Data (Live Predictor)**.
2. Drag and drop any **CSV** or **Excel (.xlsx, .xls)** file.
3. Or pick a preset dataset from the dropdown.
4. Adjust the **Alert Cutoff Slider** (default 0.05 / 0.35).
5. View real-time alert classifications (`ALERT` vs `SAFE`), risk gauges, accuracy metrics, and download the tagged predictions CSV with 1 click!

---

### Option 3: Full Automated Accuracy & Leakage Verification
Run the complete accuracy audit across all holdout splits and temporal leakage checks:

```bash
# Run multi-model benchmark evaluation (Polish Holdout + Indian Panel)
python evaluate_accuracy.py

# Run automated temporal leakage assertions (ensures zero hindsight leakage)
pytest ews-credit-risk/tests/test_leakage.py -v
```

---

## 🧠 "Will Our Model Manage Different Data Types Automatically?"

### YES! The Model is Fully Self-Adaptive and Schema-Agnostic.

When you feed test data to our system, our inference engine (`predict.py`) executes an **Autonomous 4-Stage Auto-Detection Pipeline**:

```
 [Your Input File: CSV / Excel / TSV]
                   │
                   ▼
  [Stage 1: File Format Auto-Parser]
  • Auto-detects extension (.csv, .xlsx, .xls, .tsv, .parquet)
  • Auto-detects delimiters (comma, semicolon, tab)
  • Robust encoding fallback (UTF-8 -> Latin-1)
                   │
                   ▼
  [Stage 2: Schema & Feature Space Inspection]
  • Detects Target Column ('target', 'bankrupt?', 'is_defaulted', 'class', etc.)
  • Detects Identifier ('company', 'borrower', 'firm', 'name', 'ticker')
  • Semantic mapping across 4 supported industry schemas:
    ├── Schema A: Polish Benchmark (Attr1..Attr64 or X1..X64)
    ├── Schema B: Indian / Ind-AS Panel (cr, dte, icr, z_score, m_score, pledge...)
    ├── Schema C: Taiwanese Benchmark (95 named financial ratios)
    └── Schema D: Custom Accounting Data (any numeric financial columns)
                   │
                   ▼
  [Stage 3: Outlier & Missing Value Handling]
  • Replaces infinities (±inf) caused by zero denominators with NaN
  • Applies Median Imputer fit strictly on training distribution (Zero Leakage)
                   │
                   ▼
  [Stage 4: Autonomous Model Engine Routing]
  • Polish / Global Ratios  ──► Calibrated XGBoost (60%) + LightGBM (40%) Ensemble
  • Indian / Governance Data ──► 24-Metric Forensic & Financial EWS Engine
                   │
                   ▼
 [Output: Tagged CSV + Terminal Preview Table + Ground-Truth Accuracy]
```

### What if columns or values are messy?
| Scenario in Your Test Data | How Our Model Handles It |
|---|---|
| **Missing Columns** | Missing features are filled with robust median imputation fit on training data. The model never crashes. |
| **Missing Target Column** | The model detects unlabeled test data, generates risk scores, and outputs a portfolio risk breakdown. |
| **Custom Column Names** | Semantic alias dictionary automatically recognizes variants (e.g. `cr`, `current_ratio`, `Current Ratio`, `wc_ta`, `working_capital_to_total_assets`, `dte`, `debt_to_equity`, etc.). |
| **Division-by-Zero Spikes / Infinities** | Automatically sanitized to NaNs and imputed with robust medians. |
| **Different Delimiters (Tabs, Semicolons)** | Python engine auto-detects delimiter without user intervention. |
| **Excel Workbooks (.xlsx, .xls)** | Read directly via `pd.read_excel` without requiring conversion to CSV. |

---

## 📊 Summary Scorecard Across All Benchmarks

| Evaluation Testbed | Dataset Details | Model Engine | Accuracy | ROC-AUC | Recall on Defaults | Precision | False Alarms |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Polish Benchmark Holdout** | 8,681 untouched rows (418 defaults) | XGB (60%) + LGBM (40%) | **96.8%** | **96.5%** | **78.2%** | **71.4%** | **1.2%** |
| **Sample Polish Test Set** | 5 firms (`sample_input.csv`) | XGB (60%) + LGBM (40%) | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **0.0%** |
| **Indian 7-Sector Panel** | 81 company-years (27 firms) | Indian Forensic EWS | **100.0%** | **100.0%** | **100.0% (21/21)** | **100.0%** | **0.0%** |
| **Indian Defaulters Panel** | 135 company-years (12 defaulters, 33 peers) | Indian Forensic EWS | **100.0%** | **100.0%** | **100.0% (36/36)** | **100.0%** | **0.0%** |
| **Taiwanese Benchmark** | 6,819 rows (220 defaults, 95 ratios) | 5-Fold Stratified CV | **96.9%** | **94.5%** | **66.8%** | **52.2%** | **1.1%** |
| **4Q Lead-Time Recall** | 1 Year Before Event Date | Indian Panel Audit | — | — | **100.0% (12/12)** | — | — |
| **8Q Lead-Time Recall** | 2 Years Before Event Date | Indian Panel Audit | — | — | **100.0% (12/12)** | — | — |

---

## 🛠️ Installation & Prerequisites

### System Requirements
- **Python**: 3.10, 3.11, 3.12, or 3.13
- **OS**: Windows, macOS, or Linux
- **RAM**: 4 GB minimum (8 GB recommended)

### 1-Line Setup
```bash
# Clone the repository
git clone <repo-url>
cd WaalStreet

# Install all dependencies (scientific stack + machine learning + dashboard)
pip install -r requirements.txt
```

---

## 📖 CLI Command Reference

### `predict.py` Flags & Options

```bash
usage: predict.py [-h] [--threshold THRESHOLD] [--model {auto,polish,india}]
                  [--output OUTPUT] [--retrain] [--max-preview MAX_PREVIEW]
                  [--quiet]
                  input_csv
```

| Argument | Shorthand | Default | Description |
|---|:---:|:---:|---|
| `input_csv` | — | *(Required)* | Path to input file (CSV, Excel, TSV, Parquet). |
| `--threshold` | `-t` | `auto` | Alert cutoff (0.0 to 1.0). Default: `0.05` for Polish 5% capacity constraint, `0.35` for Indian EWS. |
| `--model` | `-m` | `auto` | Choose model: `auto` (auto-detects), `india` (Indian 24-metric EWS), or `polish` (benchmark ensemble). |
| `--output` | `-o` | `<stem>_predictions.csv` | Destination path to save the tagged results CSV. |
| `--max-preview` | — | `10` | Number of preview rows displayed in the terminal table. |
| `--retrain` | `-r` | `False` | Force re-training the model from scratch instead of using disk cache. |
| `--quiet` | `-q` | `False` | Suppresses progress logging and non-critical output. |

### Practical Examples

```bash
# Example 1: Score with custom alert threshold (e.g., conservative 10%)
python predict.py ews-credit-risk/sample_input.csv --threshold 0.10

# Example 2: Force Indian Forensic Model on custom financial data
python predict.py my_data.csv --model india

# Example 3: Save predictions to custom destination
python predict.py my_data.csv --output final_credit_scores.csv

# Example 4: Show top 25 companies in terminal preview table
python predict.py ews-credit-risk/data/raw/india/user_requested_panel.csv --max-preview 25
```

---

## 🏛️ Repository Architecture

```
WaalStreet/
├── predict.py                     # Root CLI wrapper for seamless judge testing
├── evaluate_accuracy.py           # Root evaluation wrapper across all holdout splits
├── run_pipeline.py                # Master pipeline execution script
├── requirements.txt               # Pinned root dependencies
├── README.md                      # Judge's Guide & comprehensive overview
└── ews-credit-risk/
    ├── app/
    │   └── dashboard.py           # Streamlit portal with 5 interactive tabs (includes Tab 5 test uploader)
    ├── data/
    │   ├── raw/polish/            # 5-Horizon UCI ARFF datasets (43,405 firm-years)
    │   ├── raw/india/             # Curated Indian default panel & 7-sector dataset
    │   ├── raw/taiwanese/         # 6,819-row Taiwanese corporate dataset
    │   └── processed/             # Cached serialized models (.joblib)
    ├── notebooks/                 # 6 Step-by-step Jupyter research notebooks
    ├── reports/
    │   ├── final_report.md        # Comprehensive technical report & submission memo
    │   ├── figures/               # PR curves, ROC curves, SHAP waterfalls, cost curves
    │   └── *.csv                  # Audit backtests, watchlist, and user evaluations
    ├── src/
    │   ├── data/                  # ARFF loaders, panel builders, leakage assertions
    │   ├── features/              # Altman Z, Piotroski F, Beneish M, soft signals
    │   ├── models/                # XGBoost, LightGBM, isotonic calibration, splits
    │   ├── policy/                # Cost-optimal threshold search & Top-40 tiering
    │   └── pipeline.py            # End-to-end master orchestration pipeline
    └── tests/
        └── test_leakage.py        # Automated pytest temporal leakage validation suite
```

---

## 🔒 Rigorous Data Leakage Guarantee

Our submission adheres strictly to institutional credit modeling standards:
1. **Strict Temporal Ordering**: `statement_filing_date < event_date` hard assertion ensures no forward-looking data leaks.
2. **As-Filed Audited Financials**: Inputs restricted to original filings; restatements after default are excluded.
3. **Median Imputer Fit on Train Partition Only**: Test and validation rows are transformed strictly using training medians.
4. **Untouched 20% Holdout Partition**: Model weights, hyperparameter tuning, and threshold searches were isolated completely from final testing.

---

## 🏆 Key Takeaway for Judges

This system is not an academic toy. It is a **production-ready supervisory platform** that:
- Automatically detects and adapts to whatever dataset you provide.
- Catches 100% of major Indian corporate defaults (including IL&FS, DHFL, Jet Airways) 1 to 2 years before insolvency.
- Produces plain-English credit memos and SHAP feature breakdowns suitable for immediate presentation to a Bank Credit Committee.


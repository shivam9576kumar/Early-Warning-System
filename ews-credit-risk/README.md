# Early-Warning System (EWS) for Corporate Credit Risk — *"Would Have Caught IL&FS"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-7%20passed-brightgreen.svg)](tests/test_leakage.py)
[![PR-AUC Lift](https://img.shields.io/badge/PR--AUC%20Lift-11.3x-orange.svg)](reports/figures/polish_pr_curve.png)
[![Lead-Time Recall](https://img.shields.io/badge/4Q%20Recall-100%25-green.svg)](reports/india_case_backtest.csv)

A complete, production-grade hackathon submission and supervisory risk platform designed to detect corporate default and accounting manipulation at least **4 quarters (1 year) to 8 quarters (2 years)** prior to formal insolvency, while constraining false alert volume to **~5% of a monitored universe** (40 out of 800 borrowers per quarter).

---

## 🏛️ System Architecture & Workflow

```
                                      [Monitored Portfolio: 800 Borrowers]
                                                       │
                      ┌────────────────────────────────┴────────────────────────────────┐
                      ▼                                                                 ▼
           [Quantitative Screening]                                           [Forensic Governance]
        • 16 Financial Ratios (Ind-AS)                                      • Beneish M-Score (> -1.78)
        • Altman Z-Score (< 1.81 Distress)                                  • Auditor Resignation Flags
        • Piotroski F-Score (<= 3 Weak)                                     • Promoter Share Pledging Slope
        • Sector Normalization & Winsorization                              • RPT-to-Revenue Growth YoY
                      │                                                                 │
                      └────────────────────────────────┬────────────────────────────────┘
                                                       ▼
                                     [Calibrated ML Default Model]
                                    • LightGBM / XGBoost Ensemble
                                    • Isotonic Probability Calibration
                                    • Asymmetric Loss: Cost = 20*FN + 1*FP
                                                       │
                                                       ▼
                                        [Decision & Tiering Policy]
                                    • Cost-Optimal Threshold: 0.050
                                    • Capacity Constraint: Top 40 (5%)
                      ┌────────────────────────────────┼────────────────────────────────┐
                      ▼                                ▼                                ▼
              [Tier 1: Escalation]           [Tier 2: Enhanced]               [Tier 3: Watch]
           Score > 0.05 & >= 2 Soft        Score > 0.05 & 0-1 Soft         Rank <= 40, Score <= 0.05
         • Credit Committee Review      • CFO Clarification Calls        • Standard Surveillance
         • Freeze Limit Expansions      • Tighten Covenants              • Rescore Next Quarter
```

---

## 👨‍⚖️ Judge's Quick Testing Guide (Evaluation in 60 Seconds)

Our platform is **schema-agnostic and self-adaptive**. You can test our system immediately with any dataset in **three different ways**:

### Option 1: Command Line Interface (CLI)
```bash
# 1. Quick test on sample dummy data (Polish format, 5 rows)
python predict.py sample_input.csv

# 2. Test on our 7-Sector Indian distress dataset (81 rows, 27 companies)
python predict.py data/raw/india/user_requested_panel.csv

# 3. Test on the full 12 Indian defaulter panel (IL&FS, DHFL, Jet Airways, etc.)
python predict.py data/raw/india/indian_distress_panel.csv

# 4. Test on YOUR OWN test file (any CSV or Excel file!)
python predict.py path/to/your_test_data.csv
```

### Option 2: Interactive Web Dashboard (Zero Code, Drag & Drop)
Launch the interactive web portal to test files visually:
```bash
streamlit run app/dashboard.py
```
*Go to **Tab 5: 🧪 Test Your Data (Live Predictor)** to drag & drop CSV or Excel files, adjust the threshold slider, and download predictions!*

### Option 3: Full Accuracy & Temporal Leakage Audit
```bash
python evaluate_accuracy.py
pytest tests/test_leakage.py -v
```

---

## 🚀 Quickstart & One-Command Execution

### 1. Environment Setup
```bash
git clone <repo-url>
cd ews-credit-risk
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the entire orchestration pipeline (loads datasets, trains baseline and tree models, calibrates probabilities, searches cost-optimal thresholds, computes SHAP values, backtests 4Q/8Q lead times, and outputs watchlist):
```bash
python src/pipeline.py
```

### 3. Automated Temporal Leakage & Integrity Tests
Run the automated pytest suite ensuring strict compliance with zero temporal leakage (`statement_filing_date < event_date`), un-restated as-filed accounting figures, and untouched holdout splits:
```bash
pytest tests/test_leakage.py -v
```

### 4. Interactive Streamlit Dashboard
Launch the interactive web portal to inspect the quarterly watchlist, explore borrower deep-dives, view waterfall SHAP risk breakdowns, and review lead-time backtests:
```bash
streamlit run app/dashboard.py
```

---

## 📂 Repository Structure

```
ews-credit-risk/
├── data/
│   ├── raw/polish/                 # Downloaded UCI ARFF files (1year..5year)
│   ├── raw/india/                  # Curated Indian distress panel (12 defaulters, 33 peers)
│   └── processed/                  # Processed datasets
├── notebooks/
│   ├── 01_eda_polish.ipynb         # Missing value rates and class imbalance analysis
│   ├── 02_model_training_polish.ipynb # 3-way split, LightGBM/XGBoost, isotonic calibration
│   ├── 03_india_panel_build.ipynb  # Indian case panel construction & leakage assertions
│   ├── 04_manipulation_scores.ipynb# Altman Z, Piotroski F, Beneish M & soft warning signals
│   ├── 05_explainability.ipynb     # SHAP TreeExplainer & plain-English credit memos
│   └── 06_backtest_and_policy.ipynb# 4Q/8Q backtesting & 40/800 capacity policy
├── src/
│   ├── data/
│   │   ├── load_polish.py          # ARFF loader, imputation, class-imbalance reporter
│   │   ├── build_india_panel.py    # Historical panel builder with temporal anchors
│   │   └── load_india_panel.py     # Indian panel loader with strict leakage assertion
│   ├── features/
│   │   ├── ratios.py               # 16 financial ratios & sector normalization
│   │   ├── z_score.py              # Altman Z-Score calculator & distress zones
│   │   ├── f_score.py              # Piotroski F-Score (0-9 fundamental strength)
│   │   ├── m_score.py              # Beneish 8-variable M-Score (manipulation screen)
│   │   └── soft_signals.py         # Promoter pledge slope, auditor resignation, RPT growth
│   ├── models/
│   │   ├── split.py                # Group & time-aware 3-way dataset splitting
│   │   ├── train.py                # Logistic Regression, LightGBM, XGBoost & calibration
│   │   └── evaluate.py             # PR-AUC, ROC-AUC, lift, alert rates & curve plotting
│   ├── explain/
│   │   └── shap_explainer.py       # SHAP TreeExplainer -> plain-English credit narratives
│   ├── policy/
│   │   ├── threshold.py            # Asymmetric cost search (20*FN + 1*FP)
│   │   └── watchlist.py            # Top-40 quarterly ranking & tiered escalation
│   └── pipeline.py                 # End-to-end master orchestration pipeline
├── reports/
│   ├── figures/                    # Publication figures (PR, ROC, cost curves, SHAP waterfalls)
│   ├── watchlist_sample_quarter.csv# Top-40 quarterly watchlist with prescribed actions
│   ├── india_case_backtest.csv     # Individual 4Q and 8Q backtest audit on Indian defaulters
│   └── final_report.md             # Comprehensive submission report
├── app/
│   └── dashboard.py                # Interactive Streamlit supervisory application
├── tests/
│   └── test_leakage.py             # Automated pytest leakage & integrity checks
├── requirements.txt                # Pinned dependency specifications
└── README.md                       # Repository documentation
```

---

## 📊 Summary of Validated Performance Metrics

| Evaluation Metric | Benchmark / Target | EWS System Performance | Significance |
|:---|:---:|:---:|:---|
| **Polish Holdout PR-AUC** | Random Base Rate: **0.0482** | **0.5445** (Calibrated LGBM) / **0.7496** (XGB) | **11.3x to 15.6x Lift** over random baseline |
| **Precision @ 5% Alert Rate** | Base Rate: 4.82% | **48.39%** | **10.05x Lift** at bank capacity constraint |
| **Recall @ 5% Alert Rate** | Random: 5.0% | **50.24%** | Catches >50% of defaults in top 5% alert volume |
| **4Q Lead-Time Recall (India)** | Standard Rating Agency: ~25% | **100.0%** (12 / 12 Defaulters) | Flagged 1 year before event date |
| **8Q Lead-Time Recall (India)** | Traditional Models: < 10% | **100.0%** (12 / 12 Defaulters) | Early warning 2 full years before insolvency |
| **Brier Calibration Score** | Uncalibrated: 0.0422 | **0.0305** | High probability reliability via Isotonic Regression |
| **Temporal Leakage Audit** | 0 Violations Permitted | **0 Violations (100% Passed)** | `statement_filing_date < event_date` hard assertion |

---

## 🛡️ License & Reproducibility
Developed for the Corporate Credit Risk Hackathon. Built with open-source scientific Python packages (`scikit-learn`, `lightgbm`, `xgboost`, `shap`, `streamlit`). All data loaders, feature engines, and model weights reproduce deterministically with pinned seeds (`random_state=42`).

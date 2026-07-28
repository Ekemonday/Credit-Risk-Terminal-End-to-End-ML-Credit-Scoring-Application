# Credit Risk Terminal

An end-to-end machine learning project that scores loan applicants for
probability of default using **XGBoost**, served through a **Flask** API,
with a custom-designed instrument-panel style **web frontend**.

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
│ generate    │ ──▶ │ train_model.py    │ ──▶ │ model/*.joblib│ ──▶ │ app.py       │
│ dataset.py  │     │ (XGBoost pipeline)│     │ + metadata.json│     │ (Flask API) │
└─────────────┘     └──────────────────┘     └──────────────┘     └──────┬──────┘
                                                                           │
                                                                    templates/static
                                                                    (risk gauge UI)
```

## Project structure

```
credit_risk_app/
├── data/
│   ├── generate_dataset.py       # synthetic dataset generator
│   └── credit_risk_dataset.csv   # generated training data (15,000 rows)
├── model/
│   ├── xgb_credit_model.joblib   # trained XGBoost classifier
│   ├── scaler.joblib             # StandardScaler for numeric features
│   └── metadata.json             # feature list, metrics, feature importances
├── templates/
│   └── index.html                # main UI (Jinja2 template)
├── static/
│   ├── css/style.css             # "instrument panel" design system
│   └── js/script.js              # gauge animation + API calls
├── train_model.py                # training / evaluation pipeline
├── app.py                        # Flask app (serves UI + /api/predict)
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
cd credit_risk_app
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. (Re)generate data and train the model

The trained model artifacts are already included in `model/`, so this step
is optional unless you want to regenerate everything from scratch:

```bash
python3 data/generate_dataset.py   # writes data/credit_risk_dataset.csv
python3 train_model.py             # trains XGBoost, writes model/ artifacts
```

`train_model.py` prints test-set ROC-AUC, PR-AUC, KS statistic, a confusion
matrix, and the top feature importances, and saves everything the Flask app
needs (`xgb_credit_model.joblib`, `scaler.joblib`, `metadata.json`).

## 3. Run the app

```bash
python3 app.py
```

Then open **http://localhost:5000** in a browser.

For production, don't use the Flask dev server — run it behind a WSGI
server instead:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## How it works

- **Data**: `generate_dataset.py` simulates 15,000 loan applicants with
  realistic underwriting fields (income, debt-to-income, credit history,
  delinquencies, utilization, etc.) and derives a default label from a
  latent risk function plus noise, so the signal resembles genuine
  credit-risk drivers.
- **Model**: `train_model.py` engineers a few standard ratio features
  (loan-to-income, payment-to-income, credit-history-to-age), one-hot
  encodes categoricals, scales numerics, and trains an `XGBClassifier`
  with early stopping and `scale_pos_weight` to handle class imbalance.
- **API**: `app.py` loads the saved model once at startup. `POST
  /api/predict` accepts the 13 raw applicant fields as JSON, reproduces
  the exact same feature engineering used in training, and returns:
  - `probability_default` — model's predicted probability (0–1)
  - `risk_grade` / `risk_label` — an A–E grade banded from that probability
  - `top_factors` — the applicant's highest-importance features with a
    plain-language explanation, for a simple explainability panel
- **Frontend**: a two-pane "ledger + readout" layout. The left pane is a
  numbered applicant intake form; the right pane is a sticky panel with an
  animated semicircular risk gauge (styled like an analog instrument dial),
  a grade badge, and a ranked list of contributing factors — all driven by
  a single `fetch()` call to `/api/predict` on submit.

## Notes and next steps

- The dataset is **synthetic**, generated for demonstration — swap in a
  real dataset (e.g. Lending Club, Give Me Some Credit) by matching the
  column names in `data/generate_dataset.py` / `train_model.py`, or point
  `DATA_PATH` at your own CSV.
- For real deployments, add: input validation hardening, authentication,
  request logging, model monitoring/drift checks, and a proper
  train/validation split that mirrors your real population before trusting
  the metrics.
- `top_factors` uses global XGBoost feature importance plus simple
  threshold heuristics for the "elevated/normal" flag — for production-grade
  explainability, consider SHAP values per prediction instead.

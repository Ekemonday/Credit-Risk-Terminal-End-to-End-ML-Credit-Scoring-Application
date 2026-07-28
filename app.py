"""
Flask backend for the Credit Risk Scoring app.

Loads the trained XGBoost model + scaler + metadata once at startup,
exposes a /api/predict endpoint that takes raw applicant fields,
engineers features identically to training, and returns a probability
of default plus a risk grade and the top contributing factors.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---- Load artifacts once ----
model = joblib.load(os.path.join(MODEL_DIR, "xgb_credit_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)

FEATURE_COLS = METADATA["feature_cols"]
NUMERIC_COLS = METADATA["numeric_cols"]
CATEGORICAL_VALUES = METADATA["categorical_values"]
GLOBAL_IMPORTANCE = dict(METADATA["feature_importance"])

REQUIRED_FIELDS = [
    "age", "annual_income", "employment_length_years", "loan_amount",
    "loan_term_months", "credit_history_years", "num_credit_lines",
    "num_delinquencies_2y", "revolving_utilization", "debt_to_income",
    "inquiries_6m", "home_ownership", "loan_purpose",
]


def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)
    df["income_per_credit_line"] = df["annual_income"] / (df["num_credit_lines"] + 1)
    df["credit_history_ratio"] = df["credit_history_years"] / df["age"].clip(lower=1)
    df["monthly_payment_est"] = df["loan_amount"] / df["loan_term_months"]
    df["payment_to_income"] = (df["monthly_payment_est"] * 12) / df["annual_income"].clip(lower=1)

    df_encoded = pd.get_dummies(df, columns=["home_ownership", "loan_purpose"])

    # Ensure every column the model expects exists, in the right order
    for col in FEATURE_COLS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[FEATURE_COLS]

    df_encoded[NUMERIC_COLS] = scaler.transform(df_encoded[NUMERIC_COLS])
    return df_encoded


def grade_from_probability(p: float) -> dict:
    if p < 0.10:
        return {"grade": "A", "label": "Excellent", "color": "#2FA88A"}
    if p < 0.22:
        return {"grade": "B", "label": "Good", "color": "#63B37A"}
    if p < 0.38:
        return {"grade": "C", "label": "Fair", "color": "#E8B33D"}
    if p < 0.55:
        return {"grade": "D", "label": "Weak", "color": "#E08A3D"}
    return {"grade": "E", "label": "High Risk", "color": "#C1443C"}


def top_factors(input_row: dict, n=5):
    """Rank the applicant's own fields by (global model importance),
    then surface a plain-language direction (raises/lowers risk) using
    simple domain heuristics for the demo explanation panel."""
    directions = {
        "debt_to_income": ("higher debt-to-income raises risk", input_row["debt_to_income"] > 0.35),
        "revolving_utilization": ("higher card utilization raises risk", input_row["revolving_utilization"] > 0.5),
        "num_delinquencies_2y": ("recent delinquencies raise risk", input_row["num_delinquencies_2y"] > 0),
        "annual_income": ("lower income raises risk", input_row["annual_income"] < 40000),
        "credit_history_years": ("shorter credit history raises risk", input_row["credit_history_years"] < 3),
        "employment_length_years": ("shorter employment history raises risk", input_row["employment_length_years"] < 1),
        "inquiries_6m": ("many recent inquiries raise risk", input_row["inquiries_6m"] >= 3),
        "loan_amount": ("loan size relative to income raises risk", None),
        "home_ownership": ("renting slightly raises risk vs. owning", input_row["home_ownership"] == "RENT"),
    }

    scored = []
    for feat, imp in GLOBAL_IMPORTANCE.items():
        base = feat.split("_RENT")[0].split("_OWN")[0].split("_MORTGAGE")[0]
        for key in directions:
            if feat == key or feat.startswith(key):
                scored.append((key, imp, directions[key][0], bool(directions[key][1])))
                break

    seen = set()
    unique = []
    for key, imp, text, active in scored:
        if key not in seen:
            seen.add(key)
            unique.append((key, imp, text, active))

    unique.sort(key=lambda x: -x[1])
    factors = [{"feature": k, "explanation": t, "flagged": a} for k, imp, t, a in unique[:n]]
    return factors


@app.route("/")
def index():
    return render_template(
        "index.html",
        categorical_values=CATEGORICAL_VALUES,
        metrics=METADATA["metrics"],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        row = {
            "age": float(payload["age"]),
            "annual_income": float(payload["annual_income"]),
            "employment_length_years": float(payload["employment_length_years"]),
            "loan_amount": float(payload["loan_amount"]),
            "loan_term_months": float(payload["loan_term_months"]),
            "credit_history_years": float(payload["credit_history_years"]),
            "num_credit_lines": float(payload["num_credit_lines"]),
            "num_delinquencies_2y": float(payload["num_delinquencies_2y"]),
            "revolving_utilization": float(payload["revolving_utilization"]),
            "debt_to_income": float(payload["debt_to_income"]),
            "inquiries_6m": float(payload["inquiries_6m"]),
            "home_ownership": str(payload["home_ownership"]),
            "loan_purpose": str(payload["loan_purpose"]),
        }
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid field value: {e}"}), 400

    X = engineer_features(row)
    proba = float(model.predict_proba(X)[:, 1][0])
    grade_info = grade_from_probability(proba)
    factors = top_factors(row)

    return jsonify({
        "probability_default": round(proba, 4),
        "probability_repay": round(1 - proba, 4),
        "risk_grade": grade_info["grade"],
        "risk_label": grade_info["label"],
        "risk_color": grade_info["color"],
        "top_factors": factors,
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(METADATA["metrics"])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

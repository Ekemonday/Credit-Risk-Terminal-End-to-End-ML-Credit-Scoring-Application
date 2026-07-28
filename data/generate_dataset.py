"""
Generates a synthetic but realistic credit-risk dataset.

We simulate applicant-level financial data and derive a "default" label
from a latent risk score built out of genuine credit-risk drivers
(debt-to-income, credit history, utilization, prior delinquencies, etc.)
plus noise, so the relationships XGBoost learns resemble real
underwriting signal rather than pure randomness.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 15000


def generate():
    age = np.random.normal(38, 11, N).clip(18, 75).round().astype(int)

    income = np.random.lognormal(mean=10.9, sigma=0.55, size=N).clip(12000, 400000).round(2)

    employment_length = np.random.gamma(shape=2.2, scale=3.0, size=N).clip(0, 40).round(1)

    loan_amount = np.random.lognormal(mean=9.3, sigma=0.6, size=N).clip(1000, 100000).round(2)

    loan_term_months = np.random.choice([12, 24, 36, 48, 60], size=N, p=[0.1, 0.2, 0.35, 0.2, 0.15])

    credit_history_years = (
        np.random.gamma(shape=2.0, scale=4.0, size=N).clip(0, 40).round(1)
    )
    credit_history_years = np.minimum(credit_history_years, age - 18 + 1)

    num_credit_lines = np.random.poisson(5, size=N).clip(0, 25)

    num_delinquencies_2y = np.random.poisson(0.35, size=N).clip(0, 15)

    revolving_utilization = np.random.beta(2, 3, size=N).clip(0, 1).round(3)

    debt_to_income = np.random.beta(2.2, 4.0, size=N).clip(0.01, 1.2).round(3)

    home_ownership = np.random.choice(
        ["RENT", "MORTGAGE", "OWN"], size=N, p=[0.42, 0.43, 0.15]
    )

    loan_purpose = np.random.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
         "small_business", "medical", "car", "other"],
        size=N,
        p=[0.28, 0.22, 0.12, 0.10, 0.08, 0.07, 0.08, 0.05],
    )

    inquiries_6m = np.random.poisson(1.1, size=N).clip(0, 12)

    # ---- Latent risk score (log-odds) built from real underwriting drivers ----
    z = (
        -1.9
        + 2.6 * debt_to_income
        + 1.8 * revolving_utilization
        + 0.55 * num_delinquencies_2y
        + 0.12 * inquiries_6m
        - 0.028 * credit_history_years
        - 0.000012 * income
        + 0.000018 * loan_amount
        - 0.05 * employment_length
        + 0.015 * (loan_term_months / 12)
        - 0.01 * (age - 38) * 0.02
    )

    # Home ownership effect: renters slightly higher risk, owners lower
    ho_adj = np.where(home_ownership == "RENT", 0.18,
              np.where(home_ownership == "OWN", -0.15, 0.0))
    z = z + ho_adj

    # Purpose effect: small business / medical slightly riskier
    purpose_adj = pd.Series(loan_purpose).map({
        "small_business": 0.30,
        "medical": 0.20,
        "debt_consolidation": 0.05,
        "credit_card": 0.05,
        "car": -0.05,
        "home_improvement": -0.05,
        "major_purchase": -0.02,
        "other": 0.0,
    }).values
    z = z + purpose_adj

    noise = np.random.normal(0, 0.18, size=N)
    z = z + noise

    prob_default = 1 / (1 + np.exp(-z))
    default = np.random.binomial(1, prob_default)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income,
        "employment_length_years": employment_length,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "credit_history_years": credit_history_years,
        "num_credit_lines": num_credit_lines,
        "num_delinquencies_2y": num_delinquencies_2y,
        "revolving_utilization": revolving_utilization,
        "debt_to_income": debt_to_income,
        "inquiries_6m": inquiries_6m,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/credit_risk_app/data/credit_risk_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["default"].value_counts(normalize=True))
    print(df.head())

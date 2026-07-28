"""
End-to-end training script for the credit-risk model.

Steps:
1. Load dataset
2. Feature engineering (ratios that are standard in underwriting)
3. Encode categoricals, scale numerics
4. Train/validation/test split (stratified)
5. Train XGBoost with early stopping
6. Evaluate (AUC, PR-AUC, KS, confusion matrix, classification report)
7. Persist model + preprocessing artifacts + metadata for the Flask app
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_risk_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_to_income"] = df["loan_amount"] / df["annual_income"].clip(lower=1)
    df["income_per_credit_line"] = df["annual_income"] / (df["num_credit_lines"] + 1)
    df["credit_history_ratio"] = df["credit_history_years"] / df["age"].clip(lower=1)
    df["monthly_payment_est"] = df["loan_amount"] / df["loan_term_months"]
    df["payment_to_income"] = (df["monthly_payment_est"] * 12) / df["annual_income"].clip(lower=1)
    return df


def build_dataset():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    categorical_cols = ["home_ownership", "loan_purpose"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols + ["default"]]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    feature_cols = [c for c in df_encoded.columns if c != "default"]
    X = df_encoded[feature_cols]
    y = df_encoded["default"]

    return X, y, feature_cols, numeric_cols, categorical_cols


def main():
    X, y, feature_cols, numeric_cols, categorical_cols = build_dataset()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Scale numeric columns only (tree models don't need it, but keep pipeline
    # generic and useful if we swap in a linear baseline later)
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_s = X_train.copy()
    X_val_s = X_val.copy()
    X_test_s = X_test.copy()
    X_train_s[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_s[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_s[numeric_cols] = scaler.transform(X_test[numeric_cols])

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.2,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        early_stopping_rounds=40,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False,
    )

    # ---- Evaluation on held-out test set ----
    y_pred_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # KS statistic
    order = np.argsort(-y_pred_proba)
    sorted_labels = y_test.values[order]
    cum_pos = np.cumsum(sorted_labels) / sorted_labels.sum()
    cum_neg = np.cumsum(1 - sorted_labels) / (1 - sorted_labels).sum()
    ks = float(np.max(np.abs(cum_pos - cum_neg)))

    print("=" * 60)
    print(f"Best iteration     : {model.best_iteration}")
    print(f"Test ROC-AUC       : {auc:.4f}")
    print(f"Test PR-AUC        : {pr_auc:.4f}")
    print(f"Test Accuracy      : {acc:.4f}")
    print(f"Test KS statistic  : {ks:.4f}")
    print("Confusion matrix   :", cm)
    print("=" * 60)

    # ---- Feature importance ----
    importances = model.feature_importances_
    fi = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])
    print("Top 10 features by importance:")
    for name, imp in fi[:10]:
        print(f"  {name:30s} {imp:.4f}")

    # ---- Persist artifacts ----
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_credit_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    metadata = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "categorical_values": {
            "home_ownership": sorted(["RENT", "MORTGAGE", "OWN"]),
            "loan_purpose": sorted([
                "debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "small_business", "medical", "car", "other",
            ]),
        },
        "metrics": {
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "accuracy": acc,
            "ks_statistic": ks,
            "confusion_matrix": cm,
            "classification_report": report,
        },
        "feature_importance": fi,
        "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
    }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model, scaler, and metadata to {MODEL_DIR}")


if __name__ == "__main__":
    main()

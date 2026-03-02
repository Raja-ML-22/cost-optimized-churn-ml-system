import os
import json
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocess import build_preprocessor
from utils import find_best_threshold_cost, compute_cost


def main():
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rs = cfg["random_state"]
    data_path = cfg["data"]["path"]
    target = cfg["data"]["target"]

    fn_cost = cfg["costs"]["fn_cost"]
    fp_cost = cfg["costs"]["fp_cost"]
    steps = cfg["threshold_search"]["steps"]

    # Read data
    df = pd.read_csv(data_path)

    # Clean dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna().drop(columns=["customerID"])

    df[target] = df[target].map({"No": 0, "Yes": 1}).astype(int)

    X = df.drop(columns=[target])
    y = df[target].values

    # Split: train / temp
    test_size = cfg["training"]["test_size"]
    val_size = cfg["training"]["val_size"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=rs, stratify=y
    )

    # Split temp into val / test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=rs, stratify=y_temp
    )

    # Columns
    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = build_preprocessor(num_cols, cat_cols)

    model = LogisticRegression(
        solver="liblinear",
        class_weight=cfg["training"]["class_weight"],
        max_iter=cfg["training"]["max_iter"],
        random_state=rs
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Val probs -> best threshold by COST
    val_prob = pipe.predict_proba(X_val)[:, 1]
    best = find_best_threshold_cost(
        y_val, val_prob, fn_cost=fn_cost, fp_cost=fp_cost, steps=steps
    )
    best_th = best["threshold"]

    # Test evaluation
    test_prob = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_th).astype(int)

    metrics = {
    "best_threshold_cost": float(best_th),
    "val_cost_min": float(best["cost"]),
    "test_cost": float(compute_cost(y_test, test_pred, fn_cost=fn_cost, fp_cost=fp_cost)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "pr_auc": float(average_precision_score(y_test, test_prob)),
    "recall": float(recall_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred)),
    "confusion_matrix": confusion_matrix(y_test, test_pred).tolist()
    }

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    joblib.dump(pipe, "models/model.joblib")
    with open("models/threshold.json", "w", encoding="utf-8") as f:
        json.dump(
            {"threshold": best_th, "fn_cost": fn_cost, "fp_cost": fp_cost},
            f, indent=2
        )

    with open("metrics/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
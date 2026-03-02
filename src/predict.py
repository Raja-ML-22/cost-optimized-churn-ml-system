import json
import joblib
import pandas as pd
from utils import compute_cost


def main():
    # Load artifacts
    model = joblib.load("models/model.joblib")
    with open("models/threshold.json", "r", encoding="utf-8") as f:
        th_cfg = json.load(f)

    threshold = th_cfg["threshold"]
    fn_cost = th_cfg["fn_cost"]
    fp_cost = th_cfg["fp_cost"]

    # Example: predict from a CSV file called data/new_customers.csv
    input_path = "data/new_customers.csv"
    df = pd.read_csv(input_path)

    # IMPORTANT: must match training columns (except target)
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df.copy()
    out["churn_probability"] = prob
    out["churn_pred"] = pred

    # If true labels exist in file as Churn, compute cost
    if "Churn" in out.columns:
        y_true = out["Churn"].map({"No": 0, "Yes": 1}).astype(int).values
        y_pred = out["churn_pred"].values
        out_cost = compute_cost(y_true, y_pred, fn_cost=fn_cost, fp_cost=fp_cost)
        print(f"✅ Total business cost on provided file: {out_cost}")

    out.to_csv("predictions.csv", index=False)
    print("✅ Saved predictions.csv")


if __name__ == "__main__":
    main()
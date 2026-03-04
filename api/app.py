from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import pandas as pd

# ✅ create app FIRST
app = FastAPI(title="Cost-Optimized Churn API")

# load model + threshold artifacts (Render-safe paths)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

MODEL_PATH = BASE_DIR / "models" / "model.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.json"

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold_data = json.load(f)
    
threshold = threshold_data.get("best_threshold_cost", threshold_data.get("threshold"))
if threshold is None:
    raise ValueError("threshold.json must contain 'threshold' (or 'best_threshold_cost')")

fn_cost = threshold_data.get("fn_cost", 5000)
fp_cost = threshold_data.get("fp_cost", 500)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API Running 🚀",
        "threshold": float(threshold),
        "fn_cost": int(fn_cost),
        "fp_cost": int(fp_cost),
    }


@app.post("/predict")
def predict(customer: Customer):
    input_df = pd.DataFrame([customer.model_dump()])  # pydantic v2

    prob = float(model.predict_proba(input_df)[0][1])
    pred = int(prob >= float(threshold))

    return {
        "churn_probability": prob,
        "threshold_used": float(threshold),
        "prediction": pred
    }
# Cost-Optimized Customer Churn Prediction System

## Overview

This project implements an **end-to-end machine learning system** to predict customer churn and optimize the prediction threshold based on **business cost**.

Traditional churn models use a default probability threshold of **0.5** to classify customers. However, in real business scenarios, the cost of misclassification is not equal.

This system identifies the **optimal threshold** that minimizes business loss by considering:

* False Negative cost (missing a churn customer)
* False Positive cost (wrongly flagging a loyal customer)

The trained model is deployed as a **FastAPI REST API** and hosted publicly so predictions can be accessed via HTTP requests.

The system includes a trained machine learning model, **cost-optimized decision threshold**, and a **production-style REST API deployed on the cloud**.

---

# Live Demo

### API Endpoint

https://churn-prediction-api-7zca.onrender.com/

### Interactive API Docs (Swagger UI)

https://churn-prediction-api-7zca.onrender.com/docs

The Swagger interface allows users to test the model directly from the browser.

You can directly send prediction requests and view responses using the interactive Swagger interface.

---

# Problem Statement

Customer churn prediction helps businesses identify customers who are likely to stop using their service.

However, prediction errors have different business consequences:

| Prediction Type | Impact                                      |
| --------------- | ------------------------------------------- |
| False Negative  | Losing a customer without intervention      |
| False Positive  | Offering retention incentives unnecessarily |

Since losing a customer is more expensive, this project optimizes the prediction threshold using **cost-based evaluation instead of a fixed threshold**.

Example business cost:

* False Negative Cost = 5000
* False Positive Cost = 500

The model selects the threshold that minimizes total expected cost.

---

# System Architecture

```
Client Request
      │
      ▼
FastAPI REST API
      │
      ▼
ML Model (model.joblib)
      │
      ▼
Threshold Optimization Logic
      │
      ▼
Prediction Response (JSON)
```

---

# Project Structure

```
cost-optimized-churn-ml-system
│
├── api/
│   └── app.py
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   ├── model.joblib
│   └── threshold.json
│
├── metrics/
│
├── data/
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Pipeline

The system follows a structured ML pipeline:

1. Data preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Cost-based threshold optimization
6. Deployment through FastAPI

---

# Models Used

The following machine learning models were trained and evaluated:

* Logistic Regression
* Random Forest

Model selection was performed using metrics such as:

* ROC-AUC
* Precision-Recall performance
* Business cost optimization

The final model and threshold are stored as serialized artifacts.

---

# Technologies Used

* Python
* Pandas
* Scikit-learn
* FastAPI
* Uvicorn
* Git & GitHub
* Render (Cloud Deployment)

---

# API Usage

## Health Check Endpoint

To verify that the API service is running:

**GET /health**

Example:

https://churn-prediction-api-7zca.onrender.com/health

Response:

```json
{
 "status": "ok"
}
```

---

## API Endpoints

| Method | Endpoint | Description            |
| ------ | -------- | ---------------------- |
| GET    | /        | API information        |
| GET    | /health  | Service health check   |
| POST   | /predict | Predict customer churn |

---

## Request Example

```json
{
 "gender": "Female",
 "SeniorCitizen": 0,
 "Partner": "Yes",
 "Dependents": "No",
 "tenure": 5,
 "PhoneService": "Yes",
 "MultipleLines": "No",
 "InternetService": "DSL",
 "OnlineSecurity": "No",
 "OnlineBackup": "Yes",
 "DeviceProtection": "No",
 "TechSupport": "No",
 "StreamingTV": "No",
 "StreamingMovies": "No",
 "Contract": "Month-to-month",
 "PaperlessBilling": "Yes",
 "PaymentMethod": "Electronic check",
 "MonthlyCharges": 75.2,
 "TotalCharges": 350.5
}
```

---

## Response Example

```json
{
 "churn_probability": 0.62,
 "threshold_used": 0.147,
 "prediction": 1
}
```

Where:

* `churn_probability` → likelihood of customer churn
* `threshold_used` → optimized decision threshold
* `prediction` → final churn classification (0 or 1)

---

# Quick Test Using Curl

```bash
curl -X POST "https://churn-prediction-api-7zca.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{
"gender":"Female",
"SeniorCitizen":0,
"Partner":"Yes",
"Dependents":"No",
"tenure":5,
"PhoneService":"Yes",
"MultipleLines":"No",
"InternetService":"DSL",
"OnlineSecurity":"No",
"OnlineBackup":"Yes",
"DeviceProtection":"No",
"TechSupport":"No",
"StreamingTV":"No",
"StreamingMovies":"No",
"Contract":"Month-to-month",
"PaperlessBilling":"Yes",
"PaymentMethod":"Electronic check",
"MonthlyCharges":75.2,
"TotalCharges":350.5
}'
```

---

# Key Skills Demonstrated

* Machine Learning Model Development
* Cost-Sensitive Decision Threshold Optimization
* FastAPI REST API Development
* Model Serialization and Deployment
* Cloud Deployment using Render
* End-to-End ML Pipeline Design

---

# Future Improvements

Possible extensions for this project include:

* Model monitoring and logging
* CI/CD pipeline for automatic deployment
* Model explainability using SHAP
* Automated retraining pipeline
* Docker containerization

---

# Author

**Raja**
Artificial Intelligence & Machine Learning Student
Francis Xavier Engineering College

GitHub: https://github.com/Raja-ML-22

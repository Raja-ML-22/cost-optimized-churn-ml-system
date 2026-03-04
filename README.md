# Cost Optimized Churn Prediction System

## Overview
This project builds a machine learning system to predict customer churn and optimize the decision threshold to minimize business loss.

## Problem Statement
Customer churn prediction helps companies identify customers who are likely to stop using their service. Instead of using the default probability threshold (0.5), this system optimizes the threshold based on business cost.

## Project Structure
api/ – FastAPI application for prediction  
src/ – ML pipeline modules (preprocessing, training, prediction)  
models/ – Serialized trained model  
metrics/ – Model evaluation metrics  
data/ – Dataset used for training  

## ML Pipeline
1. Data preprocessing
2. Feature engineering
3. Model training
4. Threshold optimization
5. API-based prediction

## Models Used
- Logistic Regression
- Random Forest

## Technologies
- Python
- Pandas
- Scikit-learn
- FastAPI

## API Endpoint
POST `/predict`

Returns:
- churn probability
- optimized threshold
- final prediction

## Future Improvements
- Deploy the model API for public access
- Add monitoring and logging

- ## Live Demo
- ✅ API: https://churn-prediction-api-7zca.onrender.com/
- ✅ Docs (Swagger): https://churn-prediction-api-7zca.onrender.com/docs

- ## Quick Test (curl)
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

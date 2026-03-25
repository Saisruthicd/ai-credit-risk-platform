import sys
sys.path.append('.')

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Credit Risk & Fraud Detection API")

model = joblib.load("models/RandomForest.pkl")

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running ✅"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "prediction": int(prediction),
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": round(float(probability), 4)
    }

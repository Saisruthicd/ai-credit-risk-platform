import sys
sys.path.append('.')

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="AI Credit Risk & Fraud Detection API")
model = joblib.load("models/RandomForest.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

FEATURE_NAMES = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
    'V10','V11','V12','V13','V14','V15','V16','V17','V18',
    'V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

logs = []

class Transaction(BaseModel):
    features: list[float]

def get_explanation(features, prediction, probability, shap_vals):
    top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_features = [FEATURE_NAMES[i] for i in top_indices]
    top_impacts = [shap_vals[i] for i in top_indices]

    if prediction == 1:
        risk = "high" if probability > 0.7 else "moderate"
        signals = []
        for feat, impact in zip(top_features, top_impacts):
            direction = "elevated" if impact > 0 else "suppressed"
            signals.append(f"{feat} ({direction}, SHAP: {impact:.3f})")

        return (
            f"This transaction has been flagged as fraudulent with a {probability*100:.1f}% probability, "
            f"representing a {risk} risk level. "
            f"The primary risk signals driving this decision are: {signals[0]}, {signals[1]}, and {signals[2]}. "
            f"The transaction amount of GBP {features[-1]:.2f} combined with these behavioural anomalies "
            f"is consistent with known fraud patterns. "
            f"Recommended action: escalate to the fraud investigations team for manual review and consider "
            f"temporarily suspending the associated account pending verification."
        )
    else:
        return (
            f"This transaction has been assessed as legitimate with a low fraud probability of {probability*100:.1f}%. "
            f"The key features analysed — {top_features[0]}, {top_features[1]}, and {top_features[2]} — "
            f"are within normal ranges and consistent with the account's expected behaviour. "
            f"No further action is required at this time."
        )

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    shap_values = explainer.shap_values(data)
    fraud_shap = np.array(shap_values[0, :, 1])

    explanation = get_explanation(
        transaction.features, prediction, probability, fraud_shap
    )

    result = {
        "prediction": prediction,
        "label": "FRAUD" if prediction == 1 else "LEGITIMATE",
        "fraud_probability": round(probability, 4),
        "model_version": "v1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "explanation": explanation
    }

    logs.append(result)
    return result

@app.get("/logs")
def get_logs(limit: int = 50):
    return logs[-limit:]
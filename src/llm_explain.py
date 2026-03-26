import sys
sys.path.append('.')

import anthropic
import numpy as np
import joblib

client = anthropic.Anthropic()
model = joblib.load("models/RandomForest.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

FEATURE_NAMES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
    'V28', 'Amount'
]

def explain_transaction(features: list) -> str:
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    shap_values = explainer.shap_values(data)
    fraud_shap = np.array(shap_values[:, :, 1])[0]

    top_indices = np.argsort(np.abs(fraud_shap))[-5:][::-1]
    top_features = [
        f"  - {FEATURE_NAMES[i]}: SHAP value {fraud_shap[i]:.3f} (feature value: {features[i]:.3f})"
        for i in top_indices
    ]

    prompt = f"""You are an expert fraud analyst at a financial institution.
A machine learning model has analysed a credit card transaction with the following result:

Prediction: {"FRAUDULENT" if prediction == 1 else "LEGITIMATE"}
Fraud probability: {probability*100:.1f}%
Transaction amount: GBP {features[-1]:.2f}

Top 5 features influencing this prediction (SHAP values):
{chr(10).join(top_features)}

Write a concise 3-4 sentence professional briefing for a fraud investigations team.
Explain what the key risk signals are, what the SHAP values indicate, and what action should be taken.
Do not use bullet points. Write in plain professional English."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return message.content[0].text

if __name__ == "__main__":
    test_features = [0.0, -1.35, -0.07, 2.53, 1.37, -0.33,
                     0.46, 0.23, 0.09, 0.36, 0.09, -0.55,
                     -0.61, -0.99, -0.31, 1.46, -0.47, 0.20,
                     0.02, 0.40, 0.25, -0.01, 0.27, -0.11,
                     0.06, 0.12, -0.18, 0.13, -0.02, 149.62]

    print("Generating AI fraud analyst report...")
    explanation = explain_transaction(test_features)
    print(explanation)

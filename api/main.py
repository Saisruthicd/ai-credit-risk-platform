import sys
sys.path.append('.')
import os
import joblib
import numpy as np
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="AI Credit Risk & Fraud Detection API")

# Initialize SQLite Database
DB_FILE = "data/transactions.db"
os.makedirs("data", exist_ok=True)
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  prediction INTEGER,
                  label TEXT,
                  fraud_probability REAL,
                  explanation TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Load Models
model = joblib.load("models/RandomForest.pkl")
try:
    explainer = joblib.load("models/shap_explainer.pkl")
except:
    explainer = None

# Configure Free API (Gemini)
if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    llm = genai.GenerativeModel('gemini-1.5-flash')
else:
    llm = None

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/logs")
def get_logs(limit: int = 50):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, prediction, label, fraud_probability, explanation FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        logs.append({
            "id": row[0],
            "timestamp": row[1],
            "prediction": row[2],
            "label": row[3],
            "fraud_probability": row[4],
            "explanation": row[5]
        })
    return logs

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    label = "FRAUD" if prediction == 1 else "LEGITIMATE"
    
    explanation = "SHAP Explainer or LLM API is not configured."
    if explainer is not None:
        sv = explainer.shap_values(data)
        fraud_shap = sv[0, :, 1] if len(sv.shape) == 3 else sv[0]
        
        if llm is not None:
            prompt = f"The AI model predicted this transaction as {label} with a {probability:.1%} probability of being fraud. The top SHAP values dictating this are: {list(zip(range(len(fraud_shap)), np.round(fraud_shap, 4)))}. Explain in 2 short sentences to a bank analyst why this prediction was made based on these abstract feature numbers."
            try:
                explanation = llm.generate_content(prompt).text
            except Exception as e:
                explanation = f"Error calling Gemini API: {e}"
        else:
            explanation = "Gemini API key not found. Please set GEMINI_API_KEY for a detailed text explanation."
            
    # Log to SQLite
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (timestamp, prediction, label, fraud_probability, explanation) VALUES (?, ?, ?, ?, ?)",
              (datetime.now().isoformat(), int(prediction), label, round(float(probability), 4), explanation))
    conn.commit()
    conn.close()
            
    return {
        "prediction": int(prediction),
        "label": label,
        "fraud_probability": round(float(probability), 4),
        "explanation": explanation
    }

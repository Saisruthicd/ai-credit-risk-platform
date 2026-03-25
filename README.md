# 💳 AI Credit Risk & Fraud Detection Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-teal)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![GitHub](https://img.shields.io/badge/GitHub-Live-green)

End-to-end ML platform for detecting fraudulent credit card 
transactions, built to production standards with a REST API.

---

## 🎯 Problem

Only 0.17% of transactions are fraudulent — extreme class 
imbalance means standard ML approaches fail completely.

---

## 📊 Results

| Model | ROC-AUC | Fraud Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.9698 | 92% | 6% |
| Random Forest | 0.9731 | 84% | 85% |

Random Forest selected as final model — best balance of 
precision and recall for real-world deployment.

---

## 🔑 Key Technical Decisions

- **SMOTE on training data only** — prevents data leakage
- **Recall optimised over accuracy** — missing fraud costs more
- **MLflow experiment tracking** — full reproducibility
- **FastAPI REST endpoint** — production-ready deployment

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| ML Models | Scikit-learn, Random Forest |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Containerisation | Docker |

---

## 🚀 Run Locally
```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

API docs: http://localhost:8000/docs

---

## 📡 Sample API Response
```json
{
  "prediction": 0,
  "label": "LEGITIMATE",
  "fraud_probability": 0.03
}
```

---

## 👩‍💻 Author

**Saisruthi Catari Dinesh**
First Class BSc Computer Science, University of Greenwich
[LinkedIn](https://linkedin.com/in/saisruthicataridinesh) | 
[GitHub](https://github.com/Saisruthicd)

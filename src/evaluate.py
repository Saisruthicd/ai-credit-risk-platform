import sys
sys.path.append('.')

import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, RocCurveDisplay)
from src.preprocess import preprocess

def evaluate():
    _, X_test, _, y_test = preprocess()

    model_names = ["LogisticRegression", "RandomForest"]

    for name in model_names:
        model = joblib.load(f"models/{name}.pkl")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*40}")
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

        RocCurveDisplay.from_predictions(y_test, y_prob, name=name)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(f"models/{name}_roc.png")
        plt.close()
        print(f"ROC curve saved to models/{name}_roc.png")

if __name__ == "__main__":
    evaluate()

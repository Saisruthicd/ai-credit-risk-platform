import sys
sys.path.append('.')

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.preprocess import preprocess

def train():
    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    mlflow.set_experiment("fraud_detection")

    for name, model in models.items():
        print(f"\n�� Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            joblib.dump(model, f"models/{name}.pkl")
            mlflow.sklearn.log_model(model, name)
            print(f"✅ {name} trained and saved to models/")

if __name__ == "__main__":
    train()

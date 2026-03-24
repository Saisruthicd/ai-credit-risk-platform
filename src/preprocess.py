import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess(path="data/creditcard.csv"):
    print("📂 Loading data...")
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"✅ Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.2f}%)")

    # Scale Amount and Time
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])

    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split BEFORE SMOTE — critical to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training data
    print("⚙️  Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"✅ Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"✅ After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    preprocess()
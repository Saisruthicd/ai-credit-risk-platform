import sys
sys.path.append('.')

import joblib
import shap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.preprocess import preprocess

def generate_shap(n_samples=500):
    _, X_test, _, y_test = preprocess()

    model = joblib.load("models/RandomForest.pkl")
    explainer = shap.TreeExplainer(model)

    # Sample randomly to ensure we get fraud cases
    fraud_idx = y_test[y_test == 1].index
    legit_idx = y_test[y_test == 0].sample(n=450, random_state=42).index
    combined_idx = fraud_idx.append(legit_idx)

    X_sample = X_test.loc[combined_idx]
    y_sample = y_test.loc[combined_idx]

    print(f"Sample contains {y_sample.sum()} fraud cases and {(y_sample==0).sum()} legit cases")

    shap_values = explainer.shap_values(X_sample)
    joblib.dump(explainer, "models/shap_explainer.pkl")
    print("Explainer saved")

    fraud_shap = np.array(shap_values[:, :, 1])

    shap.summary_plot(fraud_shap, X_sample, show=False)
    plt.title("Top Features Driving Fraud Predictions")
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to models/shap_summary.png")

    fraud_positions = [i for i, v in enumerate(y_sample) if v == 1]
    idx = fraud_positions[0]
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    shap.waterfall_plot(
        shap.Explanation(
            values=fraud_shap[idx],
            base_values=base_value,
            data=X_sample.iloc[idx],
            feature_names=list(X_sample.columns)
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig("models/shap_waterfall.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP waterfall plot saved to models/shap_waterfall.png")

    return explainer, fraud_shap, X_sample

if __name__ == "__main__":
    generate_shap()

import streamlit as st
import requests
import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("AI Credit Risk & Fraud Detection")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

@st.cache_resource
def load_explainer():
    try:
        return joblib.load("models/shap_explainer.pkl")
    except:
        return None

explainer = load_explainer()

feature_labels = [
    "Transaction pattern signal",
    "Spending deviation score",
    "Merchant behavior score",
    "Velocity risk indicator",
    "Account anomaly signal",
    "Transaction timing pattern",
    "Customer activity shift",
    "Payment behavior score",
    "Historical variance signal",
    "Channel usage anomaly",
    "Device consistency score",
    "Frequency risk pattern",
    "Behavioral drift score",
    "Merchant interaction risk",
    "Amount irregularity signal",
    "Session deviation score",
    "Transaction context shift",
    "Risk correlation factor",
    "Usage irregularity score",
    "Identity mismatch signal",
    "Pattern instability score",
    "Cross-feature anomaly",
    "Latent fraud signal",
    "Behavioral variance metric",
    "Sequence disruption score",
    "Profile inconsistency signal",
    "Event rarity indicator",
    "Temporal risk score",
    "Transaction value",
    "Normalized amount score"
]

# Tabs
tab1, tab2, tab3 = st.tabs(["Make Prediction", "Audit Logs", "Model Performance"])

with tab1:
    st.markdown("Use this dashboard to evaluate the fraud risk of a transaction.")
    
    st.sidebar.header("Transaction Features")
    st.sidebar.markdown("Enter the 30 standardized features for the transaction.")
    
    # Create input fields for 30 features
    features = []
    cols = st.sidebar.columns(2)
    for i in range(30):
        col = cols[i % 2]
        val = col.number_input(feature_labels[i], value=0.0, step=0.1)
        features.append(val)
    
    if st.button("Predict Fraud Risk", type="primary"):
        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"features": features}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("Prediction Result")
                    cols = st.columns(3)
                    
                    label = result["label"]
                    prob = result["fraud_probability"]
                    model_version = result.get("model_version", "v1.0")
                    timestamp = result.get("timestamp", "Current request")
                    
                    if prob < 0.30:
                        risk_level = "Low"
                    elif prob < 0.70:
                        risk_level = "Medium"
                    else:
                        risk_level = "High"

                    cols[0].metric("Status", label, delta="High Risk" if label=="FRAUD" else "Safe", delta_color="inverse")
                    cols[1].metric("Fraud Probability", f"{prob*100:.2f}%")
                    cols[2].metric("Risk Level", risk_level)
                    
                    st.caption(f"Model Version: {model_version} | Timestamp: {timestamp}")
                    
                    if risk_level == "High":
                        st.error(f"Risk Level: {risk_level}")
                    elif risk_level == "Medium":
                        st.warning(f"Risk Level: {risk_level}")
                    else:
                        st.success(f"Risk Level: {risk_level}")
                    
                    st.subheader("AI Explanation")
                    st.info(result.get("explanation", "No explanation provided."))
                    
                    if explainer is not None:
                        st.subheader("Feature Impact Visual")
                        data = np.array(features).reshape(1, -1)
                        sv = explainer.shap_values(data)
                        
                        fraud_shap = sv[0, :, 1] if len(sv.shape) == 3 else sv[0]
                        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values=fraud_shap,
                                base_values=base_value,
                                data=data[0],
                                feature_names=feature_labels
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    st.markdown("---")
                    if label == "FRAUD":
                        st.error("Fraudulent transaction detected")
                    else:
                        st.success("Transaction appears legitimate")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to the FastAPI server. Make sure it is running on port 8000. Error: {e}")

with tab2:
    st.markdown("### Transaction Database Logs")
    st.markdown("This tab shows recent prediction records for audit and monitoring purposes.")

    if st.button("Refresh Logs"):
        pass

    try:
        res = requests.get(f"{API_URL}/logs?limit=50")
        if res.status_code == 200:
            logs = res.json()
            if logs:
                df = pd.DataFrame(logs)

                # Add fallback columns if backend has not been updated yet
                if "risk_level" not in df.columns:
                    def derive_risk(prob):
                        if prob < 0.30:
                            return "Low"
                        elif prob < 0.70:
                            return "Medium"
                        return "High"
                    df["risk_level"] = df["fraud_probability"].apply(derive_risk)

                if "model_version" not in df.columns:
                    df["model_version"] = "v1.0"

                if "prediction" not in df.columns and "label" in df.columns:
                    df["prediction"] = df["label"]

                # Keep only the columns we want
                display_cols = [
                    "timestamp",
                    "prediction",
                    "fraud_probability",
                    "risk_level",
                    "model_version"
                ]
                df = df[display_cols]

                # Format values
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                df["fraud_probability"] = df["fraud_probability"].apply(lambda x: f"{x:.2%}")

                # Styling
                def color_prediction(val):
                    if str(val).upper() == "FRAUD":
                        return "color: red; font-weight: bold"
                    return "color: green; font-weight: bold"

                def color_risk(val):
                    if val == "High":
                        return "color: red; font-weight: bold"
                    elif val == "Medium":
                        return "color: orange; font-weight: bold"
                    return "color: green; font-weight: bold"

                styled_df = df.style.map(color_prediction, subset=["prediction"]).map(color_risk, subset=["risk_level"])

                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No logs found. Make a prediction first.")
        else:
            st.error(f"Failed to fetch logs. Status code: {res.status_code}")
    except Exception as e:
        st.error(f"Could not connect to the API to fetch logs: {e}")

with tab3:
    st.subheader("Model Performance")

    st.markdown("### Final Model Metrics")
    col1, col2 = st.columns(2)
    col1.metric("ROC-AUC", "0.97")
    col2.metric("Precision", "0.85")

    col3, col4 = st.columns(2)
    col3.metric("Recall", "0.84")
    col4.metric("F1-Score", "0.84")

    st.markdown("### Model Comparison")
    st.table({
        "Model": ["Logistic Regression", "Random Forest"],
        "ROC-AUC": [0.94, 0.97],
        "Precision": [0.81, 0.85],
        "Recall": [0.79, 0.84]
    })

    st.markdown("### ROC Curve")
    st.image("models/RandomForest_roc.png", use_container_width=True)

    st.markdown("### Why these metrics matter")
    st.write(
        "Because fraud cases are rare, accuracy is not a reliable metric. "
        "This project focuses on ROC-AUC, precision, and recall to better evaluate fraud-risk detection performance."
    )

import streamlit as st
import requests
import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("AI Credit Risk & Fraud Detection")

@st.cache_resource
def load_explainer():
    try:
        return joblib.load("models/shap_explainer.pkl")
    except:
        return None

explainer = load_explainer()

# Tabs
tab1, tab2 = st.tabs(["Make Prediction", "Audit Logs"])

with tab1:
    st.markdown("Use this dashboard to evaluate the fraud risk of a transaction.")
    
    st.sidebar.header("Transaction Features")
    st.sidebar.markdown("Enter the 30 standardized features for the transaction.")
    
    # Create input fields for 30 features
    features = []
    cols = st.sidebar.columns(2)
    for i in range(30):
        col = cols[i % 2]
        val = col.number_input(f"Feature {i}", value=0.0, step=0.1)
        features.append(val)
    
    if st.button("Predict Fraud Risk", type="primary"):
        with st.spinner("Analyzing transaction..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"features": features}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("Prediction Result")
                    cols = st.columns(3)
                    
                    label = result["label"]
                    prob = result["fraud_probability"]
                    
                    cols[0].metric("Status", label, delta="High Risk" if label=="FRAUD" else "Safe", delta_color="inverse")
                    cols[1].metric("Fraud Probability", f"{prob*100:.2f}%")
                    
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
                                feature_names=[f"Feature {i}" for i in range(30)]
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    st.markdown("---")
                    if label == "FRAUD":
                        st.error("This transaction was flagged as fraudulent.")
                    else:
                        st.success("This transaction appears legitimate.")
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to the FastAPI server. Make sure it is running on port 8000. Error: {e}")

with tab2:
    st.markdown("### Transaction Database Logs")
    st.markdown("This tab continuously fetches the latest predictions permanently logged to the SQLite database.")
    
    if st.button("Refresh Logs"):
        # Just triggers a rerun natively
        pass
        
    try:
        res = requests.get("http://127.0.0.1:8000/logs?limit=50")
        if res.status_code == 200:
            logs = res.json()
            if logs:
                df = pd.DataFrame(logs)
                # Reorder columns for UX
                df = df[["id", "timestamp", "label", "fraud_probability", "explanation", "prediction"]]
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Style the dataframe
                def color_fraud(val):
                    color = 'red' if val == 'FRAUD' else 'green'
                    return f'color: {color}; font-weight: bold'
                
                st.dataframe(df.style.map(color_fraud, subset=['label']), use_container_width=True)
            else:
                st.info("No logs found. Make a prediction first!")
        else:
            st.error("Failed to fetch logs.")
    except Exception as e:
        st.error(f"Could not connect to the API to fetch logs: {e}")

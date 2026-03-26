# AI Credit Risk & Fraud Detection Platform

A comprehensive machine learning platform designed to detect fraudulent credit card transactions, understand decisions through explainable AI (SHAP), and serve predictions through a fast backend, visualized on an interactive dashboard.

## Features
- **Machine Learning Models**: Employs Logistic Regression, Random Forest, and XGBoost with SMOTE handling for severe class imbalances.
- **Explainable AI**: SHAP (SHapley Additive exPlanations) is used to unpack "black-box" models, visually scoring exactly which features contributed to a fraud prediction.
- **Natural Language Explanations**: Integrates the **Google Gemini API** to translate numerical SHAP values into simple, transparent natural language explanations for human analysts.
- **FastAPI Backend**: Rapid and scalable prediction routing with automatically managed asynchronous capabilities.
- **Streamlit Dashboard**: A sleek, user-friendly UI for testing individual transactions on the fly.

## Project Structure
- `/src`: Contains the source code for data preprocessing, model training, and SHAP explanations.
- `/api`: The FastAPI inference server.
- `/app`: The Streamlit dashboard frontend.
- `/models`: Serialized machine learning models (`.pkl`) and saved visual artifacts.
- `/data`: Unprocessed source datasets.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd ai-credit-risk-platform
   ```

2. **Create a virtual environment** *(Recommended)*:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Training the Models & Generating Explanations

1. **Preprocess and train the models**:
   ```bash
   python3 src/train.py
   ```
2. **Generate the SHAP explainers and summary plots**:
   ```bash
   python3 src/shap_explain.py
   ```

## Running the Platform

To get the full dashboard experience, you will need to start your backend API and your frontend application.

### Start the FastAPI Backend
You will need a free API key from [Google AI Studio](https://aistudio.google.com/) for natural language SHAP explanations.

```bash
# Export the key
export GEMINI_API_KEY="your_api_key_here"

# Start the server (runs on http://127.0.0.1:8000)
python3 -m uvicorn api.main:app --reload
```

### Start the Streamlit Dashboard
Open a new terminal window:
```bash
# Start the frontend app (runs on http://localhost:8501)
streamlit run app/dashboard.py
```

## Further Improvements (Roadmap)
- Database logging (e.g. SQLite / Postgres) to monitor and track predictions over time.
- Dockerizing the application via a `docker-compose.yml`.
- Continuous Integration and Deployment (CI/CD) with Github Actions.

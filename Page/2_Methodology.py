import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="📈", layout="wide")
st.title("📈 Project Methodology")
st.write("This page details the technical implementation of the KUBERA risk model.")

st.subheader("1. Data Source")
st.markdown("""
* **Live Data:** The assessor tool uses the **Alpha Vantage API** to pull the latest annual balance sheet and income statement for any given stock ticker.
* **Training Data:** The model was trained on a hypothetical dataset of 10 sample companies.
* **Pro-Level Upgrade:** For a production-ready model, this dataset would be replaced with thousands of historical public filings (SEC 10-K/10-Q data) linked to public bankruptcy announcements.
""")

st.subheader("2. Feature Engineering")
st.markdown("""
The model does not use raw dollar amounts. It uses four key financial ratios, which are calculated in real-time from the API data:
* **Current Ratio:** `Current Assets / Current Liabilities` (Measures short-term liquidity)
* **Debt-to-Equity:** `Total Debt / Total Equity` (Measures leverage)
* **Interest Coverage Ratio:** `EBIT / Interest Expense` (Measures ability to pay interest)
* **Net Profit Margin:** `Net Income / Total Revenue` (Measures profitability)
""")

st.subheader("3. Model & Validation")
st.markdown("""
* **Model:** The core model is an **XGBoost Classifier**. This gradient-boosted tree model is the industry standard for performance on tabular data, as it's highly accurate and robust.
* **Validation (for this demo):** The model is trained on sample data for demonstration.
* **Pro-Level Validation:** A real-world model would be validated on a hold-out test set, with a focus on **AUC-ROC** (ability to separate classes) and **Recall** (ability to *find* the high-risk companies).
""")

st.subheader("4. Explainability (XAI)")
st.markdown("""
To ensure the model is not a "black box," this tool implements **SHAP (SHapley Additive exPlanations)**.
The force plot on the results page shows *exactly* which financial ratios contributed to the final risk score (pushing it higher or lower), providing a transparent and auditable decision.
""")
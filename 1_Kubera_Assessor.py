# --- 1. Import Libraries ---
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- 2. Set Page Configuration ---
st.set_page_config(
    page_title="Kubera Risk Assessor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Inject Special CSS ---
special_css = """
<style>
/* Main "Assess Risk" button */
.stButton>button {
    background-color: #004E9A; color: #FFFFFF; border: none;
    border-radius: 8px; padding: 10px 20px; font-size: 16px;
    font-weight: bold; box-shadow: 0 4px 14px 0 rgba(0, 78, 154, 0.3);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #003366;
    box-shadow: 0 6px 20px 0 rgba(0, 51, 102, 0.4);
}
/* Metric (the result box) */
[data-testid="stMetric"] {
    background-color: #FFFFFF; border: 1px solid #E0E0E0;
    border-radius: 10px; padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #333333;
}
[data-testid="stMetricLabel"] { color: #555555; }
</style>
"""
st.markdown(special_css, unsafe_allow_html=True)

# --- 4. Load ALL Saved Model Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('model/risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    explainer = joblib.load('model/explainer.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    return model, scaler, explainer, feature_names

model, scaler, explainer, feature_names = load_assets()

# --- 5. API Helper Function (FMP Version) ---
def get_financial_data(ticker, api_key):
    try:
        url_income = f'https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&limit=1&apikey={api_key}'
        r_income = requests.get(url_income)
        data_income = r_income.json()[0]
        url_balance = f'https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&limit=1&apikey={api_key}'
        r_balance = requests.get(url_balance)
        data_balance = r_balance.json()[0]
        
        data = {
            "net_income": float(data_income.get('netIncome', 0)),
            "total_revenue": float(data_income.get('revenue', 0)),
            "ebit": float(data_income.get('ebitda', 0)) - float(data_income.get('depreciationAndAmortization', 0)),
            "interest_expense": float(data_income.get('interestExpense', 1e-6)),
            "current_assets": float(data_balance.get('totalCurrentAssets', 0)),
            "current_liabilities": float(data_balance.get('totalCurrentLiabilities', 0)),
            "total_debt": float(data_balance.get('longTermDebt', 0)) + float(data_balance.get('shortTermDebt', 0)),
            "total_equity": float(data_balance.get('totalStockholdersEquity', 0))
        }
        
        if data["interest_expense"] == 0: data["interest_expense"] = 1e-6
        return data, data_income.get('fillingDate', 'N/A')
        
    except Exception as e:
        print(f"API Error: {e}")
        return None, None

# --- 6. Create the App Header ---
st.image("static/logo.png", width=150) # Using the new 'logo.png' name
st.title("KUBERA: Corporate Risk Assessment")
st.write("Enter a company's stock ticker to predict its bankruptcy risk.")

# --- 7. Create the "Smart" Sidebar Input Form ---
st.sidebar.header("Enter Ticker & API Key:")
ticker = st.sidebar.text_input("Company Ticker (e.g., AAPL)").upper()
api_key = "" # Initialize api_key

try:
    # Try to get the key from Streamlit's secrets (for deployed app)
    api_key = st.secrets['FMP_KEY']
    st.sidebar.success("API Key loaded from Secrets!", icon="✅")
except:
    # If it fails (we are local), show the text box
    st.sidebar.markdown("""
    [Get your free FMP API key](https://site.financialmodelingprep.com/developer)
    """)
    api_key = st.sidebar.text_input("Financial Modeling Prep API Key", type="password")

# --- 8. Create the "Assess Risk" Button and Logic ---
if st.button("Assess Risk"):
    if not api_key or not ticker:
        st.warning("Please enter both a Ticker and an API Key.")
    else:
        with st.spinner(f"Fetching financial data for {ticker}..."):
            raw_data, report_date = get_financial_data(ticker, api_key)
        
        if raw_data is None:
            st.error("Could not fetch data. Check the Ticker or API Key. (Note: Free API is limited).")
        else:
            st.success(f"Successfully fetched data from report date: {report_date}")
            
            # 2. Calculate Ratios
            current_ratio = raw_data['current_assets'] / (raw_data['current_liabilities'] + 1e-6)
            debt_to_equity = raw_data['total_debt'] / (raw_data['total_equity'] + 1e-6)
            interest_coverage_ratio = raw_data['ebit'] / (raw_data['interest_expense'] + 1e-6)
            net_profit_margin = raw_data['net_income'] / (raw_data['total_revenue'] + 1e-6)

            # 3. Prepare Data for Model
            features = np.array([current_ratio, debt_to_equity, interest_coverage_ratio, net_profit_margin])
            scaled_features = scaler.transform(features.reshape(1, -1))

            # 4. Make Prediction
            prediction_prob = model.predict_proba(scaled_features)
            prob_of_default = prediction_prob[0][1]
            
            # --- 5. Display the Result ---
            st.subheader(f"Risk Assessment Result for {ticker}:")
            st.metric(label="Probability of Bankruptcy", value=f"{prob_of_default * 100:.2f}%")

            if prob_of_default > 0.5: st.error("Risk Level: High Risk")
            elif prob_of_default > 0.2: st.warning("Risk Level: Moderate Risk")
            else: st.success("Risk Level: Low Risk")

            with st.expander("Show Calculated Ratios"):
                st.write(f"Current Ratio: {current_ratio:.2f}")
                st.write(f"Debt-to-Equity: {debt_to_equity:.2f}")
                st.write(f"Interest Coverage Ratio: {interest_coverage_ratio:.2f}")
                st.write(f"Net Profit Margin: {net_profit_margin:.2f}")

            # --- 6. SHAP Explainability Plot (FINAL FIX) ---
            st.subheader("Why did the model decide this?")
            st.write("This plot shows which features contributed to the final risk score.")
            
            scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)
            shap_values_object = explainer(scaled_features_df)
            
            shap_values_for_class_1 = shap_values_object.values[0,:,1]
            base_value_for_class_1 = shap_values_object.base_values[0,1]

            # Create the plot as an HTML object
            plot_html = shap.force_plot(
                base_value=base_value_for_class_1,
                shap_values=shap_values_for_class_1,
                features=scaled_features_df
            )
            
            # Render the HTML object using st.components.v1.html
            components.html(str(plot_html.data), height=150, scrolling=True)
            
            st.caption("These are the SHAP values for the 'Probability of Bankruptcy' (Class 1). Features pushing the score higher (to 'High Risk') are in red. Features pushing lower are in blue.")
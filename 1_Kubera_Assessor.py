# --- 1. Import Libraries ---
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt

# --- 2. Set Page Configuration ---
st.set_page_config(
    page_title="Kubera Risk Assessor",
    page_icon="", # You can use a fire emoji or your logo path
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Inject Special CSS ---
# This CSS is for the premium button and metric box styles
special_css = """
<style>
/* Main "Assess Risk" button */
.stButton>button {
    background-color: #004E9A; /* A premium, deep blue */
    color: #FFFFFF;
    border: none;
    border-radius: 8px; /* Rounded corners */
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    box-shadow: 0 4px 14px 0 rgba(0, 78, 154, 0.3); /* A subtle shadow */
    transition: all 0.3s ease; /* Smooth hover effect */
}
.stButton>button:hover {
    background-color: #003366; /* Darker blue on hover */
    box-shadow: 0 6px 20px 0 rgba(0, 51, 102, 0.4);
}

/* Metric (the result box) */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Soft, premium shadow */
    color: #333333; /* Fixes faint text */
}

/* Fixes metric label color */
[data-testid="stMetricLabel"] {
    color: #555555;
}
</style>
"""
st.markdown(special_css, unsafe_allow_html=True)


# --- 4. Load ALL Saved Model Assets ---
@st.cache_resource
def load_assets():
    print("Loading all model assets...")
    model = joblib.load('model/risk_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    explainer = joblib.load('model/explainer.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    print("Assets loaded.")
    return model, scaler, explainer, feature_names

model, scaler, explainer, feature_names = load_assets()

# --- 5. API Helper Function ---
# This function calls the Alpha Vantage API
def get_financial_data(ticker, api_key):
    try:
        # 1. Get Income Statement
        url_income = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}'
        r_income = requests.get(url_income)
        data_income = r_income.json()
        
        # 2. Get Balance Sheet
        url_balance = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}'
        r_balance = requests.get(url_balance)
        data_balance = r_balance.json()
        
        # 3. Extract the 8 numbers we need
        report_income = data_income['annualReports'][0]
        report_balance = data_balance['annualReports'][0]
        
        data = {
            "net_income": float(report_income.get('netIncome', 0)),
            "total_revenue": float(report_income.get('totalRevenue', 0)),
            "ebit": float(report_income.get('ebit', 0)),
            "interest_expense": float(report_income.get('interestExpense', 1e-6)), # Default to tiny number
            "current_assets": float(report_balance.get('totalCurrentAssets', 0)),
            "current_liabilities": float(report_balance.get('totalCurrentLiabilities', 0)),
            "total_debt": float(report_balance.get('longTermDebt', 0)) + float(report_balance.get('shortTermDebt', 0)),
            "total_equity": float(report_balance.get('totalShareholderEquity', 0))
        }
        
        # Handle cases where interest expense might be zero
        if data["interest_expense"] == 0:
            data["interest_expense"] = 1e-6
            
        return data, report_income.get('fiscalDateEnding', 'N/A')
        
    except Exception as e:
        print(f"API Error: {e}")
        return None, None

# --- 6. Create the App Header ---
st.image("static/Kubera_logo.png", width=150) # Make sure your logo is named this
st.title("KUBERA: Corporate Risk Assessment")
st.write("Enter a company's stock ticker to predict its bankruptcy risk.")

# --- 7. Create the "Smart" Sidebar Input Form ---
st.sidebar.header("Enter Ticker & API Key:")
ticker = st.sidebar.text_input("Company Ticker (e.g., AAPL)").upper()

# This is the function to check if we're deployed or local
def check_if_secrets_exist():
    # hasattr(st, 'secrets') checks if the st.secrets feature exists
    # and st.secrets.get('ALPHA_VANTAGE_KEY') checks if our key is set
    return hasattr(st, 'secrets') and st.secrets.get('ALPHA_VANTAGE_KEY')

# Check if we are running on Streamlit Cloud (deployed)
if check_if_secrets_exist():
    st.sidebar.success("API Key loaded from Secrets!", icon="✅")
    api_key = st.secrets['ALPHA_VANTAGE_KEY']
else:
    # Otherwise, show the text box for local testing
    st.sidebar.markdown("""
    [Get your free Alpha Vantage API key](https://www.alphavantage.co/support/#api-key)
    """)
    api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
# --- 8. Create the "Assess Risk" Button and Logic ---
if st.button("Assess Risk"):
    if not api_key or not ticker:
        st.warning("Please enter both a Ticker and an API Key.")
    else:
        with st.spinner(f"Fetching financial data for {ticker}..."):
            raw_data, report_date = get_financial_data(ticker, api_key)
        
        if raw_data is None:
            st.error("Could not fetch data. Check the Ticker or API Key. (Note: Free API is limited to 5 calls/min).")
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

            if prob_of_default > 0.5:
                st.error("Risk Level: High Risk")
            elif prob_of_default > 0.2:
                st.warning("Risk Level: Moderate Risk")
            else:
                st.success("Risk Level: Low Risk")

            with st.expander("Show Calculated Ratios"):
                st.write(f"Current Ratio: {current_ratio:.2f}")
                st.write(f"Debt-to-Equity: {debt_to_equity:.2f}")
                st.write(f"Interest Coverage Ratio: {interest_coverage_ratio:.2f}")
                st.write(f"Net Profit Margin: {net_profit_margin:.2f}")

            # --- 6. SHAP Explainability Plot (FINAL BUG FIX) ---
            st.subheader("Why did the model decide this?")
            st.write("This plot shows which features contributed to the final risk score.")
            
            # Convert scaled features to a DataFrame for the explainer
            scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)
            
            # Get SHAP values (this returns a list: [class_0_values, class_1_values])
            shap_values_list = explainer(scaled_features_df)
            
            # We want the values for Class 1 (Bankruptcy)
            shap_values_for_class_1 = shap_values_list[1][0,:]
            base_value_for_class_1 = explainer.expected_value[1]

            # Create the force plot
            fig, ax = plt.subplots()
            shap.force_plot(
                base_value=base_value_for_class_1,
                shap_values=shap_values_for_class_1,
                features=scaled_features_df,
                matplotlib=True,
                show=False
            )
            st.pyplot(fig, bbox_inches='tight')
            st.caption("These are the SHAP values for the 'Probability of Bankruptcy' (Class 1). Features pushing the score higher (to 'High Risk') are in red. Features pushing lower are in blue.")
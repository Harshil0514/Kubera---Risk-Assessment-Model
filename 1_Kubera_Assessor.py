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
st.image("static/logo.png", width=150)
st.title("KUBERA: Corporate Risk Assessment")
st.write("Assess a company's bankruptcy risk using a live stock ticker or by entering financial data manually.")

# --- 7. Create Tabs for Ticker vs. Manual ---
tab1, tab2 = st.tabs(["Assess by Ticker", "Assess Manually (for Small Business)"])

# --- TAB 1: ASSESS BY TICKER (Our existing app) ---
with tab1:
    st.sidebar.header("Assess by Ticker")
    ticker = st.sidebar.text_input("Company Ticker (e.g., AAPL)").upper()
    api_key = "" # Initialize api_key

    try:
        api_key = st.secrets['FMP_KEY']
        st.sidebar.success("API Key loaded from Secrets!", icon="✅")
    except:
        st.sidebar.markdown("""
        [Get your free FMP API key](https://site.financialmodelingprep.com/developer)
        """)
        api_key = st.sidebar.text_input("Financial Modeling Prep API Key", type="password")

    if st.button("Assess Ticker Risk"):
        if not api_key or not ticker:
            st.warning("Please enter both a Ticker and an API Key.")
        else:
            with st.spinner(f"Fetching financial data for {ticker}..."):
                raw_data, report_date = get_financial_data(ticker, api_key)
            
            if raw_data is None:
                st.error("Could not fetch data. Check the Ticker or API Key. (Note: Free API is limited).")
            else:
                st.success(f"Successfully fetched data from report date: {report_date}")
                
                # --- 2. Calculate Ratios ---
                current_ratio = raw_data['current_assets'] / (raw_data['current_liabilities'] + 1e-6)
                debt_to_equity = raw_data['total_debt'] / (raw_data['total_equity'] + 1e-6)
                interest_coverage_ratio = raw_data['ebit'] / (raw_data['interest_expense'] + 1e-6)
                net_profit_margin = raw_data['net_income'] / (raw_data['total_revenue'] + 1e-6)

                # --- 3. Prepare Data for Model & Plot ---
                unscaled_data = {
                    'current_ratio': [round(current_ratio, 2)],
                    'debt_to_equity': [round(debt_to_equity, 2)],
                    'interest_coverage_ratio': [round(interest_coverage_ratio, 2)],
                    'net_profit_margin': [round(net_profit_margin, 2)]
                }
                unscaled_features_df = pd.DataFrame(unscaled_data, columns=feature_names)
                
                unrounded_features = np.array([current_ratio, debt_to_equity, interest_coverage_ratio, net_profit_margin])
                scaled_features = scaler.transform(unrounded_features.reshape(1, -1))
                scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)

                # --- 4. Make Prediction ---
                prediction_prob = model.predict_proba(scaled_features_df)
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

                # --- 6. SHAP Explainability Plot ---
                st.subheader("Why did the model decide this?")
                st.write("This plot shows which features contributed to the final risk score.")
                
                shap_values_object = explainer(scaled_features_df)
                shap_values_for_class_1 = shap_values_object.values[0,:,1]
                base_value_for_class_1 = shap_values_object.base_values[0,1]
                
                plt.rcParams.update({'font.size': 8.5})
                
                fig = shap.force_plot(
                    base_value=base_value_for_class_1,
                    shap_values=shap_values_for_class_1,
                    features=unscaled_features_df.iloc[0], 
                    feature_names=unscaled_features_df.columns,
                    matplotlib=True,
                    show=False,
                    figsize=(14, 4)
                )
                
                plt.tight_layout()
                st.pyplot(fig, bbox_inches='tight', pad_inches=0.1) 
                plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
                plt.close(fig) 
                
                st.caption("These are the SHAP values...")

# --- TAB 2: ASSESS MANUALLY (YOUR NEW, BETTER VERSION) ---
with tab2:
    st.subheader("Manual Risk Assessment (for Small Business)")
    st.write("Enter the company's most recent *annual* financial data. The model will calculate the ratios and predict the risk.")

    with st.form("manual_form"):
        st.markdown("##### 1. Profitability")
        col1, col2 = st.columns(2)
        with col1:
            total_revenue = st.number_input("Total Revenue (Sales)", min_value=0.0, value=500000.0, step=1000.0, format="%.2f")
            net_income = st.number_input("Net Income (Profit)", min_value=-500000.0, value=50000.0, step=1000.0, format="%.2f")
        with col2:
            ebit = st.number_input("EBIT (Earnings Before Interest & Taxes)", min_value=-500000.0, value=75000.0, step=1000.0, format="%.2f")
            interest_expense = st.number_input("Interest Expense", min_value=0.0, value=5000.0, step=100.0, format="%.2f")

        st.markdown("##### 2. Liquidity & Leverage")
        col3, col4 = st.columns(2)
        with col3:
            current_assets = st.number_input("Total Current Assets", min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
            current_liabilities = st.number_input("Total Current Liabilities", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        with col4:
            total_debt = st.number_input("Total Debt (Short + Long-term)", min_value=0.0, value=75000.0, step=1000.0, format="%.2f")
            total_equity = st.number_input("Total Equity", min_value=1.0, value=150000.0, step=1000.0, format="%.2f") # Min 1.0 to avoid divide-by-zero

        submitted = st.form_submit_button("Assess Manual Risk")

    if submitted:
        # --- 1. Calculate Ratios from Manual Inputs ---
        current_ratio = current_assets / (current_liabilities + 1e-6)
        debt_to_equity = total_debt / (total_equity + 1e-6)
        interest_coverage_ratio = ebit / (interest_expense + 1e-6)
        net_profit_margin = net_income / (total_revenue + 1e-6)
            
        # --- 2. Prepare Data for Model & Plot ---
        unscaled_data = {
            'current_ratio': [round(current_ratio, 2)],
            'debt_to_equity': [round(debt_to_equity, 2)],
            'interest_coverage_ratio': [round(interest_coverage_ratio, 2)],
            'net_profit_margin': [round(net_profit_margin, 2)]
        }
        unscaled_features_df = pd.DataFrame(unscaled_data, columns=feature_names)
        
        unrounded_features = np.array([current_ratio, debt_to_equity, interest_coverage_ratio, net_profit_margin])
        scaled_features = scaler.transform(unrounded_features.reshape(1, -1))
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)

        # --- 3. Make Prediction ---
        prediction_prob = model.predict_proba(scaled_features_df)
        prob_of_default = prediction_prob[0][1]
        
        # --- 4. Display the Result ---
        st.subheader("Risk Assessment Result (Manual):")
        st.metric(label="Probability of Bankruptcy", value=f"{prob_of_default * 100:.2f}%")

        if prob_of_default > 0.5: st.error("Risk Level: High Risk")
        elif prob_of_default > 0.2: st.warning("Risk Level: Moderate Risk")
        else: st.success("Risk Level: Low Risk")

        with st.expander("Show Calculated Ratios"):
            st.write(f"Current Ratio: {current_ratio:.2f}")
            st.write(f"Debt-to-Equity: {debt_to_equity:.2f}")
            st.write(f"Interest Coverage Ratio: {interest_coverage_ratio:.2f}")
            st.write(f"Net Profit Margin: {net_profit_margin:.2f}")

        # --- 5. SHAP Explainability Plot ---
        st.subheader("Why did the model decide this?")
        st.write("This plot shows which features contributed to the final risk score.")
        
        shap_values_object = explainer(scaled_features_df)
        shap_values_for_class_1 = shap_values_object.values[0,:,1]
        base_value_for_class_1 = shap_values_object.base_values[0,1]
        
        plt.rcParams.update({'font.size': 8.5})
        
        fig = shap.force_plot(
            base_value=base_value_for_class_1,
            shap_values=shap_values_for_class_1,
            features=unscaled_features_df.iloc[0], 
            feature_names=unscaled_features_df.columns,
            matplotlib=True,
            show=False,
            figsize=(14, 4)
        )
        
        plt.tight_layout()
        st.pyplot(fig, bbox_inches='tight', pad_inches=0.1) 
        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
        plt.close(fig) 
        
        st.caption("These are the SHAP values...")
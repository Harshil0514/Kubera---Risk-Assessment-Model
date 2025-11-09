# --- 1. Import Libraries ---
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# --- 2. Set Page Configuration ---
st.set_page_config(
    page_title="High-Accuracy Risk Assessor",
    page_icon="🔬",
    layout="wide"
)

# --- 3. Inject Special CSS (Same as main page) ---
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

# --- 4. Load ALL 'model_real' Assets ---
@st.cache_resource
def load_real_assets():
    try:
        model = joblib.load('model_real/risk_model_real.pkl')
        scaler = joblib.load('model_real/scaler_real.pkl')
        explainer = joblib.load('model_real/explainer_real.pkl')
        feature_names = joblib.load('model_real/feature_names_real.pkl')
        return model, scaler, explainer, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run `python train_real_model.py` first.")
        return None, None, None, None

model, scaler, explainer, feature_names = load_real_assets()

# --- 5. Create the App Header ---
st.image("static/logo.png", width=150)
st.title("KUBERA: High-Accuracy Risk Assessor")
st.write("This tool uses a machine learning model trained on real-world data (Polish companies) to predict bankruptcy risk based on the **Top 10 Most Important Financial Ratios**.")

# --- 6. Create the Manual Input Form ---
if model is not None:
    st.subheader("Enter Your Company's Financial Ratios:")
    
    with st.form("real_data_form"):
        # We will store the user's inputs in this dictionary
        inputs = {}
        
        # Create 5 rows of 2 columns for the 10 inputs
        cols = st.columns(2)
        
        for i, feature in enumerate(feature_names):
            current_col = cols[i % 2] 
            
            # Create a user-friendly label
            label = feature.replace("_", " ").replace("Attr", "Feature ").title()
            # NEW: Manually change "Yuan" to "Dollar" for a cleaner look
            label = label.replace("Revenue Per Share (Yuan ¥)", "Revenue Per Share (Dollar $)")
            
            # Store the user's input using the ORIGINAL feature name
            inputs[feature] = current_col.number_input(label, value=0.0, step=0.01, format="%.4f")

        submitted = st.form_submit_button("Assess Real Data Risk")
    if submitted:
        # --- 1. Prepare Data ---
        # Convert the dictionary of inputs into a 2D numpy array in the correct order
        input_values = np.array([inputs[feature] for feature in feature_names])
        
        # Create a DataFrame for the SHAP plot (with nice rounding)
        unscaled_features_df = pd.DataFrame([inputs])
        for col in unscaled_features_df.columns:
            unscaled_features_df[col] = unscaled_features_df[col].round(4)
        
        # Scale the data for the model
        scaled_features = scaler.transform(input_values.reshape(1, -1))
        scaled_features_df = pd.DataFrame(scaled_features, columns=feature_names)

        # --- 2. Make Prediction ---
        prediction_prob = model.predict_proba(scaled_features_df)
        prob_of_default = prediction_prob[0][1] # Probability of Class 1 (Bankruptcy)
        
        # --- 3. Display the Result ---
        st.subheader("High-Accuracy Risk Assessment Result:")
        st.metric(label="Probability of Bankruptcy", value=f"{prob_of_default * 100:.2f}%")

        if prob_of_default > 0.5: st.error("Risk Level: High Risk")
        elif prob_of_default > 0.2: st.warning("Risk Level: Moderate Risk")
        else: st.success("Risk Level: Low Risk")

        # --- 4. SHAP Explainability Plot ---
        st.subheader("Why did the model decide this?")
        st.write("This plot shows which features contributed to the final risk score.")
        
        shap_values_object = explainer(scaled_features_df.values)
        shap_values_for_class_1 = shap_values_object.values[0,:,1]
        base_value_for_class_1 = shap_values_object.base_values[0,1]
        
        # --- CHANGES START HERE ---
        
        # 1. Set a smaller font size *before* creating the plot
        plt.rcParams.update({'font.size': 8}) # Slightly smaller for labels

       # Create a new, clean list of feature names for the plot
        clean_plot_labels = []
        for f in unscaled_features_df.columns:
            label = f.replace("_", " ").replace("Attr", "Feature ").title()
            label = label.replace("Revenue Per Share (Yuan ¥)", "Revenue Per Share (Dollar $)")
            clean_plot_labels.append(label)

        fig = shap.force_plot(
            base_value=base_value_for_class_1,
            shap_values=shap_values_for_class_1,
            features=unscaled_features_df.iloc[0], 
            # Use our new clean list for the plot labels
            feature_names=clean_plot_labels,
            matplotlib=True,
            show=False,
            figsize=(18, 6),
            text_rotation=15
        )
        # Get the current axes to rotate labels
        ax = fig.gca() 
        
        # Rotate x-axis (feature value) labels
        ax.set_xticks(ax.get_xticks()) # Keep current ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right') # Rotate by 30 degrees, right align

        plt.tight_layout() # Ensures everything fits
        st.pyplot(fig, bbox_inches='tight', pad_inches=0.1) 
        
        # Reset font size to default
        plt.rcParams.update({'font.size': plt.rcParamsDefault['font.size']})
        plt.close(fig) 
        
        # --- CHANGES END HERE ---
        
        st.caption("These are the SHAP values...")
else:
    st.error("Model assets are missing. Please run `python train_real_model.py` and refresh.")
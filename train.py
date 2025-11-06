# --- 1. Import Libraries ---
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import xgboost as xgb
import shap
import numpy as np

print("Starting model training with REALISTIC data...")

# --- 2. Generate REALISTIC Sample Data (5,000 samples) ---
np.random.seed(42)
num_samples = 5000

# Generate data for "Healthy" companies (Class 0)
healthy_current_ratio = np.random.normal(loc=2.5, scale=0.8, size=num_samples // 2)
healthy_debt_to_equity = np.random.normal(loc=0.5, scale=0.3, size=num_samples // 2)
healthy_interest_coverage = np.random.normal(loc=8.0, scale=2.0, size=num_samples // 2)
healthy_net_profit_margin = np.random.normal(loc=0.15, scale=0.05, size=num_samples // 2)
healthy_went_bankrupt = np.zeros(num_samples // 2, dtype=int)

# Generate data for "At-Risk" companies (Class 1)
at_risk_current_ratio = np.random.normal(loc=1.0, scale=0.5, size=num_samples // 2)
at_risk_debt_to_equity = np.random.normal(loc=2.0, scale=0.7, size=num_samples // 2)
at_risk_interest_coverage = np.random.normal(loc=1.5, scale=1.0, size=num_samples // 2)
at_risk_net_profit_margin = np.random.normal(loc=-0.05, scale=0.05, size=num_samples // 2)
at_risk_went_bankrupt = np.ones(num_samples // 2, dtype=int)

# Combine the data
data = {
    'current_ratio': np.concatenate([healthy_current_ratio, at_risk_current_ratio]),
    'debt_to_equity': np.concatenate([healthy_debt_to_equity, at_risk_debt_to_equity]),
    'interest_coverage_ratio': np.concatenate([healthy_interest_coverage, at_risk_interest_coverage]),
    'net_profit_margin': np.concatenate([healthy_net_profit_margin, at_risk_net_profit_margin]),
    'went_bankrupt': np.concatenate([healthy_went_bankrupt, at_risk_went_bankrupt])
}
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the data

print(f"Generated {len(df)} realistic training samples.")

# --- 3. Define Features (X) and Target (y) ---
features = ['current_ratio', 'debt_to_equity', 'interest_coverage_ratio', 'net_profit_margin']
target = 'went_bankrupt'

X = df[features]
y = df[target]

# --- 4. Scale the Features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. Train the XGBoost Model ---
model = xgb.XGBClassifier(base_score=0.5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_scaled, y)
print("XGBoost model has been trained successfully on 5,000 samples.")

# --- 6. Create and Save SHAP Explainer ---
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
explainer = shap.Explainer(model.predict_proba, X_scaled_df.sample(200)) # Use 200 samples for explainer background
print("SHAP explainer created successfully.")

# --- 7. Save Everything to the 'model/' folder ---
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/risk_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(explainer, 'model/explainer.pkl')
joblib.dump(features, 'model/feature_names.pkl')

print("All new model assets saved to 'model/' folder.")
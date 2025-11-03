# --- 1. Import Libraries ---
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import xgboost as xgb
import shap
import numpy as np

print("Starting model training with XGBoost and SHAP...")

# --- 2. Create Sample Data ---
data = {
    'current_ratio': [2.5, 1.1, 0.8, 3.0, 1.5, 0.5, 2.2, 1.9, 0.9, 2.8],
    'debt_to_equity': [0.5, 1.8, 2.5, 0.3, 1.2, 3.0, 0.8, 1.0, 2.2, 0.4],
    'interest_coverage_ratio': [8.0, 2.1, 0.9, 10.0, 3.5, -1.0, 6.0, 4.0, 1.5, 9.0],
    'net_profit_margin': [0.15, 0.02, -0.05, 0.20, 0.08, -0.10, 0.12, 0.09, 0.01, 0.18],
    'went_bankrupt': [0, 1, 1, 0, 0, 1, 0, 0, 1, 0] # 0=No, 1=Yes
}
df = pd.DataFrame(data)

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
model.fit(X_scaled, y) # Train the new model
print("XGBoost model has been trained successfully.")

# --- 6. Create and Save SHAP Explainer (FINAL BUG FIX) ---
# We convert our scaled data to a DataFrame, as shap expects.
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
# We pass the model's PREDICTION FUNCTION (model.predict_proba)
# This is the most robust way and avoids all compatibility bugs.
explainer = shap.Explainer(model.predict_proba, X_scaled_df)
print("SHAP explainer created successfully.")

# --- 7. Save Everything to the 'model/' folder ---
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/risk_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(explainer, 'model/explainer.pkl')
joblib.dump(features, 'model/feature_names.pkl')

print("All model assets saved to 'model/' folder.")
# --- 1. Import Libraries ---
import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import warnings

warnings.filterwarnings('ignore')

print("Starting FINAL high-accuracy model training...")

# --- 2. Load and Clean Real-World Data ---
try:
    df = pd.read_csv('data/data.csv')
except FileNotFoundError:
    print("Error: data.csv not found in 'data/' folder.")
    exit()

print(f"Loaded {len(df)} company records.")

# Data cleaning
df = df.replace('?', np.nan)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.fillna(df.mean())

# --- 3. Define Features (X) and Target (y) ---
target = 'Bankrupt?'
features = [col for col in df.columns if col != target]
X = df[features]
y = df[target]

# --- 4. Handle Imbalanced Data ---
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# --- 5. Train a FIRST model on ALL 95 features ---
print("Training initial model on 95 features to find importance...")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler_full = StandardScaler()
X_train_scaled_full = scaler_full.fit_transform(X_train_full)

initial_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)
initial_model.fit(X_train_scaled_full, y_train_full)

# --- 6. Get Top 10 Most Important Features ---
print("Finding Top 10 features...")
importances = initial_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
top_10_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)['feature'].tolist()

print("--- Top 10 Most Important Features ---")
for i, feature in enumerate(top_10_features):
    print(f"{i+1}. {feature}")

# --- 7. Create NEW dataset with only Top 10 features ---
X_top10 = df[top_10_features]
y_top10 = df[target]

# --- 8. Create FINAL Train/Test Split on Top 10 Data ---
X_train, X_test, y_train, y_test = train_test_split(X_top10, y_top10, test_size=0.2, random_state=42, stratify=y_top10)

# --- 9. Create and fit a NEW Scaler for Top 10 Data ---
scaler_top10 = StandardScaler()
X_train_scaled = scaler_top10.fit_transform(X_train)
X_test_scaled = scaler_top10.transform(X_test)

# --- 10. Train FINAL model on Top 10 Features ---
print("\nTraining FINAL model on Top 10 features...")
final_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, 
    use_label_encoder=False, 
    eval_metric='logloss',
    random_state=42
)
final_model.fit(X_train_scaled, y_train)
print("Final model trained successfully.")

# --- 11. Evaluate the FINAL Model ---
print("\n--- FINAL Model Performance (Top 10 Features) ---")
y_pred = final_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print(f"  Accuracy: {accuracy * 100:.2f}%")
print(f" Precision: {precision * 100:.2f}%")
print(f"    Recall: {recall * 100:.2f}%")
print(classification_report(y_test, y_pred))

# --- 12. Create and Save FINAL SHAP Explainer ---
print("Creating final SHAP explainer...")
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=top_10_features)
final_explainer = shap.Explainer(final_model.predict_proba, X_train_scaled_df.sample(200, replace=True))
print("SHAP explainer created successfully.")

# --- 13. Save Everything to 'model_real/' folder ---
os.makedirs('model_real', exist_ok=True)
joblib.dump(final_model, 'model_real/risk_model_real.pkl')
joblib.dump(scaler_top10, 'model_real/scaler_real.pkl')
joblib.dump(top_10_features, 'model_real/feature_names_real.pkl')
joblib.dump(final_explainer, 'model_real/explainer_real.pkl')

print("\nAll FINAL model assets (Top 10) saved to 'model_real/' folder.")
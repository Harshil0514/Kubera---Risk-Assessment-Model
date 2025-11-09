KUBERA: A Dual-Model Corporate Risk Assessor

View Live Application
This project is a multi-page Streamlit web application that assesses corporate bankruptcy risk using two distinct machine learning models.

 1) API-Driven Assessor: A real-time model for public US companies using live data from the Financial Modeling Prep API.
 2) High-Accuracy Assessor: A high-performance model trained on real-world financial data from over 6,800 companies.

Key Features Multi-Page Streamlit App: A clean, professional user interface built with Streamlit's multi-page functionality.

1) Model 1: Real-Time API Model:
  1) Connects to the Financial Modeling Prep (FMP) API for live financial data.
  2) Takes any valid US stock ticker (e.g., AAPL, TSLA) as input.
  3) Provides a rapid risk assessment based on key financial ratios.

2) Model 2: High-Accuracy Manual Model:
  1) A robust XGBoost model trained on the Kaggle "Company Bankruptcy Prediction" dataset.
  2) Features a Top-10 Feature Form for manual data entry, making the tool usable for small businesses or private companies.
 
Explainable AI (XAI):
  1) Both models feature SHAP (SHapley Additive exPlanations) force plots.
  2) This makes the model's decisions transparent by showing exactly which features (e.g., Net Profit Margin, Debt Ratio) pushed the risk score higher or lower.
 
🛠️ Technical Stack:
  1) Language: Python 3.13
  2) Data Science: scikit-learn (StandardScaler), pandas
  3) Machine Learning: XGBoost (XGBClassifier)
  4) Explainability: shap
  5) Web Framework: Streamlit
  5) Plotting: matplotlib
  6) API: requests
  7) Deployment: Streamlit Cloud, Git/GitHub
 
🔬 High-Accuracy Model: A Deeper Dive

  The centerpiece of this project is the "Real Data Assessor," which is trained on a real-world dataset of 6,819 Polish companies.
  
  1) Feature Selection -
     The initial dataset contained 95 different financial ratios. A 95-input form is unusable

     To solve this, I first trained an XGBoost model on all 95 features. I then extracted the feature_importances_ to identify the Top 10 most predictive features. A new, final model          was then trained only on these 10 features, resulting in a model that is both highly accurate and efficient.

  The Top 10 Features were:
      1) Continuous Interest Rate (After Tax)
      2) Total Debt/Total Net Worth
      3) Debt Ratio %
      4) Persistent EPS in the Last Four Seasons
      5) Borrowing Dependency
      6) Net Value Per Share (C)
      7) Interest Expense Ratio
      8) Revenue Per Share (Yuan ¥)
      9) Operating Profit Rate
      10) Retained Earnings to Total Assets

2) Model Performance -
        
      The final model was evaluated on a 20% hold-out test set. The performance proves its ability to find complex, non-linear patterns in financial data.
        
     Metric     Score     Business Implication
     Accuracy   95.97%    The model is correct 96% of the time overall.
     Precision  50.88%    When the model predicts bankruptcy, it's correct 51% of the time.
     Recall     40.91%    The model successfully catches 41% of all actual bankruptcies.
      
     
3) Performance Analysis -

      The most important metric for a risk model is Recall, as failing to predict a bankruptcy (a false negative) is the most costly business error. A Recall of 41% is a strong                 baseline. The next step for this project would be to tune hyperparameters specifically to optimize for higher Recall (e.g., by adjusting scale_pos_weight), even if it means               slightly lowering the overall 96% accuracy. This shows a "business-first" approach to model optimization.

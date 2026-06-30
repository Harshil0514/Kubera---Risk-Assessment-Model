# KUBERA: Enterprise-Grade Corporate Bankruptcy Risk Assessor

Kubera is a production-ready, dual-engine financial risk software application engineered to automate corporate bankruptcy prediction. The core thesis of this system is solving the **"Black Box" problem** in financial artificial intelligence. By unifying high-performance gradient-boosted decision trees, cooperative game-theoretic Explainable AI (XAI), and real-time streaming REST data pipelines, Kubera eliminates the industry's reliance on opaque, non-compliant prediction frameworks.

---

## 🛠️ Production Tech Stack & Architecture

* **Language Environment:** Python 3.11
* **Predictive Frameworks:** XGBoost Open-Source (Extreme Gradient Boosting), Scikit-Learn
* **Explainable AI Engine:** SHAP (SHapley Additive exPlanations)
* **Interface Architecture:** Streamlit Framework (Stateful multi-page routing)
* **Data Engineering & Network Layer:** Pandas Vectorized Pipelines, NumPy Linear Algebra, Upstream HTTP `Requests` Client
* **Infrastructure & Security:** Git, Streamlit Cloud Enterprise Staging, Secure Environment Variables via `.streamlit/secrets.toml`

---

## 📊 Pipeline & System Topology
## 📊 Pipeline & System Topology

```text
                       ┌───────────────────────────────────────────────┐
                       │       Stateful Multi-Page Streamlit UI        │
                       └───────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
          [ Engine 1: Real-World Assessor ]              [ Engine 2: Live Market Assessor ]
                       │                                               │
                       ▼                                               ▼
           Vectorized User Inputs (10 Ratios)             Dynamic Ticker Query Execution (e.g., AAPL)
                       │                                               │
                       ▼                                               ▼
          Pre-trained XGBoost Model Execution             Upstream REST Call to FMP Cloud API Gateway
                       │                                               │
                       ▼                                               ▼
        SHAP TreeExplainer Local Attribution              JSON Payload Deserialization & Structural Parsing
                       │                                               │
                       ▼                                               ▼
         Matplotlib Rendered Force Plot UI                Z-Score & Liquidity Financial Evaluation Matrix
```

## 🔬 Deep Technical Breakdown (For Engineering Recruiters)

### 1. Mathematical Handling of Severe Class Imbalance
In corporate default datasets, bankruptcy events represent a minor fraction of historical data ($< 5\%$ positive target instances vs. $> 95\%$ healthy enterprise instances). Optimizing a standard binary cross-entropy loss function on this data creates an unviable model that aggressively biases toward majority-class predictions:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

To force the optimization algorithm to penalize missed bankruptcies (False Negatives), a cost-sensitive loss wrapper was implemented using the `scale_pos_weight` hyperparameter inside the `XGBClassifier`, calculated as:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y\_i \log(\hat{y}\_i) + (1 - y\_i) \log(1 - \hat{y}\_i) \right]$$

To force the optimization algorithm to penalize missed bankruptcies (False Negatives), a cost-sensitive loss wrapper was implemented using the `scale_pos_weight` hyperparameter inside the `XGBClassifier`, calculated as:

$$\text{scale\_pos\_weight} = \frac{\text{Total Negative Instances}}{\text{Total Positive Instances}}$$

This structural adjustment penalizes errors on minority instances, raising model sensitivity during training.

### 2. Algorithmic Feature Selection & Dimensionality Reduction
The baseline dataset contained high-dimensional noise (95 distinct financial metrics), introducing collinearity and risk of overfitting. An empirical Feature Importance evaluation pipeline was executed. The 95 baseline dimensions were successfully pruned down to the **Top 10 High-Impact Predictive Indicators** (including *Total Debt/Total Net Worth, Operating Profit Rate, Retained Earnings/Total Assets, and Net Value Per Share*) without degrading receiver operating characteristics (ROC).

### 3. Real-Time SHAP TreeExplainer Optimizations
To ensure real-time latency on web requests, the application incorporates a specialized `shap.TreeExplainer` optimization designed for tree ensemble structural layers. This maps individual user inputs into a localized force plot on demand:

$$g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i$$

Where $\phi_i$ represents the isolated contribution weight (Shapley value) assigned to an individual financial ratio relative to the base reference model expectation $\phi_0$.

---

## 💼 Business & Product Architecture (For Management Recruiters)

### 1. Mitigating the Core Risk of "Black Box" AI in Finance
Institutional risk management, credit underwriting, and auditing teams cannot deploy uninterpretable models due to legal compliance mandates (such as Fair Lending regulations). Kubera solves this directly through **Explainable AI (XAI)**. 
* Whenever a business user tests a company's ratios, the interface generates clear visual attributions (**Red** items represent ratios driving the company toward bankruptcy; **Blue** items represent features pushing them toward stability).
* This provides credit analysts with an immediate, audit-ready justification for a loan denial or high risk score.

### 2. Cost-Benefit Tradeoffs: Prioritizing Recall over Accuracy
While the system boasts a global classification accuracy of **95.97%**, the primary performance target was **Recall Optimization (40.91%)**. In commercial banking, a **False Negative** (failing to flag a defaulting firm) costs millions in bad debt. A **False Positive** (unnecessarily auditing a healthy firm) costs minor operational overhead. The model threshold optimization deliberately prioritizes raw sensitivity (Recall) to safely isolate toxic assets.

### 3. Frictionless API Automation for Enterprise Workflows
Manual corporate profiling is time-consuming. Engine 2 automates this by executing a programmatic lookup vector directly to the **Financial Modeling Prep (FMP) API Gateway**. 
* Submitting a standard stock ticker (e.g., `AAPL`, `TSLA`) triggers an instantaneous backend fetch of real-time balance sheets and income statements.
* The parsed data calculations generate comprehensive liquidity index views and solvency summaries in milliseconds, boosting data analyst efficiency.

---

## 🚀 Installation, Security, & Verification

### Prerequisites
* Verified Python 3.11+ environment initialized.
* Active Financial Modeling Prep API reference key.

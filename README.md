# 💳 Credit Card Fraud Detection

A complete end-to-end machine learning project that detects fraudulent credit card transactions using various models and a cost-sensitive approach. The project includes feature engineering, time-based evaluation, model optimization, and an interactive Streamlit dashboard for seamless exploration and predictions.

---

## 📁 Project Structure
├── creditcard.csv # Input dataset ├── models/ # Saved model files and metadata ├── preprocessing/ # Scaling and feature selection objects ├── notebooks/ # EDA and modeling notebooks ├── streamlit_app.py # Main Streamlit dashboard ├── utils/ # Helper functions for training & evaluation ├── requirements.txt # Project dependencies └── README.md # Project documentation


---

## 📌 Methodology Overview

### 1. Data Preparation
- We loaded and explored the `creditcard.csv` dataset.
- Created new features: `Hour` (from Time), `LogAmount` (log-transformed Amount).
- Visualized class imbalance and transaction statistics.

### 2. Feature Engineering
- Dropped `Time` and `Amount`.
- Selected top 20 features using `SelectKBest` with `f_classif`.
- Scaled features using `StandardScaler`.

### 3. Train-Test Split
- Used a **70/30 time-based split** to simulate real-time fraud prediction.

### 4. Modeling
- Trained the following models:
  - Logistic Regression (with class balancing)
  - Random Forest (tuned)
  - K-Nearest Neighbors
- Compared all models using consistent metrics.

### 5. Cost-Sensitive Optimization
- Tuned thresholds to **minimize false negatives**, acknowledging the higher cost of missing fraud.

### 6. Model Evaluation
- Used **5-fold cross-validation**.
- Metrics:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - PR-AUC
- Visualized confusion matrices and precision-recall curves.

### 7. Feature Importance
- Identified key predictors using Random Forest feature importance.

### 8. False Negative Analysis
- Reviewed missed frauds to uncover model blind spots.

---

## 🖥️ Streamlit Dashboard

Our interactive dashboard allows:
- ✅ Model selection and instant result switching
- 📊 Visual performance comparison
- ⚙️ Real-time threshold tuning for cost control
- 📂 CSV file upload for prediction
- 💾 Model saving/loading with metadata

To run the app:

```bash
streamlit run streamlit_app.py
```

##🧠 Models & Persistence
All trained models are saved with preprocessing steps.
-Metadata includes:
-Model parameters
-Performance scores
-Timestamped versioning

##🔍 Prediction on New Data
Upload transaction data via dashboard.
-Pipeline applies the same preprocessing.
-Get fraud probability scores + binary predictions.
-Option to download results as CSV.

##🛠️ Tech Stack
--Languages & Libraries: Python, Pandas, NumPy, Scikit-learn
--Dashboard: Streamlit
--Visualization: Matplotlib, Seaborn, Plotly
-=Model Persistence: Joblib, Pickle




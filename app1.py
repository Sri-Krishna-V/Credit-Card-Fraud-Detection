import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve, auc,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

# Streamlit Page Config
st.set_page_config(layout="wide")
sns.set_palette("viridis")
plt.style.use('ggplot')

# Load Data


@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    data['Hour'] = (data['Time'] // 3600) % 24
    data['LogAmount'] = np.log1p(data['Amount'])
    return data


data = load_data()
st.title("\U0001F4B8 Credit Card Fraud Detection Dashboard")
st.markdown(
    "Analyze and classify fraudulent transactions using various ML models.")

# Sidebar
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Select Model", ["Logistic Regression", "Random Forest", "K-Nearest Neighbors"])
cost_fn = st.sidebar.slider("Cost of False Negative", 1, 50, 10)
cost_fp = st.sidebar.slider("Cost of False Positive", 1, 10, 1)

# Model operations mode
training_mode = st.sidebar.radio(
    "Model Operation Mode",
    ["Load Existing Model (if available)", "Train New Model"]
)

# Model versioning
model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Function to save model and related components


def save_model_components(model, scaler, selected_features, threshold, model_name, version):
    base_path = f"{model_dir}/{model_name.replace(' ', '').lower()}{version}"
    model_path = f"{base_path}_model.pkl"
    scaler_path = f"{base_path}_scaler.pkl"
    features_path = f"{base_path}_features.pkl"
    threshold_path = f"{base_path}_threshold.pkl"

    # Save all components
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(selected_features.tolist(), features_path)
    joblib.dump(threshold, threshold_path)

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "date_created": datetime.datetime.now().isoformat(),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "threshold": threshold,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_path": features_path,
        "threshold_path": threshold_path
    }

    joblib.dump(metadata, f"{base_path}_metadata.pkl")
    return metadata

# Function to find available model versions


def get_available_models():
    available_models = {}
    if not os.path.exists(model_dir):
        return available_models

    for filename in os.listdir(model_dir):
        if filename.endswith("_metadata.pkl"):
            metadata = joblib.load(f"{model_dir}/{filename}")
            model_key = metadata["model_name"]
            if model_key not in available_models:
                available_models[model_key] = []
            available_models[model_key].append(metadata)

    # Sort versions by date
    for model_key in available_models:
        available_models[model_key].sort(
            key=lambda x: x["date_created"], reverse=True)

    return available_models


# Check for available models
available_models = get_available_models()

# Allow selecting a saved model version if available and in loading mode
selected_metadata = None
if training_mode == "Load Existing Model (if available)" and model_name in available_models and available_models[model_name]:
    versions = [f"{m['version']} (Created: {m['date_created'].split('T')[0]}, FN Cost: {m['cost_fn']}, FP Cost: {m['cost_fp']})"
                for m in available_models[model_name]]

    selected_version = st.sidebar.selectbox("Select Model Version", versions)
    selected_idx = versions.index(selected_version)
    selected_metadata = available_models[model_name][selected_idx]

    st.sidebar.info(
        f"Loading model version {selected_metadata['version']} created on {selected_metadata['date_created'].split('T')[0]}")
elif training_mode == "Load Existing Model (if available)" and (model_name not in available_models or not available_models[model_name]):
    st.sidebar.warning(
        f"No saved versions found for {model_name}. Will train a new model.")
    training_mode = "Train New Model"

# Train-Test Split by Time
data_sorted = data.sort_values("Time").reset_index(drop=True)
split_idx = int(0.7 * len(data_sorted))
cutoff_time = data_sorted.iloc[split_idx]["Time"]
train_data = data_sorted[data_sorted["Time"] <= cutoff_time]
test_data = data_sorted[data_sorted["Time"] > cutoff_time]

# Feature Selection
X = data.drop(['Class', 'Time', 'Amount'], axis=1)
y = data['Class']

# Cost-sensitive Threshold


def cost_sensitive_threshold_search(model, X, y, cost_fn=10, cost_fp=1):
    y_scores = model.predict_proba(X)[:, 1]
    best_thresh, min_cost = 0.5, float('inf')
    for thresh in np.linspace(0, 1, 100):
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = cost_fn * fn + cost_fp * fp
        if cost < min_cost:
            best_thresh = thresh
            min_cost = cost
    return best_thresh, min_cost


# Model Setup
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3)
}

# Load or train model
if training_mode == "Load Existing Model (if available)" and selected_metadata:
    # Load existing model and components
    model = joblib.load(selected_metadata["model_path"])
    scaler = joblib.load(selected_metadata["scaler_path"])
    selected_features = joblib.load(selected_metadata["features_path"])
    best_thresh = joblib.load(selected_metadata["threshold_path"])

    # Prepare test data with saved features
    X_test = test_data[selected_features]
    y_test = test_data['Class']
    X_test_scaled = scaler.transform(X_test)

    # Generate predictions with loaded model and threshold
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_scores >= best_thresh).astype(int)

    st.success(
        f"Successfully loaded {model_name} (version: {selected_metadata['version']})")

else:
    # Train new model
    selector = SelectKBest(f_classif, k=20)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    X_train = train_data[selected_features]
    y_train = train_data['Class']
    X_test = test_data[selected_features]
    y_test = test_data['Class']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = models[model_name]
    model.fit(X_train_scaled, y_train)

    # Find optimal threshold
    y_scores = model.predict_proba(X_test_scaled)[:, 1]
    best_thresh, min_cost = cost_sensitive_threshold_search(
        model, X_test_scaled, y_test, cost_fn, cost_fp)
    y_pred = (y_scores >= best_thresh).astype(int)

    # Save model and components
    metadata = save_model_components(
        model, scaler, selected_features, best_thresh, model_name, model_version)
    st.sidebar.success(f"Model saved as version: {model_version}")

# Display Metrics
st.subheader(f"{model_name} Evaluation")

if training_mode == "Train New Model":
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=3, scoring='f1')
    st.markdown(
        f"*Cross-Validation F1 Score (mean)*: {np.mean(cv_scores):.3f}")
    st.markdown(
        f"*Best Threshold: {best_thresh:.2f}, **Minimum Cost*: {min_cost}")
else:
    st.markdown(f"*Loaded Threshold*: {best_thresh:.2f}")

metrics = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1-Score", "ROC-AUC"],
    "Value": [precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_scores)]
})
st.dataframe(metrics.set_index("Metric"))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'], ax=ax_cm)
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_scores)
fig_pr, ax_pr = plt.subplots()
ax_pr.plot(recall_vals, precision_vals,
           label=f'PR AUC = {auc(recall_vals, precision_vals):.3f}', color='purple')
ax_pr.set_xlabel('Recall')
ax_pr.set_ylabel('Precision')
ax_pr.set_title('Precision-Recall Curve')
ax_pr.legend()
st.pyplot(fig_pr)

# Feature Importances
if model_name == "Random Forest":
    feat_imp = pd.Series(model.feature_importances_,
                         index=selected_features).sort_values(ascending=False)
    st.subheader("Feature Importances")
    fig_imp, ax_imp = plt.subplots()
    feat_imp.plot(kind='bar', color='tomato', ax=ax_imp)
    st.pyplot(fig_imp)

# False Negatives
false_negatives = test_data[(y_test.values == 1) & (y_pred == 0)]
st.subheader("False Negatives (Missed Frauds)")
st.dataframe(false_negatives.reset_index(drop=True))

# Upload for Prediction on Unseen Data
st.sidebar.header("Unseen Data Prediction")
uploaded_file = st.sidebar.file_uploader("Upload Unseen CSV", type=["csv"])

if uploaded_file is not None:
    unseen = pd.read_csv(uploaded_file)
    try:
        # Use current loaded/trained model components
        unseen_scaled = scaler.transform(unseen[selected_features])
        scores = model.predict_proba(unseen_scaled)[:, 1]
        preds = (scores >= best_thresh).astype(int)

        # Add predictions and confidence scores
        unseen["Fraud_Probability"] = scores
        unseen["Prediction"] = preds

        st.subheader("Unseen Data Predictions")
        st.dataframe(unseen)

        # Allow downloading predictions
        csv = unseen.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"predictions_{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

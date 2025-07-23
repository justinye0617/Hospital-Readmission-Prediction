# Hospital Readmission Prediction Capstone

## Executive Summary

This project focuses on predicting hospital readmissions using patient data. The goal is to help hospitals identify which patients are at higher risk of being readmitted shortly after discharge. This can lead to better resource planning, improved care, and reduced costs.

A Random Forest Classifier was trained on patient demographics and medical features using a preprocessing pipeline. The model achieved a validation accuracy of approximately **85%** and a **ROC AUC score of 0.91**. The most important features contributing to readmission included comorbidity score, discharge destination, and primary diagnosis.

This project includes exploratory data analysis, model training, evaluation, feature importance visualization, and an interactive terminal-based prediction system.

## Project Description

This is a binary classification problem using structured data. The target variable is `readmitted` (1 for yes, 0 for no). The data includes both numeric and categorical features. A machine learning pipeline was built using **scikit-learn** to streamline preprocessing and modeling. Feature scaling and one-hot encoding were handled using a `ColumnTransformer`, and the entire workflow was wrapped in a `Pipeline` for clean integration and deployment.

## Dataset

- **File:** `train_df.csv`
- **Target:** `readmitted`
- **Features:**
  - `age`
  - `days_in_hospital`
  - `num_procedures`
  - `comorbidity_score`
  - `gender`
  - `primary_diagnosis`
  - `discharge_to`

## Main Steps

1. Load and clean the data
2. Handle missing values using mode imputation
3. Preprocess using scaling and one-hot encoding
4. Train Random Forest model using a pipeline
5. Evaluate using accuracy, ROC AUC, and classification metrics
6. Visualize feature importance using Matplotlib
7. Save model pipeline with a timestamped filename using `joblib`
8. Create interactive terminal-based prediction function with type matching

## Model Performance

- **Validation Accuracy:** 85%
- **ROC AUC Score:** 0.91
- **Classification Report:** Includes precision, recall, and F1-score for both classes

## Feature Importance

Feature importance was extracted directly from the trained Random Forest model and visualized using a horizontal bar chart. This helps communicate which factors most strongly influence readmission risk.

**Top contributing features:**
- `comorbidity_score`
- `discharge_to`
- `primary_diagnosis`

## Project Files

- `train_df.csv` – Dataset used for training
- `train_mode.py` – Main script for model training, evaluation, and prediction
- `eda_analysis.ipynb` – Exploratory data analysis notebook (optional extension)
- `README.md` – Project overview and instructions
- `requirements.txt` – Required Python packages
- `saved_models/` – Folder for timestamped saved model pipelines

## Future Work

- Add grid search or randomized search for hyperparameter tuning
- Explore additional ensemble models (e.g., Gradient Boosting, XGBoost)
- Use SHAP or LIME to improve interpretability of model decisions
- Build a web app interface using Streamlit or Flask for non-technical users

## Author

**Justin Ye**  
Capstone Project – Hospital Readmission Prediction

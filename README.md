# Hospital Readmission Prediction Capstone

## Executive Summary

This project focuses on predicting hospital readmissions using patient data. The goal is to help hospitals identify which patients are at higher risk of being readmitted shortly after discharge. This can lead to better resource planning, improved care, and reduced costs.

A Random Forest Classifier was trained on patient demographics and medical features using a preprocessing pipeline. The model achieved a validation accuracy of approximately 85% and a ROC AUC score of 0.91. The most important features contributing to readmission included comorbidity score, discharge destination, and primary diagnosis.

This project includes exploratory data analysis, model training, evaluation, and an interactive terminal-based prediction system.

## Project Description

This is a binary classification problem using structured data. The target variable is `readmitted` (1 for yes, 0 for no). The data includes both numeric and categorical features. A machine learning pipeline was built using scikit-learn to streamline preprocessing and modeling.

## Dataset

- File: `train_df.csv`
- Target: `readmitted`
- Features:
  - age
  - days_in_hospital
  - num_procedures
  - comorbidity_score
  - gender
  - primary_diagnosis
  - discharge_to

## Main Steps

1. Load and clean the data
2. Handle missing values
3. Preprocess using scaling and encoding
4. Train Random Forest model in a pipeline
5. Evaluate using accuracy and ROC AUC
6. Save model with joblib
7. Create interactive terminal prediction function

## Model Performance

- Validation Accuracy: 85%
- ROC AUC Score: 0.91
- Classification report includes precision, recall, and F1-score

## Feature Importance

Feature importance was extracted from the trained Random Forest model. Top features included:

- comorbidity_score
- discharge_to
- primary_diagnosis

## Project Files

- `train_df.csv` – Dataset used for training
- `predict.py` – Main script that trains and predicts
- `eda_analysis.ipynb` – Exploratory data analysis notebook
- `README.md` – Project overview
- `requirements.txt` – Required Python packages
- `saved_models/` – Folder for saved model pipelines

## Future Work

- Add hyperparameter tuning
- Try other ensemble models
- Improve interpretability with SHAP or LIME
- Deploy with Streamlit or Flask

## Author

Justin Ye  
Capstone Project – Hospital Readmission Prediction

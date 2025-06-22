import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

#Load and Prepare the Dataset
train_df = pd.read_csv("/Users/justinye/Desktop/Python/Hospital Readmission/data/train_df.csv")
train_df.columns = train_df.columns.str.lower().str.strip()
train_df.fillna(train_df.mode().iloc[0], inplace=True)

#Define Features and Target
target_column = "readmitted"
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

#Identify Numeric and Categorical Features
numeric_features = ["age", "days_in_hospital", "num_procedures", "comorbidity_score"]
categorical_features = ["gender", "primary_diagnosis", "discharge_to"]

#Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

#Full Pipeline with Random Forest
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42))
])

#Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

#Evaluate Performance
y_val_pred = pipeline.predict(X_val)
y_val_proba = pipeline.predict_proba(X_val)[:, 1]
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation ROC AUC Score:", roc_auc_score(y_val, y_val_proba))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

#Feature Importance Visualization
importances = pipeline.named_steps["classifier"].feature_importances_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
importance_df = importance_df.sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

#Save the Pipeline with Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"/Users/justinye/Desktop/Python/Hospital Readmission/data/hospital_pipeline_{timestamp}.pkl"
joblib.dump(pipeline, model_path)
print(f"Pipeline saved as: {model_path}")

#Interactive Prediction Function
def predict_patient_readmission():
    print("\nEnter Patient Details to Predict Readmission:")

    age = float(input("Enter patient's age: "))
    days_in_hospital = float(input("Enter number of days in hospital: "))
    num_procedures = int(input("Enter number of procedures performed: "))
    comorbidity_score = int(input("Enter comorbidity score (e.g., 0-10): "))
    gender = input("Enter gender (male/female): ").strip().lower()
    primary_diagnosis = input("Enter primary diagnosis (e.g., diabetes, hypertension): ").strip().lower()
    discharge_to = input("Enter discharge type (home health care, rehab facility, skilled nursing facility): ").strip().lower()

    new_data = pd.DataFrame([{
        "age": age,
        "days_in_hospital": days_in_hospital,
        "num_procedures": num_procedures,
        "comorbidity_score": comorbidity_score,
        "gender": gender,
        "primary_diagnosis": primary_diagnosis,
        "discharge_to": discharge_to
    }])

    #Match data types
    new_data = new_data.astype(X_train.dtypes.to_dict())

    prediction = pipeline.predict(new_data)[0]
    probability = pipeline.predict_proba(new_data)[0][1]

    print("\nPrediction Result:")
    print(f"➡ Readmission probability: {probability * 100:.2f}%")

#Run the prediction interface
predict_patient_readmission()

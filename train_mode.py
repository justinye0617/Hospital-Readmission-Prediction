import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime
import joblib

# 1. Load and Prepare Dataset
df = pd.read_csv("/Users/justinye/Desktop/Python/Hospital Readmission/data/train_df.csv")
df.columns = df.columns.str.lower().str.strip()
df.fillna(df.mode().iloc[0], inplace = True)

# 2. Define Features and Target
target_col = "readmitted"
X = df.drop(columns = [target_col])
y = df[target_col]

numeric_features = ["age", "days_in_hospital", "num_procedures", "comorbidity_score"]
categorical_features = ["gender", "primary_diagnosis", "discharge_to"]

# 3. Preprocessing and Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators = 300, max_depth = 15, random_state = 42))
])

# 4. Train/Test Split and Model Training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
pipeline.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = pipeline.predict(X_val)
y_proba = pipeline.predict_proba(X_val)[:, 1]

print("\n--- Model Evaluation ---")
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.2f}")
print(f"ROC AUC Score: {roc_auc_score(y_val, y_proba):.2f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# 6. Feature Importance Plot
importances = pipeline.named_steps["classifier"].feature_importances_
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by = "importance", ascending = False)

plt.figure(figsize = (10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.xlabel("Importance Score")
plt.title("Feature Importances (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 7. Save Model with Timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"/Users/justinye/Desktop/Python/Hospital Readmission/data/hospital_pipeline_{timestamp}.pkl"
joblib.dump(pipeline, model_path)
print(f"\nModel saved to: {model_path}")

# 8. Interactive Prediction Function
def predict_patient_readmission():
    print("\n--- Predict Readmission Risk ---")
    try:
        age = float(input("Age: "))
        days_in_hospital = float(input("Days in hospital: "))
        num_procedures = int(input("Number of procedures: "))
        comorbidity_score = int(input("Comorbidity score (0–10): "))
        gender = input("Gender (male/female): ").strip().lower()
        primary_diagnosis = input("Primary diagnosis (e.g., diabetes): ").strip().lower()
        discharge_to = input("Discharge type (home health care, rehab facility, etc.): ").strip().lower()

        input_data = pd.DataFrame([{
            "age": age,
            "days_in_hospital": days_in_hospital,
            "num_procedures": num_procedures,
            "comorbidity_score": comorbidity_score,
            "gender": gender,
            "primary_diagnosis": primary_diagnosis,
            "discharge_to": discharge_to
        }])

        input_data = input_data.astype(X_train.dtypes.to_dict())

        prediction = pipeline.predict(input_data)[0]
        prob = pipeline.predict_proba(input_data)[0][1]

        print("\nPrediction:")
        print(f"Readmission probability: {prob * 100:.2f}%")
        print("Likely to be readmitted." if prediction == 1 else "Unlikely to be readmitted.")

    except Exception as e:
        print(f"Error during prediction: {e}")

# 9. Run Prediction Interface
predict_patient_readmission()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Load and Prepare the Dataset
train_df = pd.read_csv("/Users/justinye/Desktop/Python/Hospital Readmission/data/train_df.csv")

#Standardize column names (all lowercase, stripped)
train_df.columns = train_df.columns.str.lower().str.strip()

#Handle missing values (using mode imputation for simplicity)
train_df.fillna(train_df.mode().iloc[0], inplace = True)

#The target column
target_column = "readmitted"

#Define features and target variable
X = train_df.drop(columns=[target_column])
y = train_df[target_column]

#Define the Preprocessing Pipeline
#List numeric and categorical feature names exactly as in the dataset.
numeric_features = ["age", "days_in_hospital", "num_procedures", "comorbidity_score"]
categorical_features = ["gender", "primary_diagnosis", "discharge_to"]

#Define ColumnTransformer that scales numeric features and one-hot encodes categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown = "ignore"), categorical_features)
    ]
)

#Build the pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators = 300, max_depth = 15, random_state = 42))
])

#Train/Test Split and Model Training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)
pipeline.fit(X_train, y_train)

#Evaluate on the validation set
y_val_pred = pipeline.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:\n", classification_report(y_val, y_val_pred))

#Save the pipeline for later use
joblib.dump(pipeline, "/Users/justinye/Desktop/Python/Hospital Readmission/data/hospital_readmission_pipeline.pkl")
print("Pipeline saved as 'hospital_readmission_pipeline.pkl'")

#Interactive Prediction Function
def predict_patient_readmission():
    print("\nEnter Patient Details to Predict Readmission:")

    #Collect numerical inputs
    age = float(input("Enter patient's age: "))
    days_in_hospital = float(input("Enter number of days in hospital: "))
    num_procedures = int(input("Enter number of procedures performed: "))
    comorbidity_score = int(input("Enter comorbidity score (e.g., 0-10): "))

    #Collect categorical inputs (in lowercase to match training data)
    gender = input("Enter gender (male/female): ").strip().lower()
    primary_diagnosis = input("Enter primary diagnosis (diabetes, hypertension, heart disease, etc.): ").strip().lower()
    discharge_to = input("Enter discharge type (home health care, rehabilitation facility, skilled nursing facility): ").strip().lower()

    #Build a new DataFrame for the input
    new_data = pd.DataFrame([{
        "age": age,
        "days_in_hospital": days_in_hospital,
        "num_procedures": num_procedures,
        "comorbidity_score": comorbidity_score,
        "gender": gender,
        "primary_diagnosis": primary_diagnosis,
        "discharge_to": discharge_to
    }])
    
    #Use the pipeline to process the input and predict
    prediction = pipeline.predict(new_data)[0]
    probability = pipeline.predict_proba(new_data)[0][1]  #Probability for class '1'
    
    print("\nPrediction Result:")
    print(f"➡ Readmission probability: {probability * 100:.2f}%\n")

predict_patient_readmission()

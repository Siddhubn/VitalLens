import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_FILE = os.path.join(SCRIPT_DIR, 'processed_dataset.csv')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# --- Load Data ---
df = pd.read_csv(PROCESSED_DATA_FILE)

# Features
features = ['calculated_hr']
X = df[features]

# Targets
y_bp = df[['blood_pressure_systolic', 'blood_pressure_diastolic']]
y_hr = df['heart_rate']
y_stress = df['Stress level']

# --- Load Models ---
bp_model = joblib.load(os.path.join(MODELS_DIR, 'bp_model.pkl'))
hr_model = joblib.load(os.path.join(MODELS_DIR, 'hr_model.pkl'))
stress_model = joblib.load(os.path.join(MODELS_DIR, 'stress_model.pkl'))

# --- Evaluation Function ---
def evaluate_models(X, y_bp, y_hr, y_stress):
    print("\n--- Blood Pressure Evaluation ---")
    bp_preds = bp_model.predict(X)
    print("MAE:", mean_absolute_error(y_bp, bp_preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_bp, bp_preds)))
    print("R² Score:", r2_score(y_bp, bp_preds))

    print("\n--- Heart Rate Evaluation ---")
    hr_preds = hr_model.predict(X)
    print("MAE:", mean_absolute_error(y_hr, hr_preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_hr, hr_preds)))
    print("R² Score:", r2_score(y_hr, hr_preds))

    print("\n--- Stress Level Evaluation ---")
    stress_preds = stress_model.predict(X)
    print("Accuracy:", accuracy_score(y_stress, stress_preds))
    print("\nClassification Report:\n", classification_report(y_stress, stress_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_stress, stress_preds))


# --- Execute Evaluation ---
if __name__ == "__main__":
    evaluate_models(X, y_bp, y_hr, y_stress)

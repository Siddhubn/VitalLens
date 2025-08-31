import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_FILE = os.path.join(SCRIPT_DIR, 'processed_dataset.csv')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# --- Main Training Function ---
def train_models():
    """
    Loads the processed data, trains models for BP, HR, and Stress,
    and saves them to disk.
    """
    # 1. Load the processed dataset
    try:
        df = pd.read_csv(PROCESSED_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: '{PROCESSED_DATA_FILE}' not found.")
        print("Please run ml_processor.py first to generate the processed dataset.")
        return

    print("--- Dataset Loaded ---")
    print(df.head())

    # 2. Define Features (X) and Targets (y)
    # For this initial model, we will use the heart rate we calculated from the signal
    # as the primary feature. In a real project, you would add many more features.
    features = ['calculated_hr']
    X = df[features]

    # Define the different targets we want to predict
    y_bp = df[['blood_pressure_systolic', 'blood_pressure_diastolic']]
    y_hr = df['heart_rate']
    y_stress = df['Stress level'] # The column name from your CSV

    # 3. Split the data into training and testing sets
    X_train, X_test, y_bp_train, y_bp_test, y_hr_train, y_hr_test, y_stress_train, y_stress_test = train_test_split(
        X, y_bp, y_hr, y_stress, test_size=0.2, random_state=42
    )

    print(f"\nTraining with {len(X_train)} samples, testing with {len(X_test)} samples.")

    # --- Train Blood Pressure Model ---
    print("\n--- Training Blood Pressure Model ---")
    # A RandomForestRegressor can predict multiple targets at once
    bp_model = RandomForestRegressor(n_estimators=100, random_state=42)
    bp_model.fit(X_train, y_bp_train)
    
    # Evaluate the model
    bp_preds = bp_model.predict(X_test)
    bp_mae = mean_absolute_error(y_bp_test, bp_preds)
    print(f"Blood Pressure Model MAE: {bp_mae:.2f}")

    # --- Train Heart Rate Model ---
    print("\n--- Training Heart Rate Model ---")
    hr_model = RandomForestRegressor(n_estimators=100, random_state=42)
    hr_model.fit(X_train, y_hr_train)
    
    # Evaluate the model
    hr_preds = hr_model.predict(X_test)
    hr_mae = mean_absolute_error(y_hr_test, hr_preds)
    print(f"Heart Rate Model MAE: {hr_mae:.2f} bpm")

    # --- Train Stress Level Model ---
    print("\n--- Training Stress Level Model ---")
    # We use a Classifier for stress because it's a category (Low, Moderate, High)
    stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
    stress_model.fit(X_train, y_stress_train)

    # Evaluate the model
    stress_preds = stress_model.predict(X_test)
    stress_accuracy = accuracy_score(y_stress_test, stress_preds)
    print(f"Stress Model Accuracy: {stress_accuracy * 100:.2f}%")

    # 4. Save the trained models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    joblib.dump(bp_model, os.path.join(MODELS_DIR, 'bp_model.pkl'))
    joblib.dump(hr_model, os.path.join(MODELS_DIR, 'hr_model.pkl'))
    joblib.dump(stress_model, os.path.join(MODELS_DIR, 'stress_model.pkl'))
    
    print(f"\nModels saved successfully to the '{MODELS_DIR}' directory.")


# --- Main Execution ---
if __name__ == '__main__':
    train_models()
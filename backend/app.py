from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import csv
import os
import joblib
import numpy as np
import pandas as pd

# --- App & DB Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(SCRIPT_DIR, '..', 'frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

CSV_FILE = os.path.join(SCRIPT_DIR, 'users.csv')
CSV_HEADERS = ['name', 'phone', 'email', 'gender', 'password']

def initialize_database():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Database file '{CSV_FILE}' created.")

initialize_database()

# --- Load ML Models ---
bp_model = None
hr_model = None
stress_model = None
try:
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    bp_model_path = os.path.join(MODELS_DIR, 'bp_model.pkl')
    hr_model_path = os.path.join(MODELS_DIR, 'hr_model.pkl')
    stress_model_path = os.path.join(MODELS_DIR, 'stress_model.pkl')

    bp_model = joblib.load(bp_model_path)
    hr_model = joblib.load(hr_model_path)
    stress_model = joblib.load(stress_model_path)
    print("--- All ML models loaded successfully ---")
except FileNotFoundError as e:
    print(f"Error loading models: {e}.")
    print("Please ensure your .pkl files are inside a 'models' folder in your GitHub repository.")
    print("Files found in root directory:", os.listdir(SCRIPT_DIR))
    if os.path.exists(MODELS_DIR):
        print("Files found in 'models' directory:", os.listdir(MODELS_DIR))
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")


# --- Static File Serving ---
@app.route('/')
def serve_home():
    return send_from_directory(FRONTEND_DIR, 'home.html')

@app.route('/login')
def serve_login():
    return send_from_directory(FRONTEND_DIR, 'login.html')

@app.route('/signup')
def serve_signup():
    return send_from_directory(FRONTEND_DIR, 'signup.html')

@app.route('/index')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


# --- User Authentication Endpoints ---
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not all(key in data for key in CSV_HEADERS):
        return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(data)

    return jsonify({'status': 'success', 'message': 'User registered successfully'})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    with open(CSV_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for user in reader:
            if user['email'] == email and user['password'] == password:
                return jsonify({'status': 'success', 'message': 'Login successful'})

    return jsonify({'status': 'error', 'message': 'Invalid email or password'}), 401


# --- ML Prediction Endpoint ---
@app.route('/process', methods=['POST'])
def process_data():
    if not all([bp_model, hr_model, stress_model]):
        return jsonify({'status': 'error', 'message': 'Models are not loaded on the server.'}), 500

    simulated_calculated_hr = random.uniform(60.0, 100.0)
    input_df = pd.DataFrame([[simulated_calculated_hr]], columns=['calculated_hr'])

    bp_pred = bp_model.predict(input_df)
    if bp_pred.ndim == 1:
        bp_pred = bp_pred.reshape(1, -1)

    # hr_pred is a NumPy array like [85.3]
    hr_pred = hr_model.predict(input_df)

    # stress_pred is a NumPy array like ['Moderate']
    stress_pred = stress_model.predict(input_df)

    # UPDATED: Convert the NumPy number to a standard Python float before rounding.
    results = {
        'status': 'success',
        'systolic': round(bp_pred[0, 0]),
        'diastolic': round(bp_pred[0, 1]),
        'heartRate': round(float(hr_pred[0])),
        'stress': stress_pred[0]
    }

    return jsonify(results)

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
import os
import subprocess
import uuid
import sqlite3
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import cv2
from ml_processor import process_video_for_ippg, extract_features
from report_generator import create_report

# --- Database Configuration ---
DATABASE = 'users.db'

def init_db():
    """Initializes the database and creates the users table with a fullname column."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            fullname TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# --- Model Training (No changes here) ---
MODEL_PATH = 'trained_model/vital_signs_model.pkl'

def train_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Starting the training process...")
        try:
            subprocess.run(['python', 'train_model.py'], check=True, capture_output=True, text=True)
            print("✅ Training complete. Model has been created.")
        except Exception as e:
            print(f"❌ ERROR: Model training failed.\n{e}")
            exit()
    else:
        print("✅ Found existing trained model.")

# --- App Configuration & Initialization ---
app = Flask(__name__,
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
app.secret_key = 'your_super_secret_key'

init_db()
train_model_if_needed()
model = joblib.load(MODEL_PATH)
print("Flask app is ready to accept requests.")

# --- Main Routes ---
@app.route('/')
def index():
    """Renders the main landing page."""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data['username']
        password = data['password']
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT password, fullname FROM users WHERE username = ?", (username,))
        user_record = cursor.fetchone()
        conn.close()

        if user_record and check_password_hash(user_record[0], password):
            session['user'] = username
            session['fullname'] = user_record[1]
            return jsonify({'success': True})
        
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.json
        username = data['username']
        password = data['password']
        fullname = data['fullname']
        
        hashed_password = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, fullname) VALUES (?, ?, ?)",
                (username, hashed_password, fullname)
            )
            conn.commit()
            conn.close()
            return jsonify({'success': True})
        except sqlite3.IntegrityError:
            return jsonify({'success': False, 'error': 'Username already exists'}), 409
            
    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    display_name = session.get('fullname', 'User')
    return render_template('home.html', username=display_name)

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('fullname', None)
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    if not request.files.get('video_blob'): return jsonify({'error': 'No video blob received.'}), 400
    
    file = request.files.get('video_blob')
    unique_filename = f"{uuid.uuid4()}.webm"
    temp_video_path = unique_filename
    file.save(temp_video_path)

    try:
        # This function is now optimized in ml_processor.py to be fast
        filtered_signal = process_video_for_ippg(temp_video_path)
        if filtered_signal is None:
            return jsonify({'error': 'Could not detect a stable signal from the video.'}), 400

        features = extract_features(filtered_signal)
        if features is None:
            return jsonify({'error': 'Could not extract features from the signal.'}), 400

        prediction = model.predict([features])[0]
        result = {'systolic_bp': round(prediction[0]), 'diastolic_bp': round(prediction[1]), 'heart_rate': round(prediction[2])}
        session['last_measurement'] = result
        return jsonify(result)
    finally:
        # Ensure the temporary file is always deleted
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

@app.route('/download_report')
def download_report():
    if 'user' not in session: return redirect(url_for('login'))
    if 'last_measurement' not in session: return "No measurement found.", 404

    username = session.get('fullname', session['user'])
    vitals = session['last_measurement']
    report_path = create_report(username, vitals)
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


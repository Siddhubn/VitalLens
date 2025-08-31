import cv2
import numpy as np
import os
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CASCADE_PATH = os.path.join(SCRIPT_DIR, 'haarcascade_frontalface_default.xml')
VIDEO_FOLDER = os.path.join(SCRIPT_DIR, 'videos')
LABELS_FILE = os.path.join(SCRIPT_DIR, 'labels.csv')  # labels.csv in backend folder

# --- Signal Processing Functions ---

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    """
    Applies a bandpass filter to the signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        print(f"Warning: Signal too short for filtering (length={len(data)}, required>{padlen}). Skipping.")
        return None
    y = filtfilt(b, a, data)
    return y

def extract_raw_signal(video_path):
    """
    Extracts the raw iPPG signal and frame rate from a video file.
    """
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade classifier from path: {FACE_CASCADE_PATH}")
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # default if invalid FPS
        print(f"Warning: Invalid FPS, defaulting to {fps}")

    raw_signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            forehead_x = x + int(0.2 * w)
            forehead_y = y + int(0.1 * h)
            forehead_w = int(0.6 * w)
            forehead_h = int(0.2 * h)
            forehead_roi = frame[forehead_y:forehead_y + forehead_h,
                                 forehead_x:forehead_x + forehead_w]

            if forehead_roi.size > 0:
                avg_green = np.mean(forehead_roi[:, :, 1])
                raw_signal.append(avg_green)

    cap.release()
    return raw_signal, fps

def process_all_videos():
    """
    Processes all videos, extracts features, and combines them with labels.
    """
    try:
        labels_df = pd.read_csv(LABELS_FILE)
    except FileNotFoundError:
        print(f"Error: '{LABELS_FILE}' not found.")
        return None

    all_features = []

    for index, row in labels_df.iterrows():
        video_filename = row['filename'] + '.mp4'
        video_path = os.path.join(VIDEO_FOLDER, video_filename)

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found for {video_filename}. Skipping.")
            continue

        print(f"Processing: {video_filename}...")

        raw_signal, fps = extract_raw_signal(video_path)
        if not raw_signal or not fps:
            print(f"Could not extract signal from {video_filename}. Skipping.")
            continue

        filtered_signal = bandpass_filter(raw_signal, 0.75, 4.0, fps)
        if filtered_signal is None:
            continue

        # Optional: smooth the signal
        filtered_signal = np.convolve(filtered_signal, np.ones(3)/3, mode='same')

        # Find peaks with minimum distance ~0.5 sec to avoid false peaks
        min_distance = int(0.5 * fps)
        peaks, _ = find_peaks(filtered_signal, distance=min_distance)
        if len(peaks) < 2:
            print(f"Not enough peaks in {video_filename}. Skipping.")
            continue

        peak_intervals = np.diff(peaks) / fps
        avg_interval = np.mean(peak_intervals)
        heart_rate = 60.0 / avg_interval  # BPM

        features = {
            'filename': row['filename'],
            'calculated_hr': heart_rate,
        }

        all_features.append({**row.to_dict(), **features})

    if not all_features:
        print("No videos were processed successfully.")
        return None

    return pd.DataFrame(all_features)

# --- Main Execution ---
if __name__ == '__main__':
    processed_data = process_all_videos()
    if processed_data is not None:
        print("\n--- Processing Complete ---")
        print(processed_data.head())
        processed_data.to_csv('processed_dataset.csv', index=False)
        print("\nProcessed data saved to 'processed_dataset.csv'")
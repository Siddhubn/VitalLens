import cv2
import numpy as np
from scipy.signal import butter, filtfilt

# --- Configuration ---
# This line loads the pre-built face detector model from OpenCV's library.
# It should work automatically if opencv-python is installed correctly.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    print("ERROR: Could not load the Haar Cascade face detector.")
    print("Please ensure that 'opencv-python' is installed correctly.")
    exit()

def get_filtered_signal(signal, low_cutoff=0.75, high_cutoff=4.0, fs=30):
    """
    Applies a Butterworth bandpass filter to the raw signal.
    This is a crucial step to remove noise (like lighting changes or movement)
    and isolate the frequency range where human heartbeats occur (approx. 45-240 bpm).
    """
    if signal is None or len(signal) < 20: # A minimum length is required for filtering
        return None
        
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    # Use a 1st order Butterworth filter
    b, a = butter(1, [low, high], btype='band')
    
    try:
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    except ValueError:
        # This can happen if the signal is too short or contains NaNs
        return None

def extract_features(signal):
    """
    Calculates key statistical features from the clean iPPG signal.
    These features numerically describe the shape of the pulse wave and
    will be the input for the machine learning model.
    """
    if signal is None:
        return None
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.ptp(signal) # Peak-to-peak (range) of the signal
    ]

def process_video_for_ippg(video_path):
    """
    This is the main pipeline function for processing a video file. It performs:
    1. Video reading, frame by frame.
    2. Face detection in each frame.
    3. Defining a Region of Interest (ROI) on the forehead.
    4. Averaging the green channel pixel values in the ROI to get the raw signal.
    5. Filtering the raw signal to get the final, clean iPPG waveform.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None

    raw_signal = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for the face detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # Assume the largest detected face is the subject
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]

            # Define the Region of Interest (ROI) on the forehead
            # This area is less prone to noise from facial expressions.
            forehead_x = x + int(0.2 * w)
            forehead_y = y + int(0.1 * h)
            forehead_w = int(0.6 * w)
            forehead_h = int(0.15 * h)
            
            roi = frame[forehead_y : forehead_y + forehead_h, forehead_x : forehead_x + forehead_w]

            if roi.size > 0:
                # The iPPG signal is strongest in the green channel
                green_channel_mean = np.mean(roi[:, :, 1])
                raw_signal.append(green_channel_mean)
            else:
                # Append a neutral value if ROI is empty for some reason
                raw_signal.append(0)
        else:
            # If no face is detected in a frame, we append 0.
            # The filtering process helps mitigate the impact of these frames.
            raw_signal.append(0)

    cap.release()

    # Filter the raw signal to get a clean iPPG waveform
    filtered_signal = get_filtered_signal(np.array(raw_signal), fs=frame_rate)
    
    return filtered_signal

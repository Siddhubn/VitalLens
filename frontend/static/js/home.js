document.addEventListener('DOMContentLoaded', async () => {
    // --- Get all DOM elements ---
    const webcamElement = document.getElementById('webcam');
    const predictBtn = document.getElementById('predict-btn');
    const reportBtn = document.getElementById('report-btn');
    const statusText = document.getElementById('status-text');
    const faceIndicator = document.getElementById('face-indicator');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsGrid = document.getElementById('results-grid');
    const hrValueElement = document.getElementById('hr-value');
    const bpValueElement = document.getElementById('bp-value');
    const stressValueElement = document.getElementById('stress-value');

    // --- Global variables ---
    let mediaRecorder;
    let recordedChunks = [];
    let faceCheckInterval;
    let model; // Variable to hold the BlazeFace model

    // --- 1. Load the BlazeFace Model ---
    async function loadModel() {
        try {
            statusText.textContent = "Loading face detection model...";
            model = await blazeface.load();
            console.log("BlazeFace model loaded successfully.");
            statusText.textContent = "Ready to start monitoring.";
        } catch (err) {
            console.error("Error loading model:", err);
            statusText.textContent = "Error: Could not load face detection model.";
        }
    }
    
    // --- 2. Initialize Webcam and Start Processes ---
    async function initializeWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamElement.srcObject = stream;
                
                // Wait for the video to start playing to get its dimensions
                webcamElement.onloadedmetadata = () => {
                    predictBtn.disabled = false;
                    
                    // Setup MediaRecorder
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
                    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
                    mediaRecorder.onstop = () => {
                        const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
                        recordedChunks = [];
                        sendToServerForPrediction(videoBlob);
                    };

                    // Start checking for a face if the model is loaded
                    if (model) {
                        faceCheckInterval = setInterval(detectFace, 500); // Check every 500ms
                    }
                };

            } catch (err) {
                console.error("Webcam Error:", err);
                statusText.textContent = "Error: Could not access webcam.";
                faceIndicator.textContent = 'Camera Error';
            }
        }
    }

    // --- 3. NEW Face Detection Logic using TensorFlow.js ---
    async function detectFace() {
        if (!model || webcamElement.readyState < 2) {
            return; // Exit if model or video not ready
        }
        
        try {
            // Get predictions
            const predictions = await model.estimateFaces(webcamElement, false);

            if (predictions.length > 0) {
                faceIndicator.textContent = 'Face Detected';
                faceIndicator.className = 'status-indicator green';
            } else {
                faceIndicator.textContent = 'No Face Detected';
                faceIndicator.className = 'status-indicator red';
            }
        } catch (err) {
            console.error("Error during face detection:", err);
        }
    }

    // --- 4. Handle Measurement ---
    function startMeasurement() {
        if (!faceIndicator.classList.contains('green')) {
            statusText.textContent = "Please ensure your face is detected before starting.";
            setTimeout(() => {
                if(statusText.textContent === "Please ensure your face is detected before starting."){
                   statusText.textContent = "Ready to start monitoring.";
                }
            }, 3000);
            return;
        }

        resultsPlaceholder.style.display = 'block';
        resultsGrid.style.display = 'none';
        predictBtn.disabled = true;
        clearInterval(faceCheckInterval);

        const measurementDuration = 30000; // 30 seconds
        let countdown = measurementDuration / 1000;
        
        statusText.textContent = `Recording... ${countdown}s remaining`;
        const countdownInterval = setInterval(() => {
            countdown--;
            statusText.textContent = `Recording... ${countdown}s remaining`;
            if (countdown <= 0) clearInterval(countdownInterval);
        }, 1000);

        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                statusText.textContent = 'Processing... Please wait.';
            }
        }, measurementDuration);
    }

    // --- 5. Send to Server ---
    async function sendToServerForPrediction(blob) {
        const formData = new FormData();
        formData.append('video_blob', blob, 'recording.webm');
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ error: 'Server returned an error.' }));
                throw new Error(errData.error || `Prediction failed.`);
            }
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error('Prediction Error:', error);
            statusText.textContent = `Error: ${error.message}`;
        } finally {
            predictBtn.disabled = false;
            if (model) {
                faceCheckInterval = setInterval(detectFace, 500);
            }
        }
    }
    
    // --- 6. Display Results ---
    function displayResults(data) {
        resultsPlaceholder.style.display = 'none';
        resultsGrid.style.display = 'flex';
        hrValueElement.textContent = `${data.heart_rate} bpm`;
        bpValueElement.textContent = `${data.systolic_bp}/${data.diastolic_bp} mmHg`;
        
        let stress = 'Normal';
        if (data.heart_rate > 100 || data.systolic_bp > 135) stress = 'High';
        else if (data.heart_rate > 85 || data.systolic_bp > 125) stress = 'Moderate';
        stressValueElement.textContent = stress;

        statusText.textContent = 'Measurement complete. Ready to start monitoring.';
    }
    
    // --- Event Listeners ---
    predictBtn.addEventListener('click', startMeasurement);
    reportBtn.addEventListener('click', () => window.location.href = '/download_report');

    // --- Initialize the application ---
    await loadModel();
    await initializeWebcam();
});

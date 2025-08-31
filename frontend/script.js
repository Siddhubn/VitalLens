// --- DOM Element References ---
const video = document.getElementById('video');
const startMonitoringBtn = document.getElementById('start-monitoring-btn');
const statusText = document.getElementById('status-text');
const faceDetectionStatus = document.getElementById('face-detection-status');
const resultsPlaceholder = document.getElementById('results-placeholder');
const resultsContainer = document.getElementById('results-container');
const processingLoader = document.getElementById('processing-loader');
// [MODIFIED] Added reference to the new download button
const downloadReportBtn = document.getElementById('download-report-btn'); 

// --- Constants and State ---
let monitoringInterval;
let faceDetectionInterval;
const MONITORING_DURATION = 20000; // 20 seconds
// IMPORTANT: Make sure this URL matches your running Python backend
const BACKEND_URL = 'http://127.0.0.1:5000/process';

// --- Face Detection Setup ---
async function loadFaceApiModels() {
    const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights';
    try {
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        console.log("FaceAPI models loaded.");
        startVideo();
    } catch (error) {
        console.error("Error loading FaceAPI models:", error);
        statusText.textContent = "Could not load models.";
    }
}

// --- Camera and Video ---
async function startVideo() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        video.srcObject = stream;
        statusText.textContent = "Initializing camera...";
    } catch (err) {
        console.error("Error accessing camera: ", err);
        statusText.textContent = "Camera access denied.";
        startMonitoringBtn.disabled = true;
    }
}

// --- Core Logic ---
function detectFace() {
    faceDetectionInterval = setInterval(async () => {
        if (!video || video.paused || video.ended) return;
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions());
        if (detections.length > 0) {
            faceDetectionStatus.textContent = 'Face Detected';
            // Note: The below classes were not in your provided CSS. Assuming you have them defined elsewhere.
            // For this example, let's use a simpler style change that works without extra CSS.
            faceDetectionStatus.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
            startMonitoringBtn.disabled = false;
            statusText.textContent = "Ready to start monitoring.";
        } else {
            faceDetectionStatus.textContent = 'No Face Detected';
            faceDetectionStatus.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
            startMonitoringBtn.disabled = true;
            statusText.textContent = "Position your face in the camera.";
        }
    }, 1000);
}

function startMonitoring() {
    startMonitoringBtn.disabled = true;
    startMonitoringBtn.textContent = "Monitoring...";
    clearInterval(faceDetectionInterval);
    resultsContainer.innerHTML = ''; // Clear previous results
    // [MODIFIED] Hide the download button when a new monitoring session starts
    downloadReportBtn.style.display = 'none';
    
    let timeLeft = MONITORING_DURATION / 1000;
    statusText.textContent = `Recording... ${timeLeft}s remaining`;

    monitoringInterval = setInterval(() => {
        timeLeft--;
        statusText.textContent = `Recording... ${timeLeft}s remaining`;
        if (timeLeft <= 0) {
            stopMonitoring();
        }
    }, 1000);
}

async function stopMonitoring() {
    clearInterval(monitoringInterval);
    statusText.textContent = "Sending data for processing...";
    startMonitoringBtn.textContent = "Start Monitoring";
    
    // Hide placeholder and show loader
    resultsPlaceholder.style.display = 'none';
    processingLoader.style.display = 'flex';

    try {
        const response = await fetch(BACKEND_URL, { method: 'POST' });
        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        const results = await response.json();
        displayResults(results);
    } catch (error) {
        console.error("Error communicating with backend:", error);
        statusText.textContent = "Could not connect to the server.";
        processingLoader.style.display = 'none';
        resultsPlaceholder.style.display = 'block';
    } finally {
        // Restart face detection for the next session
        detectFace();
    }
}
    
function displayResults(results) {
    processingLoader.style.display = 'none';
    resultsContainer.innerHTML = ''; // Clear container before adding new results
    statusText.textContent = "Measurement complete.";

    // This function creates a single result box element
    const createResultBox = (title, value, iconHtml, delay) => {
        const box = document.createElement('div');
        box.className = 'result-box'; // The class from your CSS for styling and animation
        box.style.animationDelay = `${delay}s`;
        box.innerHTML = `
            ${iconHtml}
            <div>
                <span>${title}:</span>
                <span style="font-weight: bold; margin-left: 8px;">${value}</span>
            </div>
        `;
        resultsContainer.appendChild(box);
    };

    // --- Data mapping and creation of result boxes ---
    const resultsData = [
        {
            title: 'Heart Rate',
            value: `${results.heartRate} bpm`,
            icon: '<i class="fa-solid fa-heart-pulse"></i>',
        },
        {
            title: 'Blood Pressure',
            value: `${results.systolic}/${results.diastolic} mmHg`,
            icon: '<i class="fa-solid fa-droplet"></i>',
        },
        {
            title: 'Stress Level',
            value: results.stress,
            icon: '<i class="fa-solid fa-brain"></i>',
        }
    ];

    // Create and append each box with a staggered animation delay
    resultsData.forEach((data, index) => {
        createResultBox(data.title, data.value, data.icon, index * 0.2);
    });

    // [MODIFIED] Show the download button now that the results are visible
    downloadReportBtn.style.display = 'block';
}

// --- Initialization ---
window.onload = () => {
    // Set initial UI state
    startMonitoringBtn.disabled = true;
    statusText.textContent = "Loading models, please wait...";

    // --- Event Listeners ---
    startMonitoringBtn.addEventListener('click', startMonitoring);
    video.addEventListener('play', detectFace);
    
    // Start the process by loading the face detection models
    loadFaceApiModels();
};
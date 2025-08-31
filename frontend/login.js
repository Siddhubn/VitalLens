// --- DOM Element References ---
const loginForm = document.getElementById('login-form');
const emailInput = document.getElementById('email');
const passwordInput = document.getElementById('password');
const errorMessage = document.getElementById('error-message');

const LOGIN_URL = 'http://127.0.0.1:5000/login';

// --- Event Listener for Form Submission ---
loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    errorMessage.textContent = '';

    const email = emailInput.value;
    const password = passwordInput.value;

    try {
        const response = await fetch(LOGIN_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            console.log('Login successful, redirecting...');
            window.location.href = 'index.html';
        } else {
            errorMessage.textContent = data.message || 'Login failed.';
        }
    } catch (error) {
        console.error('Login request failed:', error);
        errorMessage.textContent = 'Could not connect to the server.';
    }
});

// Constants and configurations
const CONFIG = {
    suspiciousPatterns: [
        {
            pattern: /[0-9]+(?=[a-z])/i,
            message: "Contains suspicious number-letter combinations"
        },
        {
            pattern: /\.(tk|ml|ga|cf|gq)$/i,
            message: "Uses suspicious top-level domain"
        },
        {
            pattern: /(?:paypal|amazon|google|microsoft|apple).*\.[a-z]{2,3}\.[a-z]{2}$/i,
            message: "Potential impersonation of known brand"
        },
        {
            pattern: /[a-z0-9]{25,}/i,
            message: "Contains unusually long string of characters"
        },
        {
            pattern: /(?:secure|login|signin|banking|account).*\.[a-z]{2,3}\.[a-z]{2}$/i,
            message: "Uses suspicious security-related terms"
        },
        {
            pattern: /[А-Яа-я]/,
            message: "Contains Cyrillic characters"
        }
    ],
    animationDuration: 300,
    minAnalysisTime: 800 // Minimum time to show loading state
};

// DOM Elements
const elements = {
    urlInput: document.getElementById('urlInput'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    result: document.getElementById('result'),
    loader: document.getElementById('loader'),
    supportBtn: document.getElementById('supportBtn'),
    supportModal: document.getElementById('supportModal'),
    closeModal: document.getElementById('closeModal'),
    supportForm: document.getElementById('supportForm'),
    currentYear: document.getElementById('currentYear')
};

// Event Listeners
document.addEventListener('DOMContentLoaded', initializeApp);
elements.analyzeBtn.addEventListener('click', handleURLAnalysis);
elements.urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleURLAnalysis();
});
elements.supportBtn.addEventListener('click', openSupport);
elements.closeModal.addEventListener('click', closeSupport);
elements.supportForm.addEventListener('submit', handleSupport);
window.addEventListener('click', (e) => {
    if (e.target === elements.supportModal) closeSupport();
});

// Initialize application
function initializeApp() {
    elements.currentYear.textContent = new Date().getFullYear();
    setupFormValidation();
    setupInputValidation();
}

// URL Analysis Functions
async function handleURLAnalysis() {
    const url = elements.urlInput.value.trim();
    
    if (!url) {
        showResult('danger', 'Please enter a URL to analyze');
        return;
    }

    showLoader();
    console.log(url);
    try {
        const analysisStart = Date.now();
        const result = await analyzeURL(url);
        
        // Ensure loader shows for at least minAnalysisTime
        const analysisTime = Date.now() - analysisStart;
        if (analysisTime < CONFIG.minAnalysisTime) {
            await new Promise(resolve => setTimeout(resolve, CONFIG.minAnalysisTime - analysisTime));
        }

        displayAnalysisResult(result);
    } catch (error) {
        showResult('danger', error.message);
    } finally {
        hideLoader();
    }
}

function isValidURL(text) {
    const urlPattern = /^(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$/;
    return urlPattern.test(text);
}

async function analyzeURL(urlString) {
    let url;
    if (!isValidURL(urlString)) {
        throw new Error("Please enter a valid URL");
    }
    // try {
    //     url = new URL(urlString);
    // } catch (e) {
    //     throw new Error('Please enter a valid URL');
    // }

    showLoader();

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: urlString })
        });

        if (!response.ok) {
            throw new Error("Error fetching analysis results");
        }

        const data = await response.json();
        return {
            isSuspicious: data.prediction === "Phishing",
            warnings: [`The backend detected this URL as: ${data.prediction}`],
            url: urlString
        };
    } catch (error) {
        throw new Error("Failed to connect to phishing detection server.");
    } finally {
        hideLoader();
    }
}

function displayAnalysisResult(result) {
    const { isSuspicious, warnings, url } = result;

    if (isSuspicious) {
        const warningsList = warnings
            .map(warning => `<li>${warning}</li>`)
            .join('');

        showResult('danger', `
            <strong>⚠️ Warning: This URL shows signs of being potentially malicious!</strong>
            <p>Analysis of: ${escapeHtml(url)}</p>
            <p>Detected issues:</p>
            <ul>${warningsList}</ul>
            <p>Exercise caution and verify the source before proceeding.</p>
        `);
    } else {
        showResult('safe', `
            <strong>✓ No obvious signs of phishing detected.</strong>
            <p>Analysis of: ${escapeHtml(url)}</p>
            <p>However, always be cautious when clicking on unknown links.</p>
        `);
    }
}

// Support Form Functions
function setupFormValidation() {
    const form = elements.supportForm;
    const inputs = form.querySelectorAll('input, textarea');

    inputs.forEach(input => {
        input.addEventListener('blur', () => validateField(input));
        input.addEventListener('input', () => {
            const errorElement = input.parentElement.querySelector('.error-message');
            errorElement.textContent = '';
        });
    });
}

function validateField(field) {
    const errorElement = field.parentElement.querySelector('.error-message');
    let errorMessage = '';

    if (!field.value.trim()) {
        errorMessage = 'This field is required';
    } else if (field.type === 'email' && !isValidEmail(field.value)) {
        errorMessage = 'Please enter a valid email address';
    }

    errorElement.textContent = errorMessage;
    return !errorMessage;
}

function setupInputValidation() {
    elements.urlInput.addEventListener('input', (e) => {
        const input = e.target;
        input.classList.toggle('invalid', input.value && !isValidURL(input.value));
    });
}

// Event Handlers
async function handleSupport(event) {
    event.preventDefault();
    
    const form = event.target;
    const inputs = form.querySelectorAll('input, textarea');
    let isValid = true;

    inputs.forEach(input => {
        if (!validateField(input)) {
            isValid = false;
        }
    });

    if (!isValid) return;

    // Simulate form submission
    const submitButton = form.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.textContent = 'Sending...';

    try {
        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate API call
        showNotification('Support request submitted successfully!');
        form.reset();
        closeSupport();
    } catch (error) {
        showNotification('Failed to submit support request. Please try again.', 'error');
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Submit Request';
    }
}

// Utility Functions
function showLoader() {
    elements.loader.classList.remove('hidden');
    elements.result.classList.add('hidden');
}

function hideLoader() {
    elements.loader.classList.add('hidden');
}

function showResult(type, content) {
    const resultElement = elements.result;
    resultElement.className = `result ${type}`;
    resultElement.innerHTML = content;
    resultElement.classList.remove('hidden');
    resultElement.classList.add('visible');
}

function openSupport() {
    elements.supportModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
}

function closeSupport() {
    elements.supportModal.style.display = 'none';
    document.body.style.overflow = '';
}

function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function isValidURL(string) {
    try {
        new URL(string);
        return true;
    } catch {
        return false;
    }
}

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function showNotification(message, type = 'success') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after delay
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
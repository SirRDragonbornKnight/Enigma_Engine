/**
 * Enigma Engine Web Dashboard
 * Common JavaScript utilities
 */

// API helper functions
const API = {
    async get(endpoint) {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    },
    
    async post(endpoint, data) {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }
};

// Show notification
function showNotification(message, type = 'info') {
    // Simple console log for now
    // Could be enhanced with toast notifications
    console.log(`[${type.toUpperCase()}] ${message}`);
}

// Format date/time
function formatDateTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Enigma Engine Web Dashboard loaded');
});

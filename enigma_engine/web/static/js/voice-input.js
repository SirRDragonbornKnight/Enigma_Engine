/**
 * Web Voice Input Module
 * ======================
 * 
 * Browser-based speech-to-text input for Enigma chat.
 * 
 * Supports two modes:
 * 1. Web Speech API (browser-based, no server required)
 * 2. WebRTC + Server transcription (for better accuracy)
 * 
 * Usage:
 *     const voiceInput = new VoiceInput();
 *     voiceInput.onResult = (text) => console.log('You said:', text);
 *     voiceInput.start();
 */

class VoiceInput {
    constructor(options = {}) {
        this.options = {
            mode: 'auto',           // 'auto', 'browser', 'server'
            language: 'en-US',
            continuous: false,      // Keep listening after result
            interimResults: true,   // Show partial results
            serverEndpoint: '/api/voice/transcribe',
            ...options
        };
        
        // State
        this.isListening = false;
        this.recognition = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        // Callbacks
        this.onStart = null;
        this.onStop = null;
        this.onResult = null;
        this.onInterim = null;
        this.onError = null;
        
        // Check capabilities
        this.hasBrowserSpeech = 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
        this.hasMediaRecorder = 'MediaRecorder' in window;
        
        // Select mode
        if (this.options.mode === 'auto') {
            this.options.mode = this.hasBrowserSpeech ? 'browser' : 'server';
        }
    }
    
    /**
     * Start listening for voice input
     */
    async start() {
        if (this.isListening) return;
        
        if (this.options.mode === 'browser') {
            return this._startBrowserRecognition();
        } else {
            return this._startServerRecognition();
        }
    }
    
    /**
     * Stop listening
     */
    stop() {
        if (!this.isListening) return;
        
        if (this.recognition) {
            this.recognition.stop();
        }
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        this.isListening = false;
        this._trigger('onStop');
    }
    
    /**
     * Toggle listening state
     */
    toggle() {
        if (this.isListening) {
            this.stop();
        } else {
            this.start();
        }
    }
    
    /**
     * Browser-based speech recognition using Web Speech API
     */
    _startBrowserRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
        if (!SpeechRecognition) {
            this._trigger('onError', 'Speech recognition not supported in this browser');
            return false;
        }
        
        this.recognition = new SpeechRecognition();
        this.recognition.lang = this.options.language;
        this.recognition.continuous = this.options.continuous;
        this.recognition.interimResults = this.options.interimResults;
        
        this.recognition.onstart = () => {
            this.isListening = true;
            this._trigger('onStart');
        };
        
        this.recognition.onend = () => {
            this.isListening = false;
            this._trigger('onStop');
            
            // Auto-restart if continuous
            if (this.options.continuous && this.isListening) {
                this.recognition.start();
            }
        };
        
        this.recognition.onresult = (event) => {
            let finalTranscript = '';
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            if (interimTranscript && this.options.interimResults) {
                this._trigger('onInterim', interimTranscript);
            }
            
            if (finalTranscript) {
                this._trigger('onResult', finalTranscript);
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this._trigger('onError', event.error);
            this.isListening = false;
        };
        
        try {
            this.recognition.start();
            return true;
        } catch (error) {
            this._trigger('onError', error.message);
            return false;
        }
    }
    
    /**
     * Server-based recognition via WebRTC/MediaRecorder
     */
    async _startServerRecognition() {
        if (!this.hasMediaRecorder) {
            this._trigger('onError', 'MediaRecorder not supported');
            return false;
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: this._getSupportedMimeType()
            });
            
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstart = () => {
                this.isListening = true;
                this._trigger('onStart');
            };
            
            this.mediaRecorder.onstop = async () => {
                this.isListening = false;
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
                
                // Send to server for transcription
                if (this.audioChunks.length > 0) {
                    await this._sendToServer();
                }
                
                this._trigger('onStop');
            };
            
            // Start recording
            this.mediaRecorder.start(1000); // Collect data every second
            
            return true;
            
        } catch (error) {
            console.error('Microphone access error:', error);
            this._trigger('onError', 'Microphone access denied');
            return false;
        }
    }
    
    /**
     * Send recorded audio to server for transcription
     */
    async _sendToServer() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        
        // Convert to base64 for simpler transfer
        const reader = new FileReader();
        
        return new Promise((resolve, reject) => {
            reader.onloadend = async () => {
                try {
                    const base64Audio = reader.result.split(',')[1];
                    
                    const response = await fetch(this.options.serverEndpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            audio: base64Audio,
                            format: 'webm',
                            language: this.options.language
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success && data.text) {
                        this._trigger('onResult', data.text);
                    } else if (data.error) {
                        this._trigger('onError', data.error);
                    }
                    
                    resolve();
                    
                } catch (error) {
                    console.error('Transcription error:', error);
                    this._trigger('onError', 'Transcription failed');
                    reject(error);
                }
            };
            
            reader.onerror = () => reject(reader.error);
            reader.readAsDataURL(audioBlob);
        });
    }
    
    /**
     * Get supported audio MIME type
     */
    _getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/mp4'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        return 'audio/webm';
    }
    
    /**
     * Trigger a callback if defined
     */
    _trigger(callbackName, ...args) {
        if (this[callbackName] && typeof this[callbackName] === 'function') {
            this[callbackName](...args);
        }
    }
    
    /**
     * Get current capabilities
     */
    getCapabilities() {
        return {
            hasBrowserSpeech: this.hasBrowserSpeech,
            hasMediaRecorder: this.hasMediaRecorder,
            currentMode: this.options.mode,
            language: this.options.language
        };
    }
}


/**
 * Voice Input Button Component
 * 
 * Creates a microphone button that can be added to any input area.
 */
class VoiceInputButton {
    constructor(targetInput, options = {}) {
        this.targetInput = typeof targetInput === 'string' 
            ? document.querySelector(targetInput) 
            : targetInput;
            
        this.options = {
            buttonClass: 'voice-input-btn',
            listeningClass: 'listening',
            buttonContent: '🎤',
            listeningContent: '⏹️',
            position: 'after',  // 'before', 'after', 'replace'
            appendResult: true, // Append to existing text or replace
            ...options
        };
        
        this.voiceInput = new VoiceInput(options.voiceOptions || {});
        this.button = null;
        this.statusEl = null;
        
        this._createButton();
        this._setupCallbacks();
    }
    
    _createButton() {
        // Create button
        this.button = document.createElement('button');
        this.button.type = 'button';
        this.button.className = this.options.buttonClass;
        this.button.innerHTML = this.options.buttonContent;
        this.button.title = 'Click to speak';
        
        // Create status indicator
        this.statusEl = document.createElement('span');
        this.statusEl.className = 'voice-status';
        this.statusEl.style.display = 'none';
        
        // Position button
        if (this.options.position === 'before') {
            this.targetInput.parentNode.insertBefore(this.button, this.targetInput);
        } else if (this.options.position === 'after') {
            this.targetInput.parentNode.insertBefore(this.button, this.targetInput.nextSibling);
        }
        
        // Add status after button
        this.button.parentNode.insertBefore(this.statusEl, this.button.nextSibling);
        
        // Click handler
        this.button.addEventListener('click', () => this.voiceInput.toggle());
    }
    
    _setupCallbacks() {
        this.voiceInput.onStart = () => {
            this.button.classList.add(this.options.listeningClass);
            this.button.innerHTML = this.options.listeningContent;
            this.button.title = 'Click to stop';
            this.statusEl.textContent = 'Listening...';
            this.statusEl.style.display = 'inline';
        };
        
        this.voiceInput.onStop = () => {
            this.button.classList.remove(this.options.listeningClass);
            this.button.innerHTML = this.options.buttonContent;
            this.button.title = 'Click to speak';
            this.statusEl.style.display = 'none';
        };
        
        this.voiceInput.onInterim = (text) => {
            this.statusEl.textContent = text;
        };
        
        this.voiceInput.onResult = (text) => {
            if (this.options.appendResult) {
                const current = this.targetInput.value;
                this.targetInput.value = current + (current ? ' ' : '') + text;
            } else {
                this.targetInput.value = text;
            }
            
            // Trigger input event
            this.targetInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            this.statusEl.textContent = 'Got: ' + text.substring(0, 30) + '...';
            setTimeout(() => {
                this.statusEl.style.display = 'none';
            }, 2000);
        };
        
        this.voiceInput.onError = (error) => {
            this.statusEl.textContent = 'Error: ' + error;
            this.statusEl.style.display = 'inline';
            setTimeout(() => {
                this.statusEl.style.display = 'none';
            }, 3000);
        };
    }
    
    /**
     * Start listening programmatically
     */
    start() {
        this.voiceInput.start();
    }
    
    /**
     * Stop listening
     */
    stop() {
        this.voiceInput.stop();
    }
    
    /**
     * Destroy the button and clean up
     */
    destroy() {
        this.voiceInput.stop();
        this.button.remove();
        this.statusEl.remove();
    }
}


// Default styles (inject if not already present)
(function injectStyles() {
    if (document.getElementById('voice-input-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'voice-input-styles';
    style.textContent = `
        .voice-input-btn {
            background: var(--light, #f0f0f0);
            border: 1px solid var(--border, #ddd);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .voice-input-btn:hover {
            background: var(--border, #ddd);
            transform: scale(1.05);
        }
        
        .voice-input-btn.listening {
            background: #ff4444;
            border-color: #ff4444;
            animation: pulse 1.5s infinite;
        }
        
        .voice-status {
            font-size: 0.85rem;
            color: var(--text-muted, #666);
            margin-left: 0.5rem;
            font-style: italic;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
    `;
    document.head.appendChild(style);
})();


// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { VoiceInput, VoiceInputButton };
}

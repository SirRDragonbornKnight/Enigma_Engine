/**
 * ForgeAI Web Client
 * 
 * Handles WebSocket connections, chat interface, and API interactions.
 */

class ForgeWebClient {
    constructor() {
        this.ws = null;
        this.chatHistory = [];
        this.settings = {
            temperature: 0.8,
            maxTokens: 200
        };
        this.token = this.getTokenFromURL();
        this.useWebSocket = true;
        
        // DOM elements
        this.chatArea = document.getElementById('chatArea');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.typingIndicator = document.getElementById('typingIndicator');
        
        // Menu elements
        this.menuBtn = document.getElementById('menuBtn');
        this.sideMenu = document.getElementById('sideMenu');
        this.closeMenu = document.getElementById('closeMenu');
        
        // Settings modal
        this.settingsModal = document.getElementById('settingsModal');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.closeSettings = document.getElementById('closeSettings');
        this.tempSlider = document.getElementById('tempSlider');
        this.tempValue = document.getElementById('tempValue');
        this.maxTokensSlider = document.getElementById('maxTokensSlider');
        this.maxTokensValue = document.getElementById('maxTokensValue');
        this.saveSettings = document.getElementById('saveSettings');
        
        this.init();
    }
    
    init() {
        this.connect();
        this.setupEventListeners();
        this.loadSettings();
    }
    
    getTokenFromURL() {
        const params = new URLSearchParams(window.location.search);
        return params.get('token');
    }
    
    connect() {
        try {
            // Determine WebSocket URL
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/chat${this.token ? '?token=' + this.token : ''}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.updateStatus('connected', 'Connected (WebSocket)');
                console.log('WebSocket connected');
            };
            
            this.ws.onclose = () => {
                this.updateStatus('disconnected', 'Disconnected');
                console.log('WebSocket disconnected');
                
                // Try to reconnect after 3 seconds
                setTimeout(() => this.connect(), 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.useWebSocket = false;
                this.updateStatus('connected', 'Connected (REST API)');
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
        } catch (error) {
            console.error('Failed to connect via WebSocket:', error);
            this.useWebSocket = false;
            this.updateStatus('connected', 'Connected (REST API)');
        }
    }
    
    handleWebSocketMessage(data) {
        const { type, content, typing } = data;
        
        switch (type) {
            case 'response':
                this.typingIndicator.classList.remove('active');
                this.addMessage(content, 'ai');
                break;
            case 'typing':
                if (typing) {
                    this.typingIndicator.classList.add('active');
                } else {
                    this.typingIndicator.classList.remove('active');
                }
                break;
            case 'error':
                this.typingIndicator.classList.remove('active');
                this.addMessage(`Error: ${content}`, 'system');
                break;
            case 'pong':
                // Keep-alive response
                break;
        }
    }
    
    setupEventListeners() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Voice button (placeholder)
        this.voiceBtn.addEventListener('click', () => {
            this.startVoiceInput();
        });
        
        // Menu
        this.menuBtn.addEventListener('click', () => {
            this.sideMenu.classList.add('open');
        });
        this.closeMenu.addEventListener('click', () => {
            this.sideMenu.classList.remove('open');
        });
        
        // Menu items
        document.getElementById('newChatBtn').addEventListener('click', () => {
            this.newChat();
            this.sideMenu.classList.remove('open');
        });
        
        document.getElementById('conversationsBtn').addEventListener('click', () => {
            this.showConversations();
            this.sideMenu.classList.remove('open');
        });
        
        this.settingsBtn.addEventListener('click', () => {
            this.showSettings();
            this.sideMenu.classList.remove('open');
        });
        
        document.getElementById('statsBtn').addEventListener('click', () => {
            this.showStats();
            this.sideMenu.classList.remove('open');
        });
        
        // Settings modal
        this.closeSettings.addEventListener('click', () => {
            this.settingsModal.classList.remove('active');
        });
        
        this.tempSlider.addEventListener('input', (e) => {
            this.tempValue.textContent = e.target.value;
        });
        
        this.maxTokensSlider.addEventListener('input', (e) => {
            this.maxTokensValue.textContent = e.target.value;
        });
        
        this.saveSettings.addEventListener('click', () => {
            this.saveSettingsToServer();
        });
        
        // Keep-alive ping every 30 seconds
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }
    
    updateStatus(status, text) {
        this.statusDot.className = 'status-dot ' + status;
        this.statusText.textContent = text;
    }
    
    addMessage(text, type) {
        const msg = document.createElement('div');
        msg.className = 'message ' + type;
        msg.textContent = text;
        
        if (type !== 'system') {
            const time = document.createElement('div');
            time.className = 'message-time';
            time.textContent = new Date().toLocaleTimeString();
            msg.appendChild(time);
        }
        
        this.chatArea.appendChild(msg);
        this.chatArea.scrollTop = this.chatArea.scrollHeight;
        
        // Add to history
        if (type !== 'system') {
            this.chatHistory.push({ text, type, timestamp: new Date() });
        }
    }
    
    async sendMessage() {
        const text = this.messageInput.value.trim();
        if (!text) return;
        
        // Clear input and disable button
        this.messageInput.value = '';
        this.sendBtn.disabled = true;
        
        // Add user message
        this.addMessage(text, 'user');
        
        // Show typing indicator
        this.typingIndicator.classList.add('active');
        
        try {
            if (this.useWebSocket && this.ws && this.ws.readyState === WebSocket.OPEN) {
                // Send via WebSocket
                this.ws.send(JSON.stringify({
                    type: 'message',
                    content: text
                }));
            } else {
                // Fall back to REST API
                await this.sendViaAPI(text);
            }
        } catch (error) {
            this.typingIndicator.classList.remove('active');
            this.addMessage('Failed to send message: ' + error.message, 'system');
        } finally {
            this.sendBtn.disabled = false;
        }
    }
    
    async sendViaAPI(text) {
        try {
            const url = `/api/chat${this.token ? '?token=' + this.token : ''}`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    content: text,
                    max_tokens: this.settings.maxTokens,
                    temperature: this.settings.temperature
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.typingIndicator.classList.remove('active');
            
            if (data.success) {
                this.addMessage(data.response, 'ai');
            } else {
                this.addMessage('Error: ' + (data.error || 'Unknown error'), 'system');
            }
        } catch (error) {
            this.typingIndicator.classList.remove('active');
            this.addMessage('Connection error: ' + error.message, 'system');
        }
    }
    
    newChat() {
        this.chatHistory = [];
        this.chatArea.innerHTML = '<div class="message system">New conversation started.</div>';
    }
    
    async showConversations() {
        try {
            const url = `/api/conversations${this.token ? '?token=' + this.token : ''}`;
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.conversations && data.conversations.length > 0) {
                let message = 'Conversations:\n';
                data.conversations.forEach(conv => {
                    message += `\n- ${conv.name} (${conv.message_count} messages)`;
                });
                this.addMessage(message, 'system');
            } else {
                this.addMessage('No saved conversations', 'system');
            }
        } catch (error) {
            this.addMessage('Failed to load conversations: ' + error.message, 'system');
        }
    }
    
    showSettings() {
        this.settingsModal.classList.add('active');
    }
    
    async showStats() {
        try {
            const url = `/api/stats${this.token ? '?token=' + this.token : ''}`;
            const response = await fetch(url);
            const data = await response.json();
            
            const message = `System Statistics:
CPU: ${data.cpu_percent}%
Memory: ${data.memory_percent}%
Disk: ${data.disk_percent}%`;
            
            this.addMessage(message, 'system');
        } catch (error) {
            this.addMessage('Failed to load stats: ' + error.message, 'system');
        }
    }
    
    loadSettings() {
        // Load from localStorage
        const saved = localStorage.getItem('forgeai_settings');
        if (saved) {
            try {
                this.settings = JSON.parse(saved);
                this.tempSlider.value = this.settings.temperature;
                this.tempValue.textContent = this.settings.temperature;
                this.maxTokensSlider.value = this.settings.maxTokens;
                this.maxTokensValue.textContent = this.settings.maxTokens;
            } catch (error) {
                console.error('Failed to load settings:', error);
            }
        }
    }
    
    startVoiceInput() {
        // Check for Web Speech API support
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.addMessage('Voice input not supported in this browser. Try Chrome or Edge.', 'system');
            return;
        }
        
        // Create recognition instance
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.lang = 'en-US';
        
        // Visual feedback
        this.voiceBtn.classList.add('listening');
        this.voiceBtn.style.backgroundColor = '#ff4444';
        this.addMessage('Listening...', 'system');
        
        let finalTranscript = '';
        
        recognition.onresult = (event) => {
            let interimTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            // Show interim results in input field
            if (interimTranscript) {
                this.messageInput.value = finalTranscript + interimTranscript;
            }
        };
        
        recognition.onend = () => {
            // Reset visual feedback
            this.voiceBtn.classList.remove('listening');
            this.voiceBtn.style.backgroundColor = '';
            
            if (finalTranscript) {
                this.messageInput.value = finalTranscript;
                // Auto-send the message
                this.sendMessage();
            } else {
                this.addMessage('No speech detected. Try again.', 'system');
            }
        };
        
        recognition.onerror = (event) => {
            this.voiceBtn.classList.remove('listening');
            this.voiceBtn.style.backgroundColor = '';
            
            let errorMsg = 'Voice recognition error';
            switch (event.error) {
                case 'no-speech':
                    errorMsg = 'No speech detected. Try again.';
                    break;
                case 'audio-capture':
                    errorMsg = 'No microphone found. Check your device.';
                    break;
                case 'not-allowed':
                    errorMsg = 'Microphone access denied. Allow access in browser settings.';
                    break;
                default:
                    errorMsg = `Voice error: ${event.error}`;
            }
            this.addMessage(errorMsg, 'system');
        };
        
        // Start recognition
        try {
            recognition.start();
        } catch (error) {
            this.addMessage('Could not start voice recognition: ' + error.message, 'system');
            this.voiceBtn.classList.remove('listening');
            this.voiceBtn.style.backgroundColor = '';
        }
    }
    
    async saveSettingsToServer() {
        this.settings.temperature = parseFloat(this.tempSlider.value);
        this.settings.maxTokens = parseInt(this.maxTokensSlider.value);
        
        // Save to localStorage
        localStorage.setItem('forgeai_settings', JSON.stringify(this.settings));
        
        // Save to server
        try {
            const url = `/api/settings${this.token ? '?token=' + this.token : ''}`;
            const response = await fetch(url, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    settings: {
                        temperature: this.settings.temperature,
                        max_gen: this.settings.maxTokens
                    }
                })
            });
            
            if (response.ok) {
                this.addMessage('Settings saved successfully', 'system');
                this.settingsModal.classList.remove('active');
            } else {
                throw new Error('Failed to save settings');
            }
        } catch (error) {
            this.addMessage('Failed to save settings: ' + error.message, 'system');
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.forgeClient = new ForgeWebClient();
});

// Register service worker for PWA
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then(registration => {
                console.log('ServiceWorker registered:', registration);
            })
            .catch(error => {
                console.log('ServiceWorker registration failed:', error);
            });
    });
}

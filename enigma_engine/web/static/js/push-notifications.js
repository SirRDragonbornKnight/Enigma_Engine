/**
 * Push Notifications Module for Enigma Web Interface
 * ===================================================
 * 
 * Handles Web Push API subscription and notification management.
 * 
 * Usage:
 *     const push = new PushNotifications();
 *     await push.initialize();
 *     
 *     if (push.isSupported) {
 *         const subscribed = await push.subscribe();
 *         if (subscribed) {
 *             console.log('Push notifications enabled!');
 *         }
 *     }
 */

class PushNotifications {
    constructor(options = {}) {
        this.options = {
            serverPublicKey: null,  // VAPID public key from server
            subscribeEndpoint: '/api/push/subscribe',
            unsubscribeEndpoint: '/api/push/unsubscribe',
            testEndpoint: '/api/push/test',
            ...options
        };
        
        this.registration = null;
        this.subscription = null;
        this.permissionStatus = null;
        
        // Callbacks
        this.onPermissionChange = null;
        this.onSubscriptionChange = null;
        this.onError = null;
    }
    
    /**
     * Check if push notifications are supported
     */
    get isSupported() {
        return 'serviceWorker' in navigator && 
               'PushManager' in window &&
               'Notification' in window;
    }
    
    /**
     * Get current permission status
     */
    get permission() {
        if (!('Notification' in window)) return 'unsupported';
        return Notification.permission;  // 'granted', 'denied', 'default'
    }
    
    /**
     * Check if currently subscribed
     */
    get isSubscribed() {
        return this.subscription !== null;
    }
    
    /**
     * Initialize push notification system
     */
    async initialize() {
        if (!this.isSupported) {
            console.warn('Push notifications not supported');
            return false;
        }
        
        try {
            // Get service worker registration
            this.registration = await navigator.serviceWorker.ready;
            
            // Get existing subscription
            this.subscription = await this.registration.pushManager.getSubscription();
            
            // Get VAPID key from server if not provided
            if (!this.options.serverPublicKey) {
                await this._fetchServerPublicKey();
            }
            
            console.log('Push notifications initialized', {
                subscribed: this.isSubscribed,
                permission: this.permission
            });
            
            return true;
            
        } catch (error) {
            console.error('Push initialization failed:', error);
            this._triggerError(error);
            return false;
        }
    }
    
    /**
     * Request permission and subscribe to push notifications
     */
    async subscribe() {
        if (!this.isSupported || !this.registration) {
            console.warn('Push not initialized');
            return false;
        }
        
        try {
            // Request permission
            const permission = await Notification.requestPermission();
            
            if (permission !== 'granted') {
                console.warn('Notification permission denied');
                this._triggerPermissionChange(permission);
                return false;
            }
            
            this._triggerPermissionChange('granted');
            
            // Subscribe to push manager
            const applicationServerKey = this._urlBase64ToUint8Array(
                this.options.serverPublicKey
            );
            
            this.subscription = await this.registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: applicationServerKey
            });
            
            // Send subscription to server
            await this._sendSubscriptionToServer(this.subscription);
            
            console.log('Push subscription successful');
            this._triggerSubscriptionChange(true);
            
            return true;
            
        } catch (error) {
            console.error('Push subscription failed:', error);
            this._triggerError(error);
            return false;
        }
    }
    
    /**
     * Unsubscribe from push notifications
     */
    async unsubscribe() {
        if (!this.subscription) {
            return true;
        }
        
        try {
            // Remove from server first
            await this._removeSubscriptionFromServer(this.subscription);
            
            // Unsubscribe locally
            await this.subscription.unsubscribe();
            this.subscription = null;
            
            console.log('Push unsubscribed');
            this._triggerSubscriptionChange(false);
            
            return true;
            
        } catch (error) {
            console.error('Push unsubscribe failed:', error);
            this._triggerError(error);
            return false;
        }
    }
    
    /**
     * Toggle subscription state
     */
    async toggle() {
        if (this.isSubscribed) {
            return await this.unsubscribe();
        } else {
            return await this.subscribe();
        }
    }
    
    /**
     * Send a test notification
     */
    async sendTest(message = 'Test notification from Enigma') {
        try {
            const response = await fetch(this.options.testEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            
            return response.ok;
            
        } catch (error) {
            console.error('Test notification failed:', error);
            return false;
        }
    }
    
    /**
     * Get subscription details for debugging
     */
    getSubscriptionInfo() {
        if (!this.subscription) return null;
        
        const json = this.subscription.toJSON();
        return {
            endpoint: json.endpoint,
            expirationTime: json.expirationTime,
            keys: json.keys ? {
                p256dh: json.keys.p256dh ? '***' : null,
                auth: json.keys.auth ? '***' : null
            } : null
        };
    }
    
    // Private methods
    
    async _fetchServerPublicKey() {
        try {
            const response = await fetch('/api/push/vapid-key');
            const data = await response.json();
            
            if (data.publicKey) {
                this.options.serverPublicKey = data.publicKey;
            }
        } catch (error) {
            console.warn('Could not fetch VAPID key:', error);
        }
    }
    
    async _sendSubscriptionToServer(subscription) {
        const response = await fetch(this.options.subscribeEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subscription: subscription.toJSON()
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to save subscription on server');
        }
    }
    
    async _removeSubscriptionFromServer(subscription) {
        try {
            await fetch(this.options.unsubscribeEndpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    endpoint: subscription.endpoint
                })
            });
        } catch (error) {
            console.warn('Failed to remove subscription from server:', error);
        }
    }
    
    _urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding)
            .replace(/-/g, '+')
            .replace(/_/g, '/');
        
        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);
        
        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        
        return outputArray;
    }
    
    _triggerPermissionChange(status) {
        if (this.onPermissionChange) {
            this.onPermissionChange(status);
        }
    }
    
    _triggerSubscriptionChange(subscribed) {
        if (this.onSubscriptionChange) {
            this.onSubscriptionChange(subscribed);
        }
    }
    
    _triggerError(error) {
        if (this.onError) {
            this.onError(error);
        }
    }
}


/**
 * Push Notification UI Component
 * 
 * Creates a settings panel for push notification management.
 */
class PushNotificationSettings {
    constructor(containerId, options = {}) {
        this.container = typeof containerId === 'string'
            ? document.getElementById(containerId)
            : containerId;
        
        this.push = new PushNotifications(options);
        this.elements = {};
        
        this._createUI();
    }
    
    async initialize() {
        await this.push.initialize();
        this._updateUI();
        
        // Set up callbacks
        this.push.onPermissionChange = () => this._updateUI();
        this.push.onSubscriptionChange = () => this._updateUI();
        this.push.onError = (error) => this._showError(error);
    }
    
    _createUI() {
        this.container.innerHTML = `
            <div class="push-settings">
                <div class="push-header">
                    <h3>Push Notifications</h3>
                    <span class="push-status" id="push-status">Checking...</span>
                </div>
                
                <div class="push-info">
                    <p id="push-description">
                        Enable push notifications to receive AI responses even when the app is in the background.
                    </p>
                </div>
                
                <div class="push-actions">
                    <button id="push-toggle-btn" class="btn btn-primary" disabled>
                        Enable Notifications
                    </button>
                    <button id="push-test-btn" class="btn btn-secondary" disabled>
                        Test Notification
                    </button>
                </div>
                
                <div class="push-error" id="push-error" style="display: none;"></div>
            </div>
        `;
        
        // Cache elements
        this.elements.status = document.getElementById('push-status');
        this.elements.description = document.getElementById('push-description');
        this.elements.toggleBtn = document.getElementById('push-toggle-btn');
        this.elements.testBtn = document.getElementById('push-test-btn');
        this.elements.error = document.getElementById('push-error');
        
        // Event listeners
        this.elements.toggleBtn.addEventListener('click', () => this._handleToggle());
        this.elements.testBtn.addEventListener('click', () => this._handleTest());
    }
    
    _updateUI() {
        const { isSupported, isSubscribed, permission } = this.push;
        
        if (!isSupported) {
            this.elements.status.textContent = 'Not Supported';
            this.elements.status.className = 'push-status error';
            this.elements.description.textContent = 
                'Push notifications are not supported in this browser.';
            return;
        }
        
        if (permission === 'denied') {
            this.elements.status.textContent = 'Blocked';
            this.elements.status.className = 'push-status error';
            this.elements.description.textContent = 
                'Notifications are blocked. Please enable them in your browser settings.';
            this.elements.toggleBtn.disabled = true;
            return;
        }
        
        this.elements.toggleBtn.disabled = false;
        
        if (isSubscribed) {
            this.elements.status.textContent = 'Enabled';
            this.elements.status.className = 'push-status success';
            this.elements.toggleBtn.textContent = 'Disable Notifications';
            this.elements.testBtn.disabled = false;
            this.elements.description.textContent = 
                'Push notifications are enabled. You will receive alerts for AI responses.';
        } else {
            this.elements.status.textContent = 'Disabled';
            this.elements.status.className = 'push-status warning';
            this.elements.toggleBtn.textContent = 'Enable Notifications';
            this.elements.testBtn.disabled = true;
            this.elements.description.textContent = 
                'Enable push notifications to receive AI responses in the background.';
        }
    }
    
    async _handleToggle() {
        this.elements.toggleBtn.disabled = true;
        this.elements.toggleBtn.textContent = 'Please wait...';
        
        await this.push.toggle();
        
        this._updateUI();
    }
    
    async _handleTest() {
        this.elements.testBtn.disabled = true;
        this.elements.testBtn.textContent = 'Sending...';
        
        await this.push.sendTest();
        
        this.elements.testBtn.disabled = false;
        this.elements.testBtn.textContent = 'Test Notification';
    }
    
    _showError(error) {
        this.elements.error.textContent = `Error: ${error.message || error}`;
        this.elements.error.style.display = 'block';
        
        setTimeout(() => {
            this.elements.error.style.display = 'none';
        }, 5000);
    }
}


// Inject default styles
(function injectPushStyles() {
    if (document.getElementById('push-notification-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'push-notification-styles';
    style.textContent = `
        .push-settings {
            padding: 1rem;
            background: var(--light, #f8f9fa);
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .push-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .push-header h3 {
            margin: 0;
            font-size: 1.1rem;
        }
        
        .push-status {
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        
        .push-status.success {
            background: #d4edda;
            color: #155724;
        }
        
        .push-status.warning {
            background: #fff3cd;
            color: #856404;
        }
        
        .push-status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .push-info {
            margin-bottom: 1rem;
        }
        
        .push-info p {
            margin: 0;
            color: var(--text-muted, #666);
            font-size: 0.9rem;
        }
        
        .push-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .push-error {
            margin-top: 1rem;
            padding: 0.75rem;
            background: #f8d7da;
            color: #721c24;
            border-radius: 4px;
            font-size: 0.9rem;
        }
    `;
    document.head.appendChild(style);
})();


// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PushNotifications, PushNotificationSettings };
}

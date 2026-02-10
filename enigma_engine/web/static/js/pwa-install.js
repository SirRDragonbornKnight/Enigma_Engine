/**
 * PWA Install Prompt Module
 * ==========================
 * 
 * Handles Progressive Web App installation prompts with
 * improved UX and customizable UI.
 * 
 * Usage:
 *     const pwaInstall = new PWAInstallPrompt();
 *     pwaInstall.init();
 *     
 *     // Or with custom options
 *     const pwaInstall = new PWAInstallPrompt({
 *         promptDelay: 30000,  // Wait 30 seconds before showing
 *         showBanner: true,
 *         bannerPosition: 'bottom'
 *     });
 */

class PWAInstallPrompt {
    constructor(options = {}) {
        this.options = {
            promptDelay: 5000,         // Delay before showing prompt (ms)
            showBanner: true,          // Show install banner
            bannerPosition: 'bottom',  // 'top' or 'bottom'
            storageKey: 'pwa_install_dismissed',
            dismissDuration: 7 * 24 * 60 * 60 * 1000,  // 7 days
            minVisits: 2,              // Minimum visits before prompting
            visitsKey: 'pwa_visit_count',
            ...options
        };
        
        this.deferredPrompt = null;
        this.isInstalled = false;
        this.banner = null;
        
        // Callbacks
        this.onInstallAvailable = null;
        this.onInstallSuccess = null;
        this.onInstallDismissed = null;
    }
    
    /**
     * Initialize PWA install prompt handling
     */
    init() {
        // Check if already installed
        this.isInstalled = this._checkIfInstalled();
        
        if (this.isInstalled) {
            console.log('PWA already installed');
            return;
        }
        
        // Track visits
        this._trackVisit();
        
        // Listen for install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            this.deferredPrompt = e;
            
            console.log('PWA install prompt available');
            
            if (this.onInstallAvailable) {
                this.onInstallAvailable();
            }
            
            // Show banner after delay if conditions met
            if (this.options.showBanner && this._shouldShowPrompt()) {
                setTimeout(() => this._showBanner(), this.options.promptDelay);
            }
        });
        
        // Listen for successful install
        window.addEventListener('appinstalled', () => {
            this.isInstalled = true;
            this.deferredPrompt = null;
            this._hideBanner();
            
            console.log('PWA installed successfully');
            
            if (this.onInstallSuccess) {
                this.onInstallSuccess();
            }
        });
    }
    
    /**
     * Check if app can be installed
     */
    get canInstall() {
        return this.deferredPrompt !== null && !this.isInstalled;
    }
    
    /**
     * Trigger the native install prompt
     */
    async promptInstall() {
        if (!this.deferredPrompt) {
            console.warn('No install prompt available');
            return false;
        }
        
        try {
            // Show the prompt
            this.deferredPrompt.prompt();
            
            // Wait for user response
            const { outcome } = await this.deferredPrompt.userChoice;
            
            console.log('Install prompt outcome:', outcome);
            
            if (outcome === 'accepted') {
                this.deferredPrompt = null;
                return true;
            } else {
                if (this.onInstallDismissed) {
                    this.onInstallDismissed();
                }
                return false;
            }
            
        } catch (error) {
            console.error('Install prompt error:', error);
            return false;
        }
    }
    
    /**
     * Show custom install banner
     */
    _showBanner() {
        if (this.banner || !this.canInstall) return;
        
        // Create banner
        this.banner = document.createElement('div');
        this.banner.className = `pwa-install-banner ${this.options.bannerPosition}`;
        this.banner.innerHTML = `
            <div class="pwa-banner-content">
                <div class="pwa-banner-icon">
                    <img src="/static/icons/icon-72.png" alt="App Icon" onerror="this.style.display='none'">
                </div>
                <div class="pwa-banner-text">
                    <strong>Install Enigma Engine</strong>
                    <span>Add to home screen for the best experience</span>
                </div>
                <div class="pwa-banner-actions">
                    <button class="pwa-install-btn">Install</button>
                    <button class="pwa-dismiss-btn">Not now</button>
                </div>
            </div>
        `;
        
        // Event listeners
        this.banner.querySelector('.pwa-install-btn').addEventListener('click', () => {
            this.promptInstall();
            this._hideBanner();
        });
        
        this.banner.querySelector('.pwa-dismiss-btn').addEventListener('click', () => {
            this._dismissPrompt();
            this._hideBanner();
        });
        
        // Add to page
        document.body.appendChild(this.banner);
        
        // Animate in
        requestAnimationFrame(() => {
            this.banner.classList.add('visible');
        });
    }
    
    /**
     * Hide install banner
     */
    _hideBanner() {
        if (!this.banner) return;
        
        this.banner.classList.remove('visible');
        
        setTimeout(() => {
            if (this.banner && this.banner.parentNode) {
                this.banner.parentNode.removeChild(this.banner);
            }
            this.banner = null;
        }, 300);
    }
    
    /**
     * Show manual install instructions for iOS/other browsers
     */
    showManualInstructions() {
        const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        
        let instructions = '';
        
        if (isIOS && isSafari) {
            instructions = `
                <h3>Install on iOS</h3>
                <ol>
                    <li>Tap the <strong>Share</strong> button (box with arrow)</li>
                    <li>Scroll down and tap <strong>Add to Home Screen</strong></li>
                    <li>Tap <strong>Add</strong> to confirm</li>
                </ol>
            `;
        } else if (isIOS) {
            instructions = `
                <h3>Install on iOS</h3>
                <p>Please open this page in Safari, then:</p>
                <ol>
                    <li>Tap the <strong>Share</strong> button</li>
                    <li>Tap <strong>Add to Home Screen</strong></li>
                </ol>
            `;
        } else {
            instructions = `
                <h3>Install App</h3>
                <p>Look for the install icon in your browser's address bar, or:</p>
                <ol>
                    <li>Open browser menu (three dots)</li>
                    <li>Select <strong>Install App</strong> or <strong>Add to Home Screen</strong></li>
                </ol>
            `;
        }
        
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'pwa-install-modal';
        modal.innerHTML = `
            <div class="pwa-modal-content">
                ${instructions}
                <button class="pwa-modal-close">Got it</button>
            </div>
        `;
        
        modal.querySelector('.pwa-modal-close').addEventListener('click', () => {
            modal.remove();
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
        
        document.body.appendChild(modal);
    }
    
    /**
     * Create an install button that can be placed anywhere
     */
    createInstallButton(options = {}) {
        const btn = document.createElement('button');
        btn.className = options.className || 'pwa-install-button';
        btn.innerHTML = options.content || 'Install App';
        btn.style.display = this.canInstall ? 'inline-block' : 'none';
        
        btn.addEventListener('click', async () => {
            if (this.canInstall) {
                await this.promptInstall();
            } else {
                this.showManualInstructions();
            }
        });
        
        // Update visibility when install state changes
        const updateVisibility = () => {
            btn.style.display = this.canInstall ? 'inline-block' : 'none';
        };
        
        window.addEventListener('beforeinstallprompt', updateVisibility);
        window.addEventListener('appinstalled', updateVisibility);
        
        return btn;
    }
    
    // Private methods
    
    _checkIfInstalled() {
        // Check if running as standalone PWA
        if (window.matchMedia('(display-mode: standalone)').matches) {
            return true;
        }
        
        // Check iOS standalone mode
        if (window.navigator.standalone === true) {
            return true;
        }
        
        return false;
    }
    
    _shouldShowPrompt() {
        // Check if dismissed recently
        const dismissedAt = localStorage.getItem(this.options.storageKey);
        if (dismissedAt) {
            const elapsed = Date.now() - parseInt(dismissedAt, 10);
            if (elapsed < this.options.dismissDuration) {
                return false;
            }
        }
        
        // Check minimum visits
        const visits = parseInt(localStorage.getItem(this.options.visitsKey) || '0', 10);
        if (visits < this.options.minVisits) {
            return false;
        }
        
        return true;
    }
    
    _trackVisit() {
        const visits = parseInt(localStorage.getItem(this.options.visitsKey) || '0', 10);
        localStorage.setItem(this.options.visitsKey, String(visits + 1));
    }
    
    _dismissPrompt() {
        localStorage.setItem(this.options.storageKey, String(Date.now()));
        
        if (this.onInstallDismissed) {
            this.onInstallDismissed();
        }
    }
}


// Inject default styles
(function injectPWAStyles() {
    if (document.getElementById('pwa-install-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'pwa-install-styles';
    style.textContent = `
        .pwa-install-banner {
            position: fixed;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0;
            z-index: 10000;
            transform: translateY(100%);
            transition: transform 0.3s ease;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
        }
        
        .pwa-install-banner.bottom {
            bottom: 0;
            transform: translateY(100%);
        }
        
        .pwa-install-banner.top {
            top: 0;
            transform: translateY(-100%);
        }
        
        .pwa-install-banner.visible {
            transform: translateY(0);
        }
        
        .pwa-banner-content {
            display: flex;
            align-items: center;
            padding: 1rem;
            max-width: 800px;
            margin: 0 auto;
            gap: 1rem;
        }
        
        .pwa-banner-icon img {
            width: 48px;
            height: 48px;
            border-radius: 8px;
        }
        
        .pwa-banner-text {
            flex: 1;
        }
        
        .pwa-banner-text strong {
            display: block;
            font-size: 1.1rem;
        }
        
        .pwa-banner-text span {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        
        .pwa-banner-actions {
            display: flex;
            gap: 0.5rem;
        }
        
        .pwa-install-btn {
            background: white;
            color: #667eea;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .pwa-install-btn:hover {
            transform: scale(1.05);
        }
        
        .pwa-dismiss-btn {
            background: transparent;
            color: white;
            border: 1px solid rgba(255,255,255,0.5);
            padding: 0.75rem 1rem;
            border-radius: 25px;
            cursor: pointer;
        }
        
        .pwa-dismiss-btn:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .pwa-install-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            z-index: 10001;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }
        
        .pwa-modal-content {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            max-width: 400px;
            width: 100%;
        }
        
        .pwa-modal-content h3 {
            margin-top: 0;
            color: #333;
        }
        
        .pwa-modal-content ol {
            margin: 1rem 0;
            padding-left: 1.5rem;
        }
        
        .pwa-modal-content li {
            margin: 0.5rem 0;
            color: #555;
        }
        
        .pwa-modal-close {
            width: 100%;
            padding: 0.75rem;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
        }
        
        .pwa-install-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .pwa-install-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        @media (max-width: 600px) {
            .pwa-banner-content {
                flex-wrap: wrap;
            }
            
            .pwa-banner-text {
                flex-basis: calc(100% - 60px);
            }
            
            .pwa-banner-actions {
                flex-basis: 100%;
                justify-content: center;
            }
        }
    `;
    document.head.appendChild(style);
})();


// Auto-initialize on page load
let pwaInstallInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    pwaInstallInstance = new PWAInstallPrompt();
    pwaInstallInstance.init();
});


// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { PWAInstallPrompt };
}

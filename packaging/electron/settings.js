/**
 * Enigma AI - Settings Manager
 * 
 * Persistent settings storage for the Electron app.
 */

const { app } = require('electron');
const fs = require('fs');
const path = require('path');

const SETTINGS_FILE = path.join(app.getPath('userData'), 'settings.json');

const DEFAULT_SETTINGS = {
  // Window
  windowBounds: {
    width: 1400,
    height: 900,
    x: undefined,
    y: undefined,
  },
  windowMaximized: false,
  
  // UI
  theme: 'system', // 'light', 'dark', 'system'
  fontSize: 14,
  compactMode: false,
  showWelcome: true,
  
  // Behavior
  startMinimized: false,
  minimizeToTray: true,
  launchAtStartup: false,
  checkUpdatesOnStart: true,
  
  // Chat
  sendOnEnter: true,
  showTimestamps: true,
  enableMarkdown: true,
  enableCodeHighlight: true,
  
  // Voice
  enableVoice: true,
  voiceLanguage: 'en-US',
  voiceSpeed: 1.0,
  pushToTalk: false,
  hotkey: 'CommandOrControl+Shift+Space',
  
  // Backend
  backendPort: 8765,
  autoStartBackend: true,
  backendTimeout: 30000,
  
  // Model
  defaultModel: null,
  temperature: 0.7,
  maxTokens: 256,
  systemPrompt: '',
  
  // Privacy
  saveHistory: true,
  historyDays: 30,
  analytics: false,
  crashReports: true,
  
  // Advanced
  debugMode: false,
  gpuAcceleration: true,
  hardwareAcceleration: true,
};

class SettingsManager {
  constructor() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.loaded = false;
  }
  
  /**
   * Load settings from file
   */
  load() {
    try {
      if (fs.existsSync(SETTINGS_FILE)) {
        const data = fs.readFileSync(SETTINGS_FILE, 'utf8');
        const saved = JSON.parse(data);
        // Merge with defaults (handles new settings added in updates)
        this.settings = { ...DEFAULT_SETTINGS, ...saved };
      }
      this.loaded = true;
    } catch (error) {
      console.error('Failed to load settings:', error);
      this.settings = { ...DEFAULT_SETTINGS };
    }
    return this.settings;
  }
  
  /**
   * Save settings to file
   */
  save() {
    try {
      const dir = path.dirname(SETTINGS_FILE);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(SETTINGS_FILE, JSON.stringify(this.settings, null, 2));
    } catch (error) {
      console.error('Failed to save settings:', error);
    }
  }
  
  /**
   * Get a setting value
   */
  get(key) {
    if (!this.loaded) this.load();
    return key.split('.').reduce((obj, k) => obj?.[k], this.settings);
  }
  
  /**
   * Set a setting value
   */
  set(key, value) {
    if (!this.loaded) this.load();
    
    const keys = key.split('.');
    const last = keys.pop();
    const obj = keys.reduce((o, k) => o[k] = o[k] || {}, this.settings);
    obj[last] = value;
    
    this.save();
    return value;
  }
  
  /**
   * Get all settings
   */
  getAll() {
    if (!this.loaded) this.load();
    return { ...this.settings };
  }
  
  /**
   * Update multiple settings
   */
  update(updates) {
    if (!this.loaded) this.load();
    
    for (const [key, value] of Object.entries(updates)) {
      const keys = key.split('.');
      const last = keys.pop();
      const obj = keys.reduce((o, k) => o[k] = o[k] || {}, this.settings);
      obj[last] = value;
    }
    
    this.save();
    return this.settings;
  }
  
  /**
   * Reset settings to defaults
   */
  reset() {
    this.settings = { ...DEFAULT_SETTINGS };
    this.save();
    return this.settings;
  }
  
  /**
   * Export settings
   */
  export() {
    return JSON.stringify(this.settings, null, 2);
  }
  
  /**
   * Import settings
   */
  import(json) {
    try {
      const imported = JSON.parse(json);
      this.settings = { ...DEFAULT_SETTINGS, ...imported };
      this.save();
      return true;
    } catch (error) {
      console.error('Failed to import settings:', error);
      return false;
    }
  }
}

// Singleton instance
const settings = new SettingsManager();

module.exports = { settings, DEFAULT_SETTINGS, SETTINGS_FILE };

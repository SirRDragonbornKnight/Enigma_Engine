/**
 * Enigma AI Engine - Electron Preload Script
 * 
 * Exposes safe APIs to the renderer process via contextBridge.
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('enigmaAPI', {
  // Chat
  sendMessage: (message) => ipcRenderer.invoke('chat', message),
  
  // Models
  getModels: () => ipcRenderer.invoke('get-models'),
  loadModel: (name) => ipcRenderer.invoke('load-model', name),
  
  // Files
  openFile: (options) => ipcRenderer.invoke('open-file', options),
  saveFile: (options) => ipcRenderer.invoke('save-file', options),
  
  // App
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  toggleDarkMode: () => ipcRenderer.invoke('toggle-dark-mode'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  
  // Events from main process
  onVoiceCommand: (callback) => {
    ipcRenderer.on('voice-command', (event, command) => callback(command));
    return () => ipcRenderer.removeAllListeners('voice-command');
  },
  
  onNavigate: (callback) => {
    ipcRenderer.on('navigate', (event, page) => callback(page));
    return () => ipcRenderer.removeAllListeners('navigate');
  },
  
  onThemeChange: (callback) => {
    ipcRenderer.on('theme-change', (event, isDark) => callback(isDark));
    return () => ipcRenderer.removeAllListeners('theme-change');
  }
});

// Expose version info
contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron
});

console.log('Enigma AI preload script loaded');

/**
 * Enigma AI Engine - Electron Preload Script
 * 
 * Exposes safe APIs to the renderer process via contextBridge.
 * This script runs in a sandboxed context with access to Node.js.
 */

const { contextBridge, ipcRenderer, clipboard, shell } = require('electron');

// Expose protected methods to the renderer process
contextBridge.exposeInMainWorld('enigmaAPI', {
  // ==================== Chat ====================
  sendMessage: (message, options) => ipcRenderer.invoke('chat', message, options),
  streamMessage: (message, options) => ipcRenderer.invoke('chat-stream', message, options),
  cancelGeneration: () => ipcRenderer.invoke('cancel-generation'),
  
  // ==================== Models ====================
  getModels: () => ipcRenderer.invoke('get-models'),
  loadModel: (name) => ipcRenderer.invoke('load-model', name),
  unloadModel: () => ipcRenderer.invoke('unload-model'),
  getModelInfo: () => ipcRenderer.invoke('get-model-info'),
  downloadModel: (url, name) => ipcRenderer.invoke('download-model', url, name),
  
  // ==================== Image Generation ====================
  generateImage: (prompt, options) => ipcRenderer.invoke('generate-image', prompt, options),
  
  // ==================== Voice ====================
  startListening: () => ipcRenderer.invoke('voice-start'),
  stopListening: () => ipcRenderer.invoke('voice-stop'),
  speak: (text, options) => ipcRenderer.invoke('voice-speak', text, options),
  stopSpeaking: () => ipcRenderer.invoke('voice-stop-speaking'),
  
  // ==================== Files ====================
  openFile: (options) => ipcRenderer.invoke('open-file', options),
  saveFile: (options) => ipcRenderer.invoke('save-file', options),
  readFile: (path) => ipcRenderer.invoke('read-file', path),
  writeFile: (path, content) => ipcRenderer.invoke('write-file', path, content),
  
  // ==================== Settings ====================
  getSetting: (key) => ipcRenderer.invoke('settings-get', key),
  setSetting: (key, value) => ipcRenderer.invoke('settings-set', key, value),
  getAllSettings: () => ipcRenderer.invoke('settings-get-all'),
  resetSettings: () => ipcRenderer.invoke('settings-reset'),
  exportSettings: () => ipcRenderer.invoke('settings-export'),
  importSettings: (json) => ipcRenderer.invoke('settings-import', json),
  
  // ==================== History ====================
  getHistory: (limit) => ipcRenderer.invoke('history-get', limit),
  searchHistory: (query) => ipcRenderer.invoke('history-search', query),
  deleteHistory: (id) => ipcRenderer.invoke('history-delete', id),
  clearHistory: () => ipcRenderer.invoke('history-clear'),
  exportHistory: () => ipcRenderer.invoke('history-export'),
  
  // ==================== App ====================
  getAppInfo: () => ipcRenderer.invoke('get-app-info'),
  toggleDarkMode: () => ipcRenderer.invoke('toggle-dark-mode'),
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  checkForUpdates: () => ipcRenderer.invoke('check-updates'),
  quitApp: () => ipcRenderer.invoke('quit-app'),
  minimizeToTray: () => ipcRenderer.invoke('minimize-to-tray'),
  showWindow: () => ipcRenderer.invoke('show-window'),
  
  // ==================== Window ====================
  setWindowTitle: (title) => ipcRenderer.invoke('set-window-title', title),
  setWindowSize: (width, height) => ipcRenderer.invoke('set-window-size', width, height),
  toggleFullscreen: () => ipcRenderer.invoke('toggle-fullscreen'),
  isFullscreen: () => ipcRenderer.invoke('is-fullscreen'),
  
  // ==================== Clipboard ====================
  clipboard: {
    readText: () => clipboard.readText(),
    writeText: (text) => clipboard.writeText(text),
    readImage: () => clipboard.readImage()?.toDataURL() || null,
    writeImage: (dataUrl) => {
      const nativeImage = require('electron').nativeImage;
      const image = nativeImage.createFromDataURL(dataUrl);
      clipboard.writeImage(image);
    },
    clear: () => clipboard.clear(),
  },
  
  // ==================== Shell ====================
  shell: {
    openExternal: (url) => shell.openExternal(url),
    openPath: (path) => shell.openPath(path),
    showItemInFolder: (path) => shell.showItemInFolder(path),
  },
  
  // ==================== Notifications ====================
  showNotification: (title, body, options) => {
    return ipcRenderer.invoke('show-notification', title, body, options);
  },
  
  // ==================== Events from Main Process ====================
  onVoiceCommand: (callback) => {
    const handler = (event, command) => callback(command);
    ipcRenderer.on('voice-command', handler);
    return () => ipcRenderer.removeListener('voice-command', handler);
  },
  
  onNavigate: (callback) => {
    const handler = (event, page) => callback(page);
    ipcRenderer.on('navigate', handler);
    return () => ipcRenderer.removeListener('navigate', handler);
  },
  
  onThemeChange: (callback) => {
    const handler = (event, isDark) => callback(isDark);
    ipcRenderer.on('theme-change', handler);
    return () => ipcRenderer.removeListener('theme-change', handler);
  },
  
  onStreamChunk: (callback) => {
    const handler = (event, chunk) => callback(chunk);
    ipcRenderer.on('stream-chunk', handler);
    return () => ipcRenderer.removeListener('stream-chunk', handler);
  },
  
  onBackendStatus: (callback) => {
    const handler = (event, status) => callback(status);
    ipcRenderer.on('backend-status', handler);
    return () => ipcRenderer.removeListener('backend-status', handler);
  },
  
  onDownloadProgress: (callback) => {
    const handler = (event, progress) => callback(progress);
    ipcRenderer.on('download-progress', handler);
    return () => ipcRenderer.removeListener('download-progress', handler);
  },
  
  onUpdateAvailable: (callback) => {
    const handler = (event, info) => callback(info);
    ipcRenderer.on('update-available', handler);
    return () => ipcRenderer.removeListener('update-available', handler);
  },
  
  onError: (callback) => {
    const handler = (event, error) => callback(error);
    ipcRenderer.on('error', handler);
    return () => ipcRenderer.removeListener('error', handler);
  },
});

// Expose platform info
contextBridge.exposeInMainWorld('platform', {
  os: process.platform,
  arch: process.arch,
  isWindows: process.platform === 'win32',
  isMac: process.platform === 'darwin',
  isLinux: process.platform === 'linux',
});

// Expose version info
contextBridge.exposeInMainWorld('versions', {
  node: () => process.versions.node,
  chrome: () => process.versions.chrome,
  electron: () => process.versions.electron,
  app: () => ipcRenderer.invoke('get-app-version'),
});

console.log('Enigma AI preload script loaded - Platform:', process.platform);

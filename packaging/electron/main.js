/**
 * Enigma AI Engine - Electron Main Process
 * 
 * Desktop application wrapper for Enigma AI Engine.
 * Provides native window, system tray, and Python backend integration.
 */

const { app, BrowserWindow, ipcMain, Tray, Menu, dialog, shell, nativeTheme } = require('electron');
const { autoUpdater } = require('electron-updater');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Keep references to prevent garbage collection
let mainWindow = null;
let tray = null;
let pythonProcess = null;
let isQuitting = false;

// Configuration
const config = {
  pythonPath: getPythonPath(),
  backendPort: 8765,
  devMode: process.argv.includes('--dev'),
  debug: process.argv.includes('--debug')
};

/**
 * Get the appropriate Python executable path
 */
function getPythonPath() {
  // Check for bundled Python first
  const bundledPython = path.join(process.resourcesPath || __dirname, 'python', 'python.exe');
  if (fs.existsSync(bundledPython)) {
    return bundledPython;
  }
  
  // Fall back to system Python
  if (process.platform === 'win32') {
    return 'python';
  }
  return 'python3';
}

/**
 * Get the Enigma Engine path
 */
function getEnginePath() {
  // In production, use bundled resources
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'enigma_engine');
  }
  // In development, use parent directory
  return path.join(__dirname, '..', '..', 'enigma_engine');
}

/**
 * Start the Python backend server
 */
function startPythonBackend() {
  return new Promise((resolve, reject) => {
    const enginePath = getEnginePath();
    const runScript = path.join(enginePath, '..', 'run.py');
    
    console.log(`Starting Python backend: ${config.pythonPath} ${runScript}`);
    
    pythonProcess = spawn(config.pythonPath, [
      runScript,
      '--serve',
      '--port', config.backendPort.toString(),
      '--no-gui'
    ], {
      cwd: path.dirname(runScript),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        ENIGMA_ELECTRON: '1'
      }
    });
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`[Python] ${output}`);
      
      // Check if server is ready
      if (output.includes('Server running') || output.includes('Listening on')) {
        resolve(config.backendPort);
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`[Python Error] ${data.toString()}`);
    });
    
    pythonProcess.on('error', (error) => {
      console.error('Failed to start Python:', error);
      reject(error);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      if (!isQuitting) {
        // Restart if crashed
        setTimeout(() => startPythonBackend(), 1000);
      }
    });
    
    // Timeout for server startup
    setTimeout(() => {
      resolve(config.backendPort); // Assume ready after timeout
    }, 10000);
  });
}

/**
 * Stop the Python backend
 */
function stopPythonBackend() {
  if (pythonProcess) {
    pythonProcess.kill();
    pythonProcess = null;
  }
}

/**
 * Create the main application window
 */
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    icon: getIconPath(),
    title: 'Enigma AI',
    show: false, // Show when ready
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#1e1e1e' : '#ffffff',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
      spellcheck: true
    }
  });
  
  // Load the app
  if (config.devMode) {
    mainWindow.loadURL('http://localhost:3000');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  }
  
  // Show when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Check for updates
    if (!config.devMode) {
      checkForUpdates();
    }
  });
  
  // Handle close - minimize to tray instead
  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
      
      // Show tray notification on first hide
      if (tray && !app.isPackaged) {
        tray.displayBalloon({
          title: 'Enigma AI',
          content: 'Running in background. Click tray icon to restore.'
        });
      }
    }
  });
  
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  
  return mainWindow;
}

/**
 * Get the icon path based on platform
 */
function getIconPath() {
  const iconName = process.platform === 'win32' ? 'icon.ico' : 
                   process.platform === 'darwin' ? 'icon.icns' : 'icon.png';
  return path.join(__dirname, 'assets', iconName);
}

/**
 * Create the system tray icon
 */
function createTray() {
  const trayIcon = process.platform === 'darwin' ? 'tray-icon.png' : 'icon.png';
  tray = new Tray(path.join(__dirname, 'assets', trayIcon));
  
  const contextMenu = Menu.buildFromTemplate([
    {
      label: 'Show Window',
      click: () => {
        if (mainWindow) {
          mainWindow.show();
          mainWindow.focus();
        }
      }
    },
    {
      label: 'Quick Chat',
      click: () => {
        // Open mini chat overlay
        createQuickChatWindow();
      }
    },
    { type: 'separator' },
    {
      label: 'Start Listening',
      click: () => {
        mainWindow?.webContents.send('voice-command', 'start');
      }
    },
    {
      label: 'Stop Listening',
      click: () => {
        mainWindow?.webContents.send('voice-command', 'stop');
      }
    },
    { type: 'separator' },
    {
      label: 'Settings',
      click: () => {
        mainWindow?.show();
        mainWindow?.webContents.send('navigate', 'settings');
      }
    },
    {
      label: 'Check for Updates',
      click: checkForUpdates
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        isQuitting = true;
        app.quit();
      }
    }
  ]);
  
  tray.setToolTip('Enigma AI');
  tray.setContextMenu(contextMenu);
  
  // Double-click to show window
  tray.on('double-click', () => {
    if (mainWindow) {
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

/**
 * Create quick chat overlay window
 */
function createQuickChatWindow() {
  const quickChat = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });
  
  quickChat.loadFile(path.join(__dirname, 'renderer', 'quick-chat.html'));
}

/**
 * Check for application updates
 */
function checkForUpdates() {
  autoUpdater.checkForUpdatesAndNotify().catch(err => {
    console.log('Update check failed:', err);
  });
}

/**
 * Set up auto-updater events
 */
function setupAutoUpdater() {
  autoUpdater.on('update-available', (info) => {
    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Available',
      message: `Version ${info.version} is available. Download now?`,
      buttons: ['Yes', 'Later']
    }).then(({ response }) => {
      if (response === 0) {
        autoUpdater.downloadUpdate();
      }
    });
  });
  
  autoUpdater.on('update-downloaded', (info) => {
    dialog.showMessageBox(mainWindow, {
      type: 'info',
      title: 'Update Ready',
      message: `Version ${info.version} has been downloaded. Restart to install?`,
      buttons: ['Restart', 'Later']
    }).then(({ response }) => {
      if (response === 0) {
        isQuitting = true;
        autoUpdater.quitAndInstall();
      }
    });
  });
  
  autoUpdater.on('error', (error) => {
    console.error('Auto-updater error:', error);
  });
}

/**
 * Set up IPC handlers for renderer communication
 */
function setupIPC() {
  // Chat message
  ipcMain.handle('chat', async (event, message) => {
    try {
      const response = await fetch(`http://localhost:${config.backendPort}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  });
  
  // Get models
  ipcMain.handle('get-models', async () => {
    try {
      const response = await fetch(`http://localhost:${config.backendPort}/api/models`);
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  });
  
  // Load model
  ipcMain.handle('load-model', async (event, modelName) => {
    try {
      const response = await fetch(`http://localhost:${config.backendPort}/api/models/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName })
      });
      return await response.json();
    } catch (error) {
      return { error: error.message };
    }
  });
  
  // Open file dialog
  ipcMain.handle('open-file', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
  });
  
  // Save file dialog
  ipcMain.handle('save-file', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
  });
  
  // Get app info
  ipcMain.handle('get-app-info', () => {
    return {
      version: app.getVersion(),
      platform: process.platform,
      arch: process.arch,
      electronVersion: process.versions.electron,
      nodeVersion: process.versions.node
    };
  });
  
  // Toggle dark mode
  ipcMain.handle('toggle-dark-mode', () => {
    nativeTheme.themeSource = nativeTheme.shouldUseDarkColors ? 'light' : 'dark';
    return nativeTheme.shouldUseDarkColors;
  });
  
  // Restart backend
  ipcMain.handle('restart-backend', async () => {
    stopPythonBackend();
    await startPythonBackend();
    return { success: true };
  });
}

/**
 * Application ready
 */
app.whenReady().then(async () => {
  // Start Python backend
  try {
    await startPythonBackend();
  } catch (error) {
    console.error('Failed to start backend:', error);
    dialog.showErrorBox(
      'Backend Error',
      'Failed to start Python backend. Please ensure Python is installed.'
    );
  }
  
  // Create window and tray
  createMainWindow();
  createTray();
  
  // Set up IPC and auto-updater
  setupIPC();
  setupAutoUpdater();
  
  // macOS: recreate window when dock icon clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    } else if (mainWindow) {
      mainWindow.show();
    }
  });
});

/**
 * Quit when all windows closed (except macOS)
 */
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    isQuitting = true;
    app.quit();
  }
});

/**
 * Clean up before quit
 */
app.on('before-quit', () => {
  isQuitting = true;
  stopPythonBackend();
});

/**
 * Handle certificate errors (for local development)
 */
app.on('certificate-error', (event, webContents, url, error, certificate, callback) => {
  if (url.startsWith('https://localhost')) {
    event.preventDefault();
    callback(true);
  } else {
    callback(false);
  }
});

// Prevent multiple instances
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    // Focus existing window if another instance tries to start
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.show();
      mainWindow.focus();
    }
  });
}

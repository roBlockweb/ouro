/**
 * Ouro v2.5 Preload Script
 * Securely exposes specific APIs from the main process to the renderer process
 */

const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use IPC
contextBridge.exposeInMainWorld('api', {
  // System information
  getSystemInfo: () => ipcRenderer.invoke('get-system-info'),
  
  // Directory selection
  selectInstallDirectory: () => ipcRenderer.invoke('select-install-directory'),
  
  // Model recommendations
  getModelRecommendations: (systemInfo) => ipcRenderer.invoke('get-model-recommendations', systemInfo),
  
  // Installation process
  startInstallation: (options) => ipcRenderer.invoke('start-installation', options),
  
  // File progress event
  onFileCopyProgress: (callback) => {
    const listener = (event, data) => callback(data);
    ipcRenderer.on('file-copy-progress', listener);
    return () => {
      ipcRenderer.removeListener('file-copy-progress', listener);
    };
  },
  
  // Installation progress event
  onInstallationProgress: (callback) => {
    const listener = (event, data) => callback(data);
    ipcRenderer.on('installation-progress', listener);
    return () => {
      ipcRenderer.removeListener('installation-progress', listener);
    };
  },
  
  // Launch Ouro after installation
  launchOuro: (installPath) => ipcRenderer.invoke('launch-ouro', installPath),
  
  // Open external URL
  openExternalUrl: (url) => ipcRenderer.invoke('open-external-url', url),
  
  // Check dependencies
  checkDependencies: () => ipcRenderer.invoke('check-dependencies'),
  
  // Install dependency
  installDependency: (dependency) => ipcRenderer.invoke('install-dependency', dependency),
  
  // Logging functions
  logInfo: (message) => ipcRenderer.send('log-info', message),
  logError: (message, error) => ipcRenderer.send('log-error', message, error),
});
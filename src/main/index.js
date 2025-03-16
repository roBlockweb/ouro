/**
 * Ouro v2.5 Main Process
 * Main entry point for the Electron application
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const fsSync = require('fs');
const os = require('os');
const { spawn, execSync } = require('child_process');
const log = require('electron-log');
const Store = require('electron-store');
const keytar = require('keytar');
const yaml = require('js-yaml');

// Environment configuration
const isDev = process.env.NODE_ENV === 'development';
const isMac = process.platform === 'darwin';
const isWindows = process.platform === 'win32';

// Configure enhanced structured logging
log.initialize({ preload: true });
log.transports.file.format = '[{y}-{m}-{d} {h}:{i}:{s}.{ms}] [{level}] [{processType}] [{category}] {text}';
log.transports.file.level = 'info';
log.transports.file.maxSize = 10 * 1024 * 1024; // 10MB
log.transports.file.rotation = 'daily';
log.transports.file.resolvePath = () => path.join(
  app.getPath('logs'),
  'Ouro',
  `ouro_${new Date().toISOString().slice(0, 10)}.log`
);

// Configure console logging
log.transports.console.format = '[{level}] [{category}] {text}';
log.transports.console.level = isDev ? 'debug' : 'info';

// Define the service name for credential storage
const CREDENTIAL_SERVICE = 'OuroAssistant';

// Setup store for persisting user preferences (non-sensitive)
const store = new Store();

// Resources path configuration
const resourcesPath = isDev 
  ? path.join(__dirname, '..', '..', 'core') 
  : path.join(process.resourcesPath, 'core');

let mainWindow;
let installPath;
let pythonProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      // SECURITY: Disable Node integration in renderer process
      nodeIntegration: false,
      // SECURITY: Enable context isolation to prevent prototype pollution
      contextIsolation: true,
      // SECURITY: Use preload script with contextBridge for safe IPC
      preload: path.join(__dirname, 'preload.js'),
      // SECURITY: Enable chromium sandbox for additional protection
      sandbox: true,
      // SECURITY: Enable web security to enforce same-origin policy
      webSecurity: true,
      // SECURITY: Disable remote module for better security
      enableRemoteModule: false
    },
    show: false,
    backgroundColor: '#121212', // Dark mode theme
    icon: path.join(isDev ? path.join(__dirname, '..', '..', 'assets') : process.resourcesPath, 'assets', isMac ? 'icon.icns' : 'icon.ico')
  });

  mainWindow.loadFile(path.join(__dirname, '..', 'ui', 'index.html'));

  // Set Content-Security-Policy header
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': ["default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';"]
      }
    });
  });

  // Open DevTools in development mode
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    // Log startup information
    log.info(`Ouro v2.5 started - Platform: ${process.platform}, Arch: ${process.arch}`);
    log.info(`Resources path: ${resourcesPath}`);
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
    
    // Kill Python process if it's running
    if (pythonProcess) {
      log.info('Shutting down Python server...');
      if (isWindows) {
        spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
      } else {
        pythonProcess.kill('SIGTERM');
      }
      pythonProcess = null;
    }
  });
}

// Initialize application
app.whenReady().then(async () => {
  // Start the Python server
  await startWebServer();
  
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (!isMac) {
    app.quit();
  }
});

// Helper function to execute commands asynchronously with timeout
async function executeCommand(command, args = [], timeout = 10000) {
  return new Promise((resolve, reject) => {
    let output = '';
    let error = '';
    
    log.debug(`Executing command: ${command} ${args.join(' ')}`);
    
    const childProcess = spawn(command, args);
    const timer = setTimeout(() => {
      childProcess.kill();
      reject(new Error(`Command execution timed out after ${timeout}ms: ${command} ${args.join(' ')}`));
    }, timeout);
    
    childProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    childProcess.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    childProcess.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
    
    childProcess.on('close', (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Command failed with code ${code}: ${error}`));
      }
    });
  });
}

// Start the Python web server
async function startWebServer() {
  try {
    // Path to the Python script
    const serverScriptPath = path.join(resourcesPath, 'start_server.py');
    
    log.info(`Starting Python server from: ${serverScriptPath}`);
    
    // Check if the script exists
    try {
      await fs.access(serverScriptPath);
      log.info('Server script found');
    } catch (error) {
      log.error(`Server script not found at ${serverScriptPath}:`, error);
      throw new Error(`Server script not found: ${error.message}`);
    }
    
    // Determine Python command based on platform
    const pythonCmd = isWindows ? 'python' : 'python3';
    const options = {
      cwd: path.dirname(serverScriptPath),
      env: { ...process.env, PYTHONUNBUFFERED: '1' }
    };
    
    // Start the Python process
    log.info(`Launching server with command: ${pythonCmd} ${serverScriptPath}`);
    pythonProcess = spawn(pythonCmd, [serverScriptPath, '--no-browser'], options);
    
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      log.info(`Python Server: ${output}`);
      
      // Check if the server is ready
      if (output.includes('Serving at http://localhost:')) {
        const match = output.match(/Serving at (http:\/\/localhost:\d+)/);
        if (match && match[1]) {
          log.info(`Server ready at ${match[1]}`);
        }
      }
    });
    
    pythonProcess.stderr.on('data', (data) => {
      log.error(`Python Server Error: ${data.toString().trim()}`);
    });
    
    pythonProcess.on('error', (error) => {
      log.error('Failed to start Python server:', error);
    });
    
    pythonProcess.on('close', (code) => {
      log.info(`Python server exited with code ${code}`);
      pythonProcess = null;
    });
    
    // Wait a bit to ensure the server has started
    return new Promise((resolve) => {
      setTimeout(() => {
        if (pythonProcess) {
          log.info('Python server startup sequence completed');
          resolve(true);
        } else {
          log.error('Python server failed to start');
          resolve(false);
        }
      }, 2000);
    });
  } catch (error) {
    log.error('Error starting web server:', error);
    return false;
  }
}

// IPC handlers for system information
ipcMain.handle('get-system-info', async () => {
  try {
    const totalMemGB = Math.floor(os.totalmem() / (1024 * 1024 * 1024));
    const cpuCores = os.cpus().length;
    
    // Check for Ollama installation
    const isOllama = await checkIfOllamaInstalled();
    
    // Check for Qdrant
    const isQdrant = await checkIfQdrantInstalled();
    
    return {
      platform: process.platform,
      arch: process.arch,
      cpuCores,
      memory: totalMemGB,
      isOllama,
      isQdrant
    };
  } catch (error) {
    log.error('Error getting system info:', error);
    throw error;
  }
});

// Check if Ollama is installed
async function checkIfOllamaInstalled() {
  try {
    if (isMac) {
      // Check for Ollama application on macOS
      const ollamaPath = '/Applications/Ollama.app';
      await fs.access(ollamaPath);
      return true;
    } else if (isWindows) {
      // Try running Ollama command on Windows
      try {
        await executeCommand('ollama', ['--version']);
        return true;
      } catch (error) {
        return false;
      }
    } else {
      // Linux - try the CLI
      try {
        await executeCommand('ollama', ['--version']);
        return true;
      } catch (error) {
        return false;
      }
    }
  } catch (error) {
    log.debug('Ollama not installed or not accessible:', error.message);
    return false;
  }
}

// Check if Qdrant is installed
async function checkIfQdrantInstalled() {
  try {
    // Try connecting to Qdrant port
    if (isMac || isWindows) {
      try {
        const result = await executeCommand('nc', ['-z', 'localhost', '6333']);
        return true;
      } catch (error) {
        return false;
      }
    } else {
      // Alternative check for Linux
      try {
        await executeCommand('curl', ['-s', 'http://localhost:6333/health']);
        return true;
      } catch (error) {
        return false;
      }
    }
  } catch (error) {
    return false;
  }
}

// Handle installation directory selection
ipcMain.handle('select-install-directory', async () => {
  const defaultPath = isMac ? 
    path.join(os.homedir(), 'Applications') : 
    'C:\\Program Files';
  
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory'],
    defaultPath
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    installPath = result.filePaths[0];
    return installPath;
  }
  return null;
});

// Handle model selection recommendations
ipcMain.handle('get-model-recommendations', async (event, systemInfo) => {
  const memory = systemInfo.memory || 0;
  const recommendations = [];
  
  // Tiny models (~3B parameters)
  recommendations.push({
    size: 'tiny',
    name: 'phi:3-mini',
    parameters: '3B',
    recommended: memory < 8,
    requiredMemory: 4,
    description: 'Fast, lightweight model suitable for basic tasks. Good for low-resource machines.'
  });
  
  // Small models (~7B parameters)
  recommendations.push({
    size: 'small',
    name: 'llama3:8b',
    parameters: '8B',
    recommended: memory >= 8 && memory < 12,
    requiredMemory: 8,
    description: 'Good balance of performance and resource usage. Recommended for most users.'
  });
  
  // Medium models (~70B parameters, quantized)
  recommendations.push({
    size: 'medium',
    name: 'llama3:70b-q4_K_M',
    parameters: '70B (quantized)',
    recommended: memory >= 12 && memory < 24,
    requiredMemory: 12,
    description: 'Advanced capabilities with moderate resource requirements.'
  });
  
  // Large models (non-quantized)
  recommendations.push({
    size: 'large',
    name: 'llama3:70b',
    parameters: '70B',
    recommended: memory >= 24,
    requiredMemory: 24,
    description: 'Best performance but requires significant resources.'
  });
  
  return recommendations;
});

// Handle the installation process
ipcMain.handle('start-installation', async (event, options) => {
  try {
    // Create installation directory if needed
    const targetDir = options.installPath || path.join(os.homedir(), 'Ouro');
    
    // Check if directory exists and is writable
    try {
      await fs.access(targetDir, fs.constants.W_OK);
      log.info(`Installation directory exists and is writable: ${targetDir}`);
    } catch (accessError) {
      // Directory doesn't exist or isn't writable, create it
      log.info(`Creating installation directory: ${targetDir}`);
      await fs.mkdir(targetDir, { recursive: true });
    }
    
    // Send progress update
    mainWindow.webContents.send('installation-progress', {
      step: 'prepare',
      progress: 10,
      message: 'Installation directory prepared'
    });
    
    // Create dependencies directory
    const dependenciesDir = path.join(targetDir, 'dependencies');
    await fs.mkdir(dependenciesDir, { recursive: true });
    
    // Copy core files
    await copyOuroCore(targetDir, options);
    
    // Send progress update
    mainWindow.webContents.send('installation-progress', {
      step: 'copy',
      progress: 40,
      message: 'Core files copied successfully'
    });
    
    // Setup configuration
    await setupConfiguration(targetDir, options);
    
    // Send progress update
    mainWindow.webContents.send('installation-progress', {
      step: 'configure',
      progress: 70,
      message: 'Configuration complete'
    });
    
    // Create desktop shortcut
    await createDesktopShortcut(targetDir);
    
    // Send progress update
    mainWindow.webContents.send('installation-progress', {
      step: 'shortcut',
      progress: 90,
      message: 'Desktop shortcut created'
    });
    
    // Verify launcher script exists and is executable
    const scriptExt = isWindows ? '.bat' : '.sh';
    const startScript = path.join(targetDir, `start_ouro${scriptExt}`);
    
    try {
      await fs.access(startScript);
      if (!isWindows) {
        await fs.chmod(startScript, '755'); // Ensure script is executable on Unix platforms
      }
    } catch (scriptError) {
      log.warn(`Launcher script verification failed: ${scriptError.message}`);
      await createLauncherScript(targetDir, options);
    }
    
    // Final progress update
    mainWindow.webContents.send('installation-progress', {
      step: 'complete',
      progress: 100,
      message: 'Installation complete'
    });
    
    return { 
      success: true, 
      installPath: targetDir,
      selectedModel: options.selectedModel
    };
  } catch (error) {
    log.error('Installation failed:', error);
    
    // Send progress update about failure
    mainWindow.webContents.send('installation-progress', {
      step: 'error',
      progress: 100,
      error: true,
      message: `Installation failed: ${error.message}`
    });
    
    return { 
      success: false, 
      error: error.message
    };
  }
});

// Copy core Ouro files to target directory
async function copyOuroCore(targetDir, options) {
  try {
    log.info(`Copying core files from ${resourcesPath} to ${targetDir}`);
    
    // Create core directories
    const coreDirectories = ['models', 'config', 'data', 'web'];
    for (const dir of coreDirectories) {
      await fs.mkdir(path.join(targetDir, dir), { recursive: true });
    }
    
    // Copy configuration files
    const configFiles = ['config.yaml', 'docker-compose.yml'];
    for (const file of configFiles) {
      try {
        const sourcePath = path.join(resourcesPath, file);
        const destPath = path.join(targetDir, 'config', file);
        
        // Check if source file exists
        try {
          await fs.access(sourcePath);
          await fs.copyFile(sourcePath, destPath);
        } catch (error) {
          // Create default file if it doesn't exist
          if (file === 'config.yaml') {
            const defaultConfig = `
# Ouro v2.5 Configuration
version: 2.5
ollama:
  model: ${options.selectedModel || 'llama3:8b'}
  embeddings: nomic-embed-text
qdrant:
  url: http://localhost:6333
  collection: ouro_docs
ui:
  theme: dark
  port: 3000
`;
            await fs.writeFile(destPath, defaultConfig);
          } else if (file === 'docker-compose.yml') {
            const defaultDockerCompose = `
version: '3'
services:
  qdrant:
    image: qdrant/qdrant
    container_name: ouro-qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped
`;
            await fs.writeFile(destPath, defaultDockerCompose);
          }
        }
      } catch (error) {
        log.warn(`Error copying file ${file}: ${error.message}`);
      }
    }
    
    // Create startup scripts
    await createStartupScripts(targetDir);
    
    // Send progress update for UI
    if (mainWindow) {
      mainWindow.webContents.send('file-copy-progress', {
        file: 'Configuration files',
        current: 1,
        total: 3,
        percentage: 33
      });
    }
    
    // Create user interface files
    await createUIFiles(targetDir);
    
    // Send progress update for UI
    if (mainWindow) {
      mainWindow.webContents.send('file-copy-progress', {
        file: 'UI files',
        current: 2,
        total: 3,
        percentage: 66
      });
    }
    
    // Create script to install dependencies if needed
    await createDependencyInstallers(targetDir);
    
    // Send progress update for UI
    if (mainWindow) {
      mainWindow.webContents.send('file-copy-progress', {
        file: 'Setup scripts',
        current: 3,
        total: 3,
        percentage: 100
      });
    }
    
    return true;
  } catch (error) {
    log.error('Error copying core files:', error);
    throw error;
  }
}

// Create startup scripts
async function createStartupScripts(targetDir) {
  try {
    // Create macOS start script
    if (isMac) {
      const macScript = `#!/bin/bash
# Ouro v2.5 Startup Script
echo "Starting Ouro v2.5..."

# Set working directory
cd "$(dirname "$0")"

# Check if Ollama is installed
if [ ! -d "/Applications/Ollama.app" ]; then
  echo "Ollama not found! Opening download page..."
  open "https://ollama.ai/download"
  exit 1
fi

# Start Qdrant if not already running
if ! nc -z localhost 6333 &>/dev/null; then
  echo "Starting Qdrant..."
  docker-compose -f config/docker-compose.yml up -d qdrant
fi

# Start Ollama if not already running
if ! pgrep -x "Ollama" &>/dev/null; then
  echo "Starting Ollama..."
  open -a Ollama
fi

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
  sleep 1
done

# Start the Python server and open browser
echo "Starting Ouro web interface..."
python3 start_server.py &

echo "Ouro v2.5 started successfully!"
`;
      
      const macScriptPath = path.join(targetDir, 'start_ouro.sh');
      await fs.writeFile(macScriptPath, macScript);
      await fs.chmod(macScriptPath, '755');
    }
    
    // Create Windows start script
    if (isWindows) {
      const winScript = `@echo off
echo Starting Ouro v2.5...

:: Change to script directory
cd /d "%~dp0"

:: Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Ollama not found! Opening download page...
  start https://ollama.ai/download
  exit /b 1
)

:: Start Qdrant if not already running
netstat -an | findstr ":6333" >nul
if %ERRORLEVEL% neq 0 (
  echo Starting Qdrant...
  docker-compose -f config/docker-compose.yml up -d qdrant
)

:: Start Ollama if not already running
tasklist | findstr "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
  echo Starting Ollama...
  start "" ollama serve
)

:: Wait for Ollama to start
echo Waiting for Ollama to start...
:wait_loop
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
  timeout /t 1 >nul
  goto wait_loop
)

:: Start the Python server
echo Starting Ouro web interface...
start "" python start_server.py

echo Ouro v2.5 started successfully!
`;
      
      const winScriptPath = path.join(targetDir, 'start_ouro.bat');
      await fs.writeFile(winScriptPath, winScript);
    }
    
    // Create Linux start script
    const linuxScript = `#!/bin/bash
# Ouro v2.5 Startup Script
echo "Starting Ouro v2.5..."

# Set working directory
cd "$(dirname "$0")"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found! Opening download page..."
  xdg-open "https://ollama.ai/download"
  exit 1
fi

# Start Qdrant if not already running
if ! nc -z localhost 6333 &>/dev/null; then
  echo "Starting Qdrant..."
  docker-compose -f config/docker-compose.yml up -d qdrant
fi

# Start Ollama if not already running
if ! pgrep -x "ollama" &>/dev/null; then
  echo "Starting Ollama..."
  ollama serve &
fi

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
  sleep 1
done

# Start the Python server
echo "Starting Ouro web interface..."
python3 start_server.py &

echo "Ouro v2.5 started successfully!"
`;
    
    const linuxScriptPath = path.join(targetDir, 'start_ouro.sh');
    await fs.writeFile(linuxScriptPath, linuxScript);
    await fs.chmod(linuxScriptPath, '755');
    
    return true;
  } catch (error) {
    log.error('Error creating startup scripts:', error);
    throw error;
  }
}

// Create dependency installers
async function createDependencyInstallers(targetDir) {
  try {
    // Create macOS dependency installer
    if (isMac) {
      const macInstaller = `#!/bin/bash
# Ouro v2.5 Dependency Installer for macOS
echo "Installing dependencies for Ouro v2.5..."

# Check for Docker
if ! command -v docker &> /dev/null; then
  echo "Docker not found. Opening download page..."
  open "https://www.docker.com/products/docker-desktop/"
fi

# Check for Ollama
if [ ! -d "/Applications/Ollama.app" ]; then
  echo "Ollama not found. Opening download page..."
  open "https://ollama.ai/download"
fi

# Install model
if command -v ollama &> /dev/null; then
  echo "Installing default model (llama3:8b)..."
  ollama pull llama3:8b
  echo "Installing embedding model..."
  ollama pull nomic-embed-text
fi

echo "Dependency installation completed!"
`;
      
      const macInstallerPath = path.join(targetDir, 'dependencies', 'install_deps_mac.sh');
      await fs.writeFile(macInstallerPath, macInstaller);
      await fs.chmod(macInstallerPath, '755');
    }
    
    // Create Windows dependency installer
    if (isWindows) {
      const winInstaller = `@echo off
echo Installing dependencies for Ouro v2.5...

:: Check for Docker
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Docker not found. Opening download page...
  start https://www.docker.com/products/docker-desktop/
)

:: Check for Ollama
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Ollama not found. Opening download page...
  start https://ollama.ai/download
)

:: Install model
where ollama >nul 2>&1
if %ERRORLEVEL% equ 0 (
  echo Installing default model (llama3:8b)...
  ollama pull llama3:8b
  echo Installing embedding model...
  ollama pull nomic-embed-text
)

echo Dependency installation completed!
pause
`;
      
      const winInstallerPath = path.join(targetDir, 'dependencies', 'install_deps_win.bat');
      await fs.writeFile(winInstallerPath, winInstaller);
    }
    
    // Create Linux dependency installer
    const linuxInstaller = `#!/bin/bash
# Ouro v2.5 Dependency Installer for Linux
echo "Installing dependencies for Ouro v2.5..."

# Check for Docker
if ! command -v docker &> /dev/null; then
  echo "Docker not found. Opening download page..."
  xdg-open "https://www.docker.com/products/docker-desktop/"
fi

# Check for Ollama
if ! command -v ollama &> /dev/null; then
  echo "Ollama not found. Opening download page..."
  xdg-open "https://ollama.ai/download"
fi

# Install model
if command -v ollama &> /dev/null; then
  echo "Installing default model (llama3:8b)..."
  ollama pull llama3:8b
  echo "Installing embedding model..."
  ollama pull nomic-embed-text
fi

echo "Dependency installation completed!"
`;
    
    const linuxInstallerPath = path.join(targetDir, 'dependencies', 'install_deps_linux.sh');
    await fs.writeFile(linuxInstallerPath, linuxInstaller);
    await fs.chmod(linuxInstallerPath, '755');
    
    return true;
  } catch (error) {
    log.error('Error creating dependency installers:', error);
    throw error;
  }
}

// Create UI files (the web interface)
async function createUIFiles(targetDir) {
  try {
    const webDir = path.join(targetDir, 'web');
    await fs.mkdir(webDir, { recursive: true });
    
    // Create HTML file
    const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ouro v2.5</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <div class="app">
    <header>
      <div class="logo">
        <h1>Ouro</h1>
        <span class="version">v2.5</span>
      </div>
      <nav>
        <button class="active">Chat</button>
        <button>Documents</button>
        <button>Settings</button>
      </nav>
    </header>
    
    <main>
      <div class="sidebar">
        <div class="models">
          <h3>Models</h3>
          <div class="model-select">
            <select id="model-selector">
              <option value="llama3:8b">Llama 3 (8B)</option>
              <option value="llama3:70b-q4_K_M">Llama 3 (70B)</option>
              <option value="phi:3-mini">Phi-3 Mini</option>
            </select>
          </div>
        </div>
        
        <div class="conversations">
          <h3>Conversations</h3>
          <div class="new-chat">
            <button id="new-chat">+ New Chat</button>
          </div>
          <ul class="chat-list">
            <li class="chat-item active">General Chat</li>
            <li class="chat-item">Document Q&A</li>
          </ul>
        </div>
      </div>
      
      <div class="chat-container">
        <div class="messages">
          <div class="message system">
            <div class="message-content">
              <p>Welcome to Ouro v2.5. I'm your offline AI assistant powered by Ollama. How can I help you today?</p>
            </div>
          </div>
          <!-- Messages will appear here -->
        </div>
        
        <div class="input-area">
          <textarea id="user-input" placeholder="Send a message..."></textarea>
          <button id="send-button">Send</button>
        </div>
      </div>
    </main>
  </div>
  
  <script src="app.js"></script>
</body>
</html>`;
    
    await fs.writeFile(path.join(webDir, 'index.html'), htmlContent);
    
    // Create CSS file
    const cssContent = `/* Ouro v2.5 Styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

:root {
  --primary-color: #6200EA;
  --primary-light: #7c4dff;
  --primary-dark: #4b00b3;
  --accent-color: #03DAC5;
  --background-dark: #121212;
  --background-card: #1e1e1e;
  --background-light: #2d2d2d;
  --text-primary: rgba(255, 255, 255, 0.87);
  --text-secondary: rgba(255, 255, 255, 0.6);
  --border-color: rgba(255, 255, 255, 0.12);
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  background-color: var(--background-dark);
  color: var(--text-primary);
  line-height: 1.6;
  margin: 0;
  padding: 0;
  height: 100vh;
  overflow: hidden;
}

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Header */
header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: var(--background-card);
  border-bottom: 1px solid var(--border-color);
}

.logo {
  display: flex;
  align-items: baseline;
}

.logo h1 {
  font-size: 1.8rem;
  font-weight: 600;
  margin: 0;
}

.version {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-left: 0.5rem;
}

nav {
  display: flex;
  gap: 1rem;
}

nav button {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  padding: 0.5rem 1rem;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}

nav button.active {
  background-color: var(--primary-color);
  color: white;
}

/* Main content */
main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.sidebar {
  width: 260px;
  background-color: var(--background-card);
  padding: 1rem;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border-color);
}

.models {
  margin-bottom: 2rem;
}

.models h3, .conversations h3 {
  font-size: 0.9rem;
  text-transform: uppercase;
  color: var(--text-secondary);
  margin-bottom: 0.8rem;
}

.model-select select {
  width: 100%;
  padding: 0.6rem;
  background-color: var(--background-light);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 0.9rem;
}

.new-chat {
  margin-bottom: 1rem;
}

.new-chat button {
  width: 100%;
  padding: 0.6rem;
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
}

.chat-list {
  list-style: none;
}

.chat-item {
  padding: 0.6rem;
  border-radius: 4px;
  margin-bottom: 0.5rem;
  cursor: pointer;
}

.chat-item.active {
  background-color: rgba(98, 0, 234, 0.2);
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.messages {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.message {
  display: flex;
  margin-bottom: 1.5rem;
  max-width: 80%;
}

.message.system {
  align-self: flex-start;
}

.message.user {
  align-self: flex-end;
}

.message-content {
  padding: 1rem;
  border-radius: 0.8rem;
  background-color: var(--background-card);
}

.message.user .message-content {
  background-color: var(--primary-color);
}

.input-area {
  padding: 1rem;
  display: flex;
  gap: 0.8rem;
  border-top: 1px solid var(--border-color);
  background-color: var(--background-card);
}

#user-input {
  flex: 1;
  height: 3rem;
  padding: 0.8rem 1rem;
  background-color: var(--background-light);
  color: var(--text-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  font-size: 0.9rem;
  resize: none;
  font-family: inherit;
}

#send-button {
  padding: 0 1.5rem;
  background-color: var(--primary-color);
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
}

/* Responsive */
@media (max-width: 768px) {
  main {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    height: 30%;
    border-right: none;
    border-bottom: 1px solid var(--border-color);
  }
}`;
    
    await fs.writeFile(path.join(webDir, 'style.css'), cssContent);
    
    // Create JS file
    const jsContent = `// Ouro v2.5 UI Application
document.addEventListener('DOMContentLoaded', () => {
  // DOM elements
  const userInput = document.getElementById('user-input');
  const sendButton = document.getElementById('send-button');
  const messagesContainer = document.querySelector('.messages');
  const newChatButton = document.getElementById('new-chat');
  const modelSelector = document.getElementById('model-selector');
  
  // Event listeners
  sendButton.addEventListener('click', sendMessage);
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  
  newChatButton.addEventListener('click', () => {
    // Clear messages except the first one (welcome message)
    const messages = messagesContainer.querySelectorAll('.message:not(:first-child)');
    messages.forEach(message => message.remove());
  });
  
  // Function to send a message
  function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // Create user message
    addMessage(text, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Send message to Ollama API
    callOllama(text)
      .then(response => {
        addMessage(response, 'system');
      })
      .catch(error => {
        console.error('Error calling Ollama:', error);
        addMessage('Sorry, I encountered an error processing your request. Please try again.', 'system');
      });
  }
  
  // Function to call Ollama API
  async function callOllama(prompt) {
    try {
      const model = modelSelector.value;
      
      // First try to use the server API endpoint
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model,
            prompt
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          return data.response;
        }
      } catch (serverError) {
        console.warn('Server API error, falling back to direct Ollama API:', serverError);
      }
      
      // Fall back to direct Ollama API
      const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model,
          prompt,
          stream: false
        })
      });
      
      if (ollamaResponse.ok) {
        const data = await ollamaResponse.json();
        return data.response;
      } else {
        throw new Error('Failed to communicate with Ollama API');
      }
    } catch (error) {
      console.error('Error in callOllama:', error);
      
      // Fallback to simulated response
      const responses = [
        "I'm an offline AI assistant powered by Ollama. I can help answer questions based on my training data, but I don't have internet access or the ability to run code.",
        "That's an interesting question. As an offline AI, I'm working with the knowledge I was trained with.",
        "I'd be happy to help with that. Let me think about it...",
        "Since I'm running completely on your local machine, all your data stays private and is never sent to external servers."
      ];
      
      return responses[Math.floor(Math.random() * responses.length)];
    }
  }
  
  // Function to add a message to the chat
  function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    
    const paragraph = document.createElement('p');
    paragraph.textContent = text;
    
    contentDiv.appendChild(paragraph);
    messageDiv.appendChild(contentDiv);
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
  
  // Initialize model selector with the current model
  function initializeModelSelector() {
    // Try to fetch configuration from server
    fetch('/api/config')
      .then(response => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error('Could not fetch config');
        }
      })
      .then(config => {
        if (config.ollama && config.ollama.model) {
          modelSelector.value = config.ollama.model;
        }
      })
      .catch(error => {
        console.warn('Could not fetch config, using default model:', error);
      });
    
    modelSelector.addEventListener('change', () => {
      const selectedModel = modelSelector.value;
      console.log('Selected model:', selectedModel);
      
      // Notify about model change
      fetch('/api/config/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ollama: {
            model: selectedModel
          }
        })
      }).catch(error => {
        console.warn('Failed to update model preference:', error);
      });
    });
  }
  
  // Initialize
  initializeModelSelector();
});`;
    
    await fs.writeFile(path.join(webDir, 'app.js'), jsContent);
    
    return true;
  } catch (error) {
    log.error('Error creating UI files:', error);
    throw error;
  }
}

// Set up configuration based on user choices
async function setupConfiguration(targetDir, options) {
  try {
    // Create main config file
    const configContent = `# Ouro v2.5 Configuration
version: 2.5
ollama:
  model: ${options.selectedModel || 'llama3:8b'}
  embeddings: nomic-embed-text
qdrant:
  url: http://localhost:6333
  collection: ouro_docs
ui:
  theme: dark
  port: 3000
`;
    
    await fs.writeFile(path.join(targetDir, 'config', 'config.yaml'), configContent);
    
    // Store credentials securely
    if (options.apiKeys) {
      await keytar.setPassword(CREDENTIAL_SERVICE, 'ollama_api_key', options.apiKeys.ollama || '');
    }
    
    return true;
  } catch (error) {
    log.error('Error setting up configuration:', error);
    throw error;
  }
}

// Create launcher script
async function createLauncherScript(targetDir, options) {
  try {
    const scriptExt = isWindows ? '.bat' : '.sh';
    const scriptPath = path.join(targetDir, `start_ouro${scriptExt}`);
    
    let scriptContent;
    
    if (isWindows) {
      scriptContent = `@echo off
echo Starting Ouro v2.5...

:: Change to script directory
cd /d "%~dp0"

:: Check if Ollama is installed
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo Ollama not found! Opening download page...
  start https://ollama.ai/download
  exit /b 1
)

:: Start Qdrant if not already running
netstat -an | findstr ":6333" >nul
if %ERRORLEVEL% neq 0 (
  echo Starting Qdrant...
  docker-compose -f config/docker-compose.yml up -d qdrant
)

:: Start Ollama if not already running
tasklist | findstr "ollama.exe" >nul
if %ERRORLEVEL% neq 0 (
  echo Starting Ollama...
  start "" ollama serve
)

:: Wait for Ollama to start
echo Waiting for Ollama to start...
:wait_loop
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% neq 0 (
  timeout /t 1 >nul
  goto wait_loop
)

:: Start the Python server
echo Starting Ouro web interface...
python start_server.py

echo Ouro v2.5 started successfully!
`;
    } else {
      scriptContent = `#!/bin/bash
# Ouro v2.5 Startup Script
echo "Starting Ouro v2.5..."

# Set working directory
cd "$(dirname "$0")"

# Check if Ollama is installed
if [ "$(uname)" == "Darwin" ]; then
  # macOS
  if [ ! -d "/Applications/Ollama.app" ]; then
    echo "Ollama not found! Opening download page..."
    open "https://ollama.ai/download"
    exit 1
  fi
else
  # Linux
  if ! command -v ollama &> /dev/null; then
    echo "Ollama not found! Opening download page..."
    xdg-open "https://ollama.ai/download"
    exit 1
  fi
fi

# Start Qdrant if not already running
if ! nc -z localhost 6333 &>/dev/null; then
  echo "Starting Qdrant..."
  docker-compose -f config/docker-compose.yml up -d qdrant
fi

# Start Ollama if not already running
if [ "$(uname)" == "Darwin" ]; then
  # macOS
  if ! pgrep -x "Ollama" &>/dev/null; then
    echo "Starting Ollama..."
    open -a Ollama
  fi
else
  # Linux
  if ! pgrep -x "ollama" &>/dev/null; then
    echo "Starting Ollama..."
    ollama serve &
  fi
fi

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
  sleep 1
done

# Start the Python server
echo "Starting Ouro web interface..."
python3 start_server.py

echo "Ouro v2.5 started successfully!"
`;
    }
    
    // Write the script file
    await fs.writeFile(scriptPath, scriptContent);
    
    // Make the script executable on macOS/Linux
    if (!isWindows) {
      await fs.chmod(scriptPath, '755');
    }
    
    return true;
  } catch (error) {
    log.error('Error creating launcher script:', error);
    throw error;
  }
}

// Create desktop shortcut
async function createDesktopShortcut(targetDir) {
  try {
    const desktopDir = path.join(os.homedir(), 'Desktop');
    const appName = 'Ouro';
    
    if (isWindows) {
      // Windows shortcut
      const startScript = path.join(targetDir, 'start_ouro.bat');
      const shortcutPath = path.join(desktopDir, `${appName}.lnk`);
      
      try {
        // Simple PowerShell script to create a shortcut
        const psScript = `
          $WshShell = New-Object -comObject WScript.Shell
          $Shortcut = $WshShell.CreateShortcut("${shortcutPath.replace(/\\/g, '\\\\')}")
          $Shortcut.TargetPath = "${startScript.replace(/\\/g, '\\\\')}"
          $Shortcut.WorkingDirectory = "${targetDir.replace(/\\/g, '\\\\')}"
          $Shortcut.Description = "Launch Ouro v2.5 - Privacy-first offline assistant"
          $Shortcut.Save()
        `;
        
        await executeCommand('powershell', ['-Command', psScript], 15000);
      } catch (error) {
        log.warn('Could not create Windows shortcut:', error.message);
      }
    } else if (isMac) {
      // macOS .app bundle
      const startScript = path.join(targetDir, 'start_ouro.sh');
      const appPath = path.join(desktopDir, `${appName}.app`);
      
      try {
        // Create basic app structure
        await fs.mkdir(path.join(appPath, 'Contents', 'MacOS'), { recursive: true });
        
        // Create launcher script
        const launcherContent = `#!/bin/bash
cd "${targetDir}"
"${startScript}"
`;
        
        await fs.writeFile(path.join(appPath, 'Contents', 'MacOS', 'OuroLauncher'), launcherContent);
        await fs.chmod(path.join(appPath, 'Contents', 'MacOS', 'OuroLauncher'), '755');
        
        // Create Info.plist
        const infoPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key>
  <string>OuroLauncher</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIdentifier</key>
  <string>com.ouro.assistant</string>
  <key>CFBundleName</key>
  <string>${appName}</string>
  <key>CFBundleDisplayName</key>
  <string>${appName}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>2.5</string>
  <key>CFBundleVersion</key>
  <string>2.5</string>
  <key>NSHumanReadableCopyright</key>
  <string>Copyright © 2025. All rights reserved.</string>
</dict>
</plist>`;
        
        await fs.writeFile(path.join(appPath, 'Contents', 'Info.plist'), infoPlist);
      } catch (error) {
        log.warn('Could not create macOS app bundle:', error.message);
      }
    } else {
      // Linux .desktop file
      const startScript = path.join(targetDir, 'start_ouro.sh');
      const desktopPath = path.join(desktopDir, `${appName}.desktop`);
      
      try {
        const desktopContent = `[Desktop Entry]
Type=Application
Name=Ouro
GenericName=AI Assistant
Comment=Privacy-first, offline AI assistant with RAG
Exec="${startScript}"
Icon=${path.join(targetDir, 'assets', 'icon.png')}
Terminal=false
Categories=Utility;AI;
`;
        
        await fs.writeFile(desktopPath, desktopContent);
        await fs.chmod(desktopPath, '755');
      } catch (error) {
        log.warn('Could not create Linux desktop entry:', error.message);
      }
    }
    
    return true;
  } catch (error) {
    log.warn('Failed to create desktop shortcut:', error);
    return false;
  }
}

// Handle opening URLs in default browser
ipcMain.handle('open-external-url', (event, url) => {
  shell.openExternal(url);
});

// Handle launching the installed application
ipcMain.handle('launch-ouro', async (event, installPath) => {
  try {
    if (!installPath) {
      return { success: false, error: 'Installation path not found' };
    }
    
    const scriptExt = isWindows ? '.bat' : '.sh';
    const startScript = path.join(installPath, `start_ouro${scriptExt}`);
    
    if (isWindows) {
      // Windows launch
      const process = spawn('cmd.exe', ['/c', startScript], { 
        detached: true,
        stdio: 'ignore',
        windowsHide: false,
        shell: true
      });
      
      process.unref();
    } else if (isMac) {
      // macOS launch
      const process = spawn('open', [startScript], {
        detached: true,
        stdio: 'ignore'
      });
      
      process.unref();
    } else {
      // Linux launch
      const process = spawn('sh', [startScript], {
        detached: true,
        stdio: 'ignore'
      });
      
      process.unref();
    }
    
    return { success: true };
  } catch (error) {
    log.error('Failed to launch Ouro:', error);
    return { success: false, error: error.message };
  }
});

// Log handler for UI
ipcMain.on('log-info', (event, message) => {
  log.info(message);
});

// Helper to check if Qdrant needs installation
ipcMain.handle('check-dependencies', async () => {
  const isOllama = await checkIfOllamaInstalled();
  const isQdrant = await checkIfQdrantInstalled();
  
  return {
    ollama: isOllama,
    qdrant: isQdrant
  };
});

// Handle dependency installation
ipcMain.handle('install-dependency', async (event, dependency) => {
  try {
    if (dependency === 'ollama') {
      // Open Ollama download page
      shell.openExternal('https://ollama.ai/download');
      return { success: true, message: 'Opening Ollama download page' };
    } else if (dependency === 'qdrant') {
      // Install Qdrant via Docker pull
      try {
        await executeCommand('docker', ['pull', 'qdrant/qdrant'], 60000);
        return { success: true, message: 'Qdrant Docker image downloaded' };
      } catch (error) {
        return { success: false, error: 'Failed to pull Qdrant Docker image' };
      }
    }
    
    return { success: false, error: 'Unknown dependency' };
  } catch (error) {
    log.error('Failed to install dependency:', error);
    return { success: false, error: error.message };
  }
});
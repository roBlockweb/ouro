/**
 * Ouro v2.5 Installer
 * Handles the installation flow for Ouro
 */

// State management
let currentStep = 'welcome';
let systemInfo = null;
let installPath = null;
let selectedModel = null;
let installationData = {};

// DOM elements
document.addEventListener('DOMContentLoaded', () => {
  // Welcome page
  const agreeTermsCheckbox = document.getElementById('agree-terms');
  const welcomeNextButton = document.getElementById('welcome-next');
  
  // System check page
  const systemCheckBackButton = document.getElementById('system-check-back');
  const systemCheckNextButton = document.getElementById('system-check-next');
  
  // Dependencies page
  const dependenciesBackButton = document.getElementById('dependencies-back');
  const dependenciesNextButton = document.getElementById('dependencies-next');
  const browseButton = document.getElementById('browse-button');
  const installPathInput = document.getElementById('install-path');
  const installOllamaButton = document.getElementById('install-ollama');
  const installQdrantButton = document.getElementById('install-qdrant');
  
  // Model selection page
  const modelSelectionBackButton = document.getElementById('model-selection-back');
  const modelSelectionNextButton = document.getElementById('model-selection-next');
  const showAdvancedCheckbox = document.getElementById('show-advanced');
  const advancedSettings = document.querySelector('.advanced-settings');
  const customModelInput = document.getElementById('custom-model');
  
  // Installation page
  const installationBackButton = document.getElementById('installation-back');
  const installationNextButton = document.getElementById('installation-next');
  const progressBar = document.getElementById('install-progress-bar');
  const progressText = document.getElementById('install-progress-text');
  const installLog = document.getElementById('install-log');
  
  // Completion page
  const openDocsButton = document.getElementById('open-docs');
  const launchAppButton = document.getElementById('launch-app');
  
  // UI Navigation Logic
  const pages = {
    'welcome': document.getElementById('welcome-page'),
    'system-check': document.getElementById('system-check-page'),
    'dependencies': document.getElementById('dependencies-page'),
    'model-selection': document.getElementById('model-selection-page'),
    'installation': document.getElementById('installation-page'),
    'completion': document.getElementById('completion-page')
  };
  
  const steps = {
    'welcome': document.querySelector('.step[data-step="welcome"]'),
    'system-check': document.querySelector('.step[data-step="system-check"]'),
    'dependencies': document.querySelector('.step[data-step="dependencies"]'),
    'model-selection': document.querySelector('.step[data-step="model-selection"]'),
    'installation': document.querySelector('.step[data-step="installation"]'),
    'completion': document.querySelector('.step[data-step="completion"]')
  };
  
  // Navigation functions
  function goToPage(stepName) {
    // Hide all pages
    Object.values(pages).forEach(page => {
      page.style.display = 'none';
    });
    
    // Remove active class from all steps
    Object.values(steps).forEach(step => {
      step.classList.remove('active');
    });
    
    // Show the selected page
    pages[stepName].style.display = 'flex';
    
    // Add active class to the current step
    steps[stepName].classList.add('active');
    
    // Mark previous steps as completed
    const stepOrder = ['welcome', 'system-check', 'dependencies', 'model-selection', 'installation', 'completion'];
    const currentIndex = stepOrder.indexOf(stepName);
    
    for (let i = 0; i < currentIndex; i++) {
      steps[stepOrder[i]].classList.add('completed');
    }
    
    // Update current step
    currentStep = stepName;
    
    // Execute step initialization if needed
    if (stepName === 'system-check' && !systemInfo) {
      performSystemCheck();
    } else if (stepName === 'dependencies') {
      updateDependencyStatus();
    } else if (stepName === 'model-selection') {
      populateModelOptions();
    }
  }
  
  // Welcome page
  agreeTermsCheckbox.addEventListener('change', () => {
    welcomeNextButton.disabled = !agreeTermsCheckbox.checked;
  });
  
  welcomeNextButton.addEventListener('click', () => {
    goToPage('system-check');
  });
  
  // System check page
  systemCheckBackButton.addEventListener('click', () => {
    goToPage('welcome');
  });
  
  systemCheckNextButton.addEventListener('click', () => {
    goToPage('dependencies');
  });
  
  // Dependencies page
  dependenciesBackButton.addEventListener('click', () => {
    goToPage('system-check');
  });
  
  dependenciesNextButton.addEventListener('click', () => {
    goToPage('model-selection');
  });
  
  browseButton.addEventListener('click', async () => {
    const path = await window.api.selectInstallDirectory();
    if (path) {
      installPath = path;
      installPathInput.value = path;
      checkInstallReadiness();
    }
  });
  
  installOllamaButton.addEventListener('click', async () => {
    await window.api.installDependency('ollama');
    updateDependencyStatus();
  });
  
  installQdrantButton.addEventListener('click', async () => {
    await window.api.installDependency('qdrant');
    updateDependencyStatus();
  });
  
  // Model selection page
  modelSelectionBackButton.addEventListener('click', () => {
    goToPage('dependencies');
  });
  
  modelSelectionNextButton.addEventListener('click', () => {
    // Get selected model
    if (customModelInput.value.trim()) {
      selectedModel = customModelInput.value.trim();
    } else {
      const selectedOption = document.querySelector('.model-option.selected');
      if (selectedOption) {
        selectedModel = selectedOption.dataset.model;
      } else {
        selectedModel = 'llama3:8b'; // Default
      }
    }
    
    // Start installation
    goToPage('installation');
    startInstallation();
  });
  
  showAdvancedCheckbox.addEventListener('change', () => {
    advancedSettings.style.display = showAdvancedCheckbox.checked ? 'block' : 'none';
  });
  
  // Installation page
  installationBackButton.addEventListener('click', () => {
    goToPage('model-selection');
  });
  
  installationNextButton.addEventListener('click', () => {
    goToPage('completion');
    updateCompletionPage();
  });
  
  // Completion page
  openDocsButton.addEventListener('click', () => {
    window.api.openExternalUrl('https://github.com/yourusername/ouro/wiki');
  });
  
  launchAppButton.addEventListener('click', async () => {
    await window.api.launchOuro(installationData.installPath);
    const closeWindow = setTimeout(() => {
      window.close();
    }, 2000);
  });
  
  // System check logic
  async function performSystemCheck() {
    const systemChecking = document.querySelector('.system-checking');
    const systemInfoElement = document.querySelector('.system-info');
    
    systemChecking.style.display = 'flex';
    systemInfoElement.style.display = 'none';
    
    try {
      // Get system information
      systemInfo = await window.api.getSystemInfo();
      
      // Update UI
      document.getElementById('os-info').textContent = getOSName(systemInfo.platform);
      document.getElementById('cpu-info').textContent = `${systemInfo.cpuCores} cores`;
      document.getElementById('memory-info').textContent = `${systemInfo.memory} GB`;
      
      const ollamaInfo = document.getElementById('ollama-info');
      ollamaInfo.textContent = systemInfo.isOllama ? 'Installed ✓' : 'Not Installed ✗';
      ollamaInfo.className = systemInfo.isOllama ? 'info-value success' : 'info-value warning';
      
      const qdrantInfo = document.getElementById('qdrant-info');
      qdrantInfo.textContent = systemInfo.isQdrant ? 'Installed ✓' : 'Not Installed ✗';
      qdrantInfo.className = systemInfo.isQdrant ? 'info-value success' : 'info-value warning';
      
      // Show recommendations
      const recommendationsList = document.getElementById('recommendations-list');
      recommendationsList.innerHTML = '';
      
      const recommendations = await window.api.getModelRecommendations(systemInfo);
      
      recommendations.forEach(rec => {
        if (rec.recommended) {
          selectedModel = rec.name; // Set default selected model
        }
        
        const recElement = document.createElement('div');
        recElement.className = 'recommendation-item';
        recElement.innerHTML = `
          <p><strong>${rec.name}</strong> (${rec.parameters}) - ${rec.description}</p>
          <p>Required Memory: ${rec.requiredMemory}GB</p>
        `;
        
        recommendationsList.appendChild(recElement);
      });
      
      // Hide loader and show info
      systemChecking.style.display = 'none';
      systemInfoElement.style.display = 'block';
      
      // Set default install path
      if (!installPath) {
        if (systemInfo.platform === 'darwin') {
          installPath = `${process.env.HOME}/Applications/Ouro`;
        } else if (systemInfo.platform === 'win32') {
          installPath = 'C:\\Program Files\\Ouro';
        } else {
          installPath = `${process.env.HOME}/ouro`;
        }
        
        installPathInput.value = installPath;
      }
    } catch (error) {
      console.error('Error during system check:', error);
      
      // Show error in UI
      systemChecking.style.display = 'none';
      systemInfoElement.style.display = 'block';
      
      const recommendationsList = document.getElementById('recommendations-list');
      recommendationsList.innerHTML = `<p class="error">Error checking system: ${error.message || 'Unknown error'}</p>`;
    }
  }
  
  // Get OS name
  function getOSName(platform) {
    switch (platform) {
      case 'darwin':
        return 'macOS';
      case 'win32':
        return 'Windows';
      case 'linux':
        return 'Linux';
      default:
        return platform;
    }
  }
  
  // Update dependency status
  async function updateDependencyStatus() {
    try {
      const dependencies = await window.api.checkDependencies();
      
      const ollamaStatus = document.getElementById('ollama-status');
      ollamaStatus.textContent = dependencies.ollama ? 'Installed ✓' : 'Not Installed ✗';
      ollamaStatus.className = dependencies.ollama ? 'dependency-status installed' : 'dependency-status not-installed';
      
      const qdrantStatus = document.getElementById('qdrant-status');
      qdrantStatus.textContent = dependencies.qdrant ? 'Installed ✓' : 'Not Installed ✗';
      qdrantStatus.className = dependencies.qdrant ? 'dependency-status installed' : 'dependency-status not-installed';
      
      // Check if we have all requirements to proceed
      checkInstallReadiness();
    } catch (error) {
      console.error('Error checking dependencies:', error);
    }
  }
  
  // Check if we're ready to proceed with installation
  function checkInstallReadiness() {
    const ollamaInstalled = document.getElementById('ollama-status').textContent.includes('Installed');
    const installPathProvided = installPathInput.value.trim() !== '';
    
    // Enable next button if path is provided and Ollama is installed
    // Qdrant is optional as we'll set it up during installation
    dependenciesNextButton.disabled = !(ollamaInstalled && installPathProvided);
  }
  
  // Populate model options
  async function populateModelOptions() {
    const modelOptionsList = document.getElementById('model-options-list');
    modelOptionsList.innerHTML = '';
    
    try {
      const recommendations = await window.api.getModelRecommendations(systemInfo);
      
      recommendations.forEach(model => {
        const modelOption = document.createElement('div');
        modelOption.className = 'model-option';
        modelOption.dataset.model = model.name;
        
        if (model.recommended) {
          modelOption.classList.add('selected');
        }
        
        modelOption.innerHTML = `
          <div class="header">
            <div class="model-name">${model.name}</div>
            <div class="model-size">${model.parameters}</div>
          </div>
          <div class="model-description">${model.description}</div>
          <div class="model-requirements">
            <p>Required Memory: ${model.requiredMemory}GB</p>
          </div>
        `;
        
        modelOption.addEventListener('click', () => {
          // Remove selected class from all options
          document.querySelectorAll('.model-option').forEach(option => {
            option.classList.remove('selected');
          });
          
          // Add selected class to this option
          modelOption.classList.add('selected');
          
          // Clear custom model input
          customModelInput.value = '';
        });
        
        modelOptionsList.appendChild(modelOption);
      });
    } catch (error) {
      console.error('Error getting model recommendations:', error);
      modelOptionsList.innerHTML = '<p class="error">Error loading model recommendations</p>';
    }
  }
  
  // Start installation
  async function startInstallation() {
    try {
      // Disable back button during installation
      installationBackButton.disabled = true;
      
      // Set up installation options
      const options = {
        installPath: installPath,
        selectedModel: selectedModel,
        apiKeys: {}
      };
      
      // Set up progress handlers
      const removeProgressHandler = window.api.onInstallationProgress(updateProgress);
      const removeFileProgressHandler = window.api.onFileCopyProgress(updateFileProgress);
      
      // Start installation
      updateInstallLog('Starting installation...');
      installationData = await window.api.startInstallation(options);
      
      // Clean up handlers
      removeProgressHandler();
      removeFileProgressHandler();
      
      // Enable next button
      installationNextButton.disabled = false;
      
      // Complete installation
      updateProgress({
        step: 'complete',
        progress: 100,
        message: 'Installation completed successfully!'
      });
      
      updateInstallLog('Installation completed successfully!');
    } catch (error) {
      console.error('Installation error:', error);
      updateInstallLog(`Error during installation: ${error.message || 'Unknown error'}`);
      
      // Re-enable back button
      installationBackButton.disabled = false;
    }
  }
  
  // Update installation progress
  function updateProgress(data) {
    progressBar.style.width = `${data.progress}%`;
    progressText.textContent = data.message;
    
    // Update step statuses
    const steps = {
      'prepare': 'copying',
      'copy': 'copying',
      'configure': 'configuring',
      'shortcut': 'shortcuts',
      'complete': 'finalizing'
    };
    
    if (steps[data.step]) {
      const stepElement = document.querySelector(`.install-step[data-step="${steps[data.step]}"] .step-status`);
      stepElement.className = 'step-status in-progress';
      
      // Mark previous steps as completed
      const stepOrder = ['copying', 'configuring', 'shortcuts', 'finalizing'];
      const currentIndex = stepOrder.indexOf(steps[data.step]);
      
      for (let i = 0; i < currentIndex; i++) {
        const prevStepElement = document.querySelector(`.install-step[data-step="${stepOrder[i]}"] .step-status`);
        prevStepElement.className = 'step-status completed';
      }
      
      // If complete, mark all steps as completed
      if (data.step === 'complete') {
        document.querySelectorAll('.install-step .step-status').forEach(stepStatus => {
          stepStatus.className = 'step-status completed';
        });
      }
    }
  }
  
  // Update file copy progress
  function updateFileProgress(data) {
    updateInstallLog(`Copying ${data.file}: ${data.percentage}%`);
  }
  
  // Update installation log
  function updateInstallLog(message) {
    const log = document.getElementById('install-log');
    const timestamp = new Date().toLocaleTimeString();
    log.innerHTML += `[${timestamp}] ${message}<br>`;
    log.scrollTop = log.scrollHeight;
  }
  
  // Update completion page
  function updateCompletionPage() {
    document.getElementById('summary-location').textContent = installationData.installPath || installPath;
    document.getElementById('summary-model').textContent = installationData.selectedModel || selectedModel;
  }
  
  // Start the installer flow
  goToPage('welcome');
});
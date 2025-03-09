// DOM Elements
const chatContainer = document.getElementById('chat-container');
const chatMessages = document.getElementById('chat-messages');
const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const settingsForm = document.getElementById('settings-form');
const fileUploadForm = document.getElementById('file-upload-form');
const textIngestForm = document.getElementById('text-ingest-form');
const clearMemoryBtn = document.getElementById('clear-memory-btn');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperature-value');

// Helper functions
function addMessage(content, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'assistant-message');
    messageDiv.textContent = content;
    
    chatMessages.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showNotification(message, isError = false) {
    const notification = document.createElement('div');
    notification.classList.add('notification');
    if (isError) {
        notification.classList.add('error');
    }
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Event listeners
temperatureSlider.addEventListener('input', function() {
    temperatureValue.textContent = this.value;
});

queryForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) return;
    
    // Add user message to chat
    addMessage(query, true);
    
    // Clear input
    queryInput.value = '';
    
    try {
        // Create a placeholder for assistant response
        const responsePlaceholder = document.createElement('div');
        responsePlaceholder.classList.add('message', 'assistant-message');
        chatMessages.appendChild(responsePlaceholder);
        
        // Fetch streaming response
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                use_history: true
            }),
        });
        
        const reader = response.body.getReader();
        let assistantResponse = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // Decode chunk
            const chunk = new TextDecoder().decode(value);
            assistantResponse += chunk;
            
            // Update placeholder with current response
            responsePlaceholder.textContent = assistantResponse;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, true);
    }
});

settingsForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const settings = {
        model_name: formData.get('model_name'),
        top_k: parseInt(formData.get('top_k')),
        max_new_tokens: parseInt(formData.get('max_new_tokens')),
        temperature: parseFloat(formData.get('temperature'))
    };
    
    try {
        const response = await fetch('/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('Settings updated successfully');
        } else {
            showNotification('Error updating settings: ' + data.message, true);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, true);
    }
});

fileUploadForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a file to upload', true);
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showNotification('Uploading file...');
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('File uploaded and queued for ingestion');
            fileInput.value = '';
        } else {
            showNotification('Error uploading file: ' + data.message, true);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, true);
    }
});

textIngestForm.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const textInput = document.getElementById('text-ingest');
    const text = textInput.value.trim();
    
    if (!text) {
        showNotification('Please enter text to ingest', true);
        return;
    }
    
    try {
        showNotification('Ingesting text...');
        
        const response = await fetch('/ingest/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text
            }),
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('Text ingested successfully');
            textInput.value = '';
        } else {
            showNotification('Error ingesting text: ' + data.message, true);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, true);
    }
});

clearMemoryBtn.addEventListener('click', async function() {
    try {
        const response = await fetch('/clear-memory', {
            method: 'POST',
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showNotification('Conversation memory cleared');
            
            // Clear chat display
            while (chatMessages.firstChild) {
                chatMessages.removeChild(chatMessages.firstChild);
            }
            
            // Add welcome message again
            const welcomeMessage = document.createElement('div');
            welcomeMessage.classList.add('welcome-message');
            welcomeMessage.innerHTML = `
                <h2>Welcome to Ouro</h2>
                <p>Your privacy-focused local RAG system. Ask questions about your documents or use the settings panel to configure the system.</p>
            `;
            chatMessages.appendChild(welcomeMessage);
        } else {
            showNotification('Error clearing memory: ' + data.message, true);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error: ' + error.message, true);
    }
});
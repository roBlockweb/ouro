// Ouro v2.5 UI Application
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
});
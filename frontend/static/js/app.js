class HealthcareChatbot {
  constructor() {
    this.elements = {
      chatbotButton: document.getElementById('chatbotButton'),
      chatbotWindow: document.getElementById('chatbotWindow'),
      closeChatbot: document.getElementById('closeChatbot'),
      chatInput: document.getElementById('chatInput'),
      sendMessage: document.getElementById('sendMessage'),
      chatMessages: document.getElementById('chatMessages')
    };

    // Rasa server configuration
    this.rasaConfig = {
      serverUrl: 'http://localhost:5005', // Rasa server URL
      actionUrl: 'http://localhost:5055/webhook', // Action server URL
      endpoints: {
        sendMessage: '/webhooks/rest/webhook'
      }
    };

    // Generate unique session ID
    this.sessionId = 'user_' + Math.random().toString(36).substr(2, 9);
    this.isWaitingForResponse = false;

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.loadInitialGreeting();
  }

  loadInitialGreeting() {
    const initialGreetingMessage = "Xin chào! Tôi có thể giúp gì cho bạn hôm nay?";
    this.addMessage(initialGreetingMessage, 'bot');
  }

  setupEventListeners() {
    this.elements.chatbotButton.addEventListener('click', () => {
      this.toggleChatbotWindow();
    });

    this.elements.closeChatbot.addEventListener('click', () => {
      this.closeChatbotWindow();
    });

    this.elements.sendMessage.addEventListener('click', () => {
      this.sendMessageToRasa();
    });

    this.elements.chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        this.sendMessageToRasa();
      }
    });
  }

  async sendMessageToRasa() {
    const message = this.elements.chatInput.value.trim();
    if (message && !this.isWaitingForResponse) {
      this.addMessage(message, 'user');
      this.elements.chatInput.value = '';
      this.scrollToBottom();

      this.showTypingIndicator();
      this.isWaitingForResponse = true;

      try {
        const response = await this.fetchRasaResponse(message);
        this.handleRasaResponse(response);
      } catch (error) {
        console.error('Error communicating with Rasa:', error);
        this.addMessage("Tôi đang gặp sự cố khi kết nối với máy chủ. Vui lòng thử lại sau.", 'bot');
      } finally {
        this.removeTypingIndicator();
        this.isWaitingForResponse = false;
      }
    }
  }

  async fetchRasaResponse(message) {
    const response = await fetch(`${this.rasaConfig.serverUrl}${this.rasaConfig.endpoints.sendMessage}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sender: this.sessionId,
        message: message
      })
    });

    if (!response.ok) {
      throw new Error(`Rasa server responded with status ${response.status}`);
    }

    return await response.json();
  }

  handleRasaResponse(rasaMessages) {
    rasaMessages.forEach((messageObj, index) => {
      setTimeout(() => {
        // Handle text responses
        if (messageObj.text) {
          this.addMessage(messageObj.text, 'bot');
        }

        // Handle buttons
        if (messageObj.buttons && messageObj.buttons.length > 0) {
          this.addButtons(messageObj.buttons);
        }

        // Handle custom payloads
        if (messageObj.custom) {
          this.handleCustomPayload(messageObj.custom);
        }

        // Handle images
        if (messageObj.image) {
          this.addImage(messageObj.image);
        }
      }, index * 300);
    });
  }

  addButtons(buttons) {
    const buttonsContainer = document.createElement('div');
    buttonsContainer.className = 'mt-2 flex flex-wrap gap-2';

    buttons.forEach(button => {
      const btn = document.createElement('button');
      btn.className = 'bg-blue-100 hover:bg-blue-200 text-blue-800 text-sm py-1 px-3 rounded-md transition-colors';
      btn.textContent = button.title;
      btn.addEventListener('click', () => {
        // Remove buttons after selection
        buttonsContainer.remove();
        this.addMessage(button.title, 'user');
        this.sendMessageToRasa(button.payload || button.title);
      });
      buttonsContainer.appendChild(btn);
    });

    const lastMessage = this.elements.chatMessages.lastElementChild;
    if (lastMessage) {
      lastMessage.querySelector('div').appendChild(buttonsContainer);
      this.scrollToBottom();
    }
  }

  addImage(imageUrl) {
    const imgContainer = document.createElement('div');
    imgContainer.className = 'mt-2';

    const img = document.createElement('img');
    img.src = imageUrl;
    img.className = 'max-w-xs rounded-lg border border-gray-200';
    img.alt = 'Chatbot response image';

    imgContainer.appendChild(img);

    const lastMessage = this.elements.chatMessages.lastElementChild;
    if (lastMessage) {
      lastMessage.querySelector('div').appendChild(imgContainer);
      this.scrollToBottom();
    }
  }

  handleCustomPayload(payload) {
    // Implement specific handling for custom payloads
    console.log('Custom payload received:', payload);

    // Example: Handle quick replies
    if (payload.quick_replies) {
      this.addButtons(payload.quick_replies.map(reply => ({
        title: reply.title,
        payload: reply.payload
      })));
    }

    // Add more custom payload handlers as needed
  }

  // Methods for toggling and closing the chatbot window
  toggleChatbotWindow() {
    this.elements.chatbotWindow.classList.toggle('hidden');
    this.elements.chatbotWindow.classList.toggle('open');
  }

  closeChatbotWindow() {
    this.elements.chatbotWindow.classList.add('hidden');
    this.elements.chatbotWindow.classList.remove('open');
  }

  addMessage(message, sender) {
    const messageContainer = document.createElement('div');
    messageContainer.className = sender === 'user' ? 'user-message' : 'bot-message';

    const messageText = document.createElement('p');
    messageText.textContent = message;

    messageContainer.appendChild(messageText);
    this.elements.chatMessages.appendChild(messageContainer);
    this.scrollToBottom();
  }

  scrollToBottom() {
    this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
  }

  showTypingIndicator() {
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.textContent = 'Bot is typing...';

    this.elements.chatMessages.appendChild(typingIndicator);
    this.scrollToBottom();
  }

  removeTypingIndicator() {
    const typingIndicator = this.elements.chatMessages.querySelector('.typing-indicator');
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new HealthcareChatbot();
});

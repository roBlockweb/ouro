🧠 OuroGPT - The Smart AI Chatbot

Welcome to OuroGPT! This powerful AI chatbot system features two intelligent bots, Ouro and Brain, who engage in dynamic conversations, continuously learn, and even search the web for new information. Whether you're looking for an engaging chatbot or an AI assistant, OuroGPT delivers a cutting-edge experience.

🌟 Key Features

🤖 AI Conversations – Ouro and Brain chat endlessly, refining their responses over time.

📚 Memory Storage – They remember past conversations for more relevant and insightful replies.

🌍 Web Search Capability – They can browse the internet for up-to-date information.

💾 Conversation Logging – Every interaction is recorded for review and learning.

🚀 Fully Offline Mode – No internet connection is needed for chat functionality.

📂 Project Structure

.
├── 🤖 Ouro & Brain – AI chatbots
├── 📚 Memory System – Stores learned knowledge
├── 🔍 Web Search – Fetches new information
├── ⚙️ Configuration – Adjustable settings
├── 🚀 Startup Script – Launches the AI system
└── 📖 Documentation – User guide and setup instructions

🔧 Installation Guide

1️⃣ Download OuroGPT

git clone https://github.com/roBlockweb/ouro.git
cd ouro

2️⃣ Set Up a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

3️⃣ Install Dependencies

pip install --upgrade pip
pip install -r requirements.txt

4️⃣ Initialize the Memory System

python initialize_faiss.py

🚀 Running OuroGPT

🤖 Autonomous Mode (Bots Chat Automatically)

python main.py --mode=autonomous

Press Ctrl+C to stop.

Conversations are stored in conversation_logs/.

💬 Interactive Mode (Chat Directly with Brain)

python main.py --mode=interactive

You type messages, and Brain responds in real time!

📦 Data Storage

📚 Memory System – Stored in faiss_store/index.faiss

📖 Chat Logs – Saved in conversation_logs/ (new log for each day)

❓ Troubleshooting

🔧 Bots Won’t Start?

✔️ Ensure the AI model file is in the correct directory.
✔️ Check config.py to confirm all settings are correct.

🌍 Web Search Not Working?

✔️ Verify that Chrome and ChromeDriver are installed.
✔️ The bots operate in headless mode, so no browser window will appear.

🖥️ Running Slowly?

✔️ The AI model requires significant memory.
✔️ Try using a smaller, lower-precision model for improved performance.

📜 License

This project is open-source! Feel free to use, modify, and share it.

❤️ Created by Rohan & Team

Enjoying OuroGPT? Share it with others and contribute to the project!

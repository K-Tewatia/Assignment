# Marketing ROI Optimizer
AI-powered marketing analytics platform for data-driven budget optimization using RAG and LLMs.

What Does This App Do?
This application helps marketing teams analyze performance and optimize budget allocation across multiple marketing channels using:

AI Analysis: Get insights from your marketing data using Large Language Models (Mistral)

Budget Optimization: Mathematical optimization to maximize ROI/conversions

Data Visualization: Interactive charts and performance dashboards

Smart Reports: Generate comprehensive reports in Text/Markdown/CSV formats

Q&A System: Ask questions about your data in natural language

Core Workflow
text
Upload Data → AI Analysis → Budget Optimization → Generate Reports
⚡ Quick Start (5 Minutes)
1. Install Dependencies
bash
# Clone or download the project
cd marketing-roi-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install required packages
pip install -r requirements.txt

3. Run the Application
bash
streamlit run app.py
The app will open in your browser at http://localhost:8501

📁 Project Structure
text
Assignment/
|- campaign-channel-conversions # Dataset to be uploaded
|- setup_check.py             # TO TEST THE WORKING OF APP
|- outputs                    # Reports
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
├── README.md                 # This file
└── src/                      # Source code modules
    ├── config.py             # Configuration & API key management
    ├── data_processor.py     # Data cleaning & validation
    ├── pinecone_manager.py   # Vector database operations (RAG)
    ├── mistral_agent.py      # LLM agent for AI insights
    └── optimizer.py          # Budget optimization algorithms

📖 How to Use
Step 1: Upload Data
Go to "📥 Data Input" page

Upload your CSV file OR generate sample data

Click "Upload to Pinecone"

Step 2: Run Analysis
Go to "📊 AI Analysis" page

Click "Run AI Analysis"

Review performance metrics and AI insights

Step 3: Optimize Budget
Go to "Budget Optimization" page

Set total budget and constraints

Click "Optimize Budget Allocation"

Review recommendations

Step 4: Generate Report
Go to "📄 Generate Report" page

Configure report settings

Click "Generate Report"

Download in your preferred format

Step 5: Ask Questions (Optional)
Go to "❓ Q&A Chat" page

Ask questions about your data

Troubleshooting
"No data found in Pinecone"
Wait 10-15 seconds after upload for index to propagate, or use the automatic fallback to session data.

"API error 429 - Rate limit exceeded"
Solution: Replace the Mistral api in env

Solution:

bash
pip uninstall pinecone-client pinecone -y
pip install "pinecone-client>=3.0.0"

Requirements
Python Version: 3.12 or higher

Key Dependencies:
streamlit>=1.28.0
pinecone-client>=3.0.0
groq>=0.4.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
plotly>=5.17.0
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
Install all at once:

bash
pip install -r requirements.txt

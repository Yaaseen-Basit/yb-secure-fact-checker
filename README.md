# Secure AI Fact-Checker with Phishing/URL Scanner

This project provides an AI-powered misinformation detection and cybersecurity awareness tool. It performs sentence-level fact-checking, topic detection, phishing/URL scanning, and offers domain-specific authentic tips. The system is designed with dynamic model selection to balance accuracy and computational resources.

## Features
- Sentence-level classification as True, Misleading, or False with confidence scores.  
- Dynamic selection between lightweight and large models depending on system resources.  
- Topic detection across Health, Finance, Politics, Cybersecurity, and General domains.  
- Domain-specific authentic guidance based on detected topics.  
- URL scanner that flags suspicious or potentially phishing links.  
- Streamlit interface for ease of use.

## Technology Stack
- Python 3.9+  
- Streamlit for the user interface  
- Hugging Face Transformers for NLP models  
- NLTK for sentence tokenization  
- langdetect for language detection  
- Regular expressions for phishing URL heuristics  

## Project Structure
app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env # Environment variables (not committed to git)
└── README.md # Project documentation

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Yaaseen-Basit/yb-secure-fact-checker.git
cd secure-fact-checker

python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows

pip install -r requirements.txt

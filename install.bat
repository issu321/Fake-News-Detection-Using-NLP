@echo off
echo 🛡️ Fake News Detection Setup (Windows)
echo =====================================

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.11+.
    pause
    exit /b 1
)

echo 📦 Creating virtual environment...
python -m venv venv

echo 🚀 Activating environment...
call venv\Scripts\activate.bat

echo ⬇️ Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo 📥 Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo ✅ Installation complete!
echo 🌐 Starting Streamlit...
streamlit run app.py

pause

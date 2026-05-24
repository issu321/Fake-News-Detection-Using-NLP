#!/bin/bash

# ============================================================
#  FAKE NEWS DETECTION SYSTEM - LINUX INSTALLER
#  Developed by issu321
# ============================================================

set -e

# =========================
# COLORS
# =========================
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

clear

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           🛡️ FAKE NEWS DETECTION INSTALLER 🛡️             ║"
echo "║                                                              ║"
echo "║                  Developed by issu321                       ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "📰 AI-Based Fake News Classification"
echo -e "🤖 NLP Powered News Analysis"
echo -e "📊 Real-Time Prediction Dashboard"
echo -e "⚡ Streamlit Machine Learning Interface"
echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================================
# VENV CONFIRMATION
# ============================================================

echo -e "${YELLOW}[IMPORTANT NOTICE]${NC}"
echo ""
echo "Python Virtual Environment (venv) is REQUIRED."
echo ""
echo "Before continuing:"
echo " • Create a Python virtual environment"
echo " • Activate the venv"
echo ""
echo -e "Type ${GREEN}yes${NC}  -> Continue installation"
echo -e "Type ${RED}exit${NC} -> Stop installer"
echo ""

read -p "Enter choice (yes/exit): " USER_INPUT

# ============================================================
# EXIT SAFELY
# ============================================================

if [ "$USER_INPUT" = "exit" ]; then
    echo ""
    echo -e "${RED}[EXIT]${NC} Installer terminated by user."
    echo ""
    echo "Create and activate venv first:"
    echo "--------------------------------------------------"
    echo "python3 -m venv venv"
    echo "source venv/bin/activate"
    echo "bash install.sh"
    echo "--------------------------------------------------"
    echo ""
    exit 1
fi

# ============================================================
# INVALID INPUT
# ============================================================

if [ "$USER_INPUT" != "yes" ]; then
    echo ""
    echo -e "${RED}[ERROR]${NC} Invalid input detected."
    echo "Run installer again and type only: yes or exit"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}[OK]${NC} Continuing installation..."
echo ""

# ============================================================
# PYTHON VERSION CHECK
# ============================================================

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info.major, sys.version_info.minor)' | tr ' ' '.')

echo -e "${GREEN}[OK]${NC} Python $PYTHON_VERSION detected"
echo ""

# ============================================================
# INSTALLATION PROCESS
# ============================================================

echo -e "${BLUE}[1/4]${NC} Upgrading pip..."
pip install --upgrade pip -q

echo ""
echo -e "${BLUE}[2/4]${NC} Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo -e "${BLUE}[3/4]${NC} Downloading NLTK datasets..."
python3 -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo ""
echo -e "${BLUE}[4/4]${NC} Finalizing AI environment..."
sleep 1

# ============================================================
# INSTALLATION COMPLETE
# ============================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║               ✅ INSTALLATION COMPLETE ✅                   ║"
echo "║                                                              ║"
echo "║          Fake News Detection System is Ready                ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo -e "${GREEN}[SUCCESS]${NC} NLP dependencies and datasets installed."
echo ""

# ============================================================
# LAUNCH APPLICATION
# ============================================================

echo -e "${CYAN}🌐 Launching Fake News Detection Dashboard...${NC}"
echo -e "${GREEN}🚀 Streamlit Server Starting...${NC}"
echo ""

python3 -m streamlit run app.py
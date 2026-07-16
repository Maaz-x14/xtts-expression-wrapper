#!/bin/bash
set -e

echo "[SETUP] Checking Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo "[SETUP] Python 3.11 not found. Installing..."
    sudo apt update -qq
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

echo "[SETUP] Creating virtual environment with Python 3.11..."
python3.11 -m venv venv

echo "[SETUP] Activating venv..."
source venv/bin/activate

echo "[SETUP] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[SETUP] Installing PyTorch CPU (this may take a few minutes)..."
pip install torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu \
    --quiet

echo "[SETUP] Pinning transformers (BeamSearchScorer removed in 4.41+)..."
pip install "transformers==4.40.3" --quiet

echo "[SETUP] Installing project requirements..."
pip install -r requirements.txt --quiet

echo ""
echo "[DONE] Setup complete."
echo "[DONE] To activate: source venv/bin/activate"
echo "[DONE] To download RAVDESS clips: python scripts/download_ravdess.py"

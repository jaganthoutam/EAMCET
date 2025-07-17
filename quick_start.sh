#!/bin/bash

echo "⚡ EAMCET AI Tutor Quick Start"
echo "=============================="

# Check if virtual environment exists
if [ ! -d "../eamcet_env" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source ../eamcet_env/bin/activate

echo "🧪 Testing setup..."
python test_setup.py

echo ""
echo "📁 Please copy your EAMCET PDFs to data/raw_pdfs/ folder"
echo "Then run: python train.py --data_path data/raw_pdfs --stage extract"

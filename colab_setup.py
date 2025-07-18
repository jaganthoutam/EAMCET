#!/usr/bin/env python3
"""
Google Colab Setup for EAMCET Zero Manual Pipeline
This script sets up the environment and runs the automated extraction and training
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def setup_colab_environment():
    """Setup the Colab environment with all dependencies"""
    print("🚀 Setting up EAMCET Zero Manual Pipeline on Google Colab")
    print("=" * 60)
    
    # Step 1: Clone the repository
    print("\n📥 Step 1: Cloning repository from GitHub...")
    repo_url = "https://github.com/jaganthoutam/EAMCET.git"  # Update with your actual repo URL
    
    try:
        subprocess.run([
            "git", "clone", repo_url, "EAMCET"
        ], check=True, capture_output=True, text=True)
        print("✅ Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error cloning repository: {e}")
        return False
    
    # Step 2: Install system dependencies
    print("\n🔧 Step 2: Installing system dependencies...")
    system_deps = [
        "tesseract-ocr",
        "tesseract-ocr-eng",
        "libtesseract-dev",
        "libgl1-mesa-glx",
        "libglib2.0-0"
    ]
    
    for dep in system_deps:
        try:
            subprocess.run([
                "apt-get", "install", "-y", dep
            ], check=True, capture_output=True, text=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not install {dep} (may already be installed)")
    
    # Step 3: Install Python dependencies
    print("\n🐍 Step 3: Installing Python dependencies...")
    
    # Change to the project directory
    os.chdir("EAMCET")
    
    # Install requirements
    try:
        subprocess.run([
            "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True, text=True)
        print("✅ Python dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False
    
    # Step 4: Verify PyMuPDF installation
    print("\n🔍 Step 4: Verifying PyMuPDF installation...")
    try:
        import fitz
        print("✅ PyMuPDF imported successfully")
    except ImportError as e:
        print(f"❌ PyMuPDF import failed: {e}")
        return False
    
    print("\n✅ Environment setup complete!")
    return True

def run_zero_manual_pipeline(data_folder="/content/EAMCET/data/raw_pdfs"):
    """Run the zero manual pipeline"""
    print("\n🚀 Running EAMCET Zero Manual Pipeline")
    print("=" * 50)
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        print("Please upload your EAMCET PDFs to the data/raw_pdfs folder")
        return False
    
    # Run the pipeline
    try:
        result = subprocess.run([
            "python", "eamcet_zero_manual_pipeline.py",
            "--input_folder", data_folder,
            "--output_folder", "colab_results"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Pipeline completed successfully!")
        print("\n📊 Pipeline Output:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_colab_notebook():
    """Create a Colab notebook template"""
    notebook_content = '''{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "header"
      },
      "source": [
        "# EAMCET Zero Manual Pipeline - Google Colab\n",
        "\n",
        "This notebook runs the fully automated EAMCET AI tutor training pipeline without any manual annotations required.\n",
        "\n",
        "## Features:\n",
        "- ✅ No manual annotations needed\n",
        "- ✅ Intelligent pattern recognition\n",
        "- ✅ Automatic question and answer extraction\n",
        "- ✅ Automated model training\n",
        "- ✅ Works with any EAMCET PDF format"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "setup"
      },
      "source": [
        "## Step 1: Setup Environment\n",
        "\n",
        "Install all dependencies and clone the repository."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "setup_code"
      },
      "outputs": [],
      "source": [
        "# Install system dependencies\n",
        "!apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev libgl1-mesa-glx libglib2.0-0\n",
        "\n",
        "# Clone repository (update with your actual GitHub URL)\n",
        "!git clone https://github.com/your-username/EAMCET.git\n",
        "\n",
        "# Install Python dependencies\n",
        "%cd EAMCET\n",
        "!pip install -r requirements.txt\n",
        "\n",
        "# Verify installation\n",
        "import fitz\n",
        "print(\"✅ PyMuPDF imported successfully\")\n",
        "print(\"✅ Environment ready!\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "upload"
      },
      "source": [
        "## Step 2: Upload Your EAMCET PDFs\n",
        "\n",
        "Upload your EAMCET PDF files to the data/raw_pdfs folder."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "upload_code"
      },
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# Create data directory structure\n",
        "os.makedirs(\"data/raw_pdfs\", exist_ok=True)\n",
        "\n",
        "# Upload files\n",
        "uploaded = files.upload()\n",
        "\n",
        "# Move uploaded files to data folder\n",
        "for filename in uploaded.keys():\n",
        "    if filename.endswith('.pdf'):\n",
        "        os.rename(filename, f\"data/raw_pdfs/{filename}\")\n",
        "        print(f\"✅ Moved {filename} to data/raw_pdfs/\")\n",
        "\n",
        "print(f\"\\n📁 Total PDFs uploaded: {len([f for f in os.listdir('data/raw_pdfs') if f.endswith('.pdf')])}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "run"
      },
      "source": [
        "## Step 3: Run Zero Manual Pipeline\n",
        "\n",
        "Execute the fully automated extraction and training pipeline."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "run_code"
      },
      "outputs": [],
      "source": [
        "# Run the zero manual pipeline\n",
        "!python eamcet_zero_manual_pipeline.py --input_folder data/raw_pdfs --output_folder colab_results\n",
        "\n",
        "print(\"\\n✅ Pipeline completed!\")\n",
        "print(\"📁 Check the colab_results folder for outputs\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "results"
      },
      "source": [
        "## Step 4: View Results\n",
        "\n",
        "Examine the extracted data and training results."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "results_code"
      },
      "outputs": [],
      "source": [
        "import json\n",
        "import pandas as pd\n",
        "from pathlib import Path\n",
        "\n",
        "# Load pipeline summary\n",
        "if Path(\"colab_results/pipeline_summary.json\").exists():\n",
        "    with open(\"colab_results/pipeline_summary.json\", \"r\") as f:\n",
        "        summary = json.load(f)\n",
        "    \n",
        "    print(\"📊 PIPELINE SUMMARY:\")\n",
        "    print(\"=\" * 40)\n",
        "    \n",
        "    print(f\"Total questions extracted: {summary['extraction_stats']['total_questions']}\")\n",
        "    print(f\"Questions with answers: {summary['extraction_stats']['paired_questions']}\")\n",
        "    \n",
        "    print(\"\\n📚 Subject Breakdown:\")\n",
        "    for subject, count in summary['extraction_stats']['subjects'].items():\n",
        "        if count > 0:\n",
        "            print(f\"  {subject}: {count} questions\")\n",
        "    \n",
        "    print(\"\\n🎯 Training Data Created:\")\n",
        "    for data_type, count in summary['training_data_stats'].items():\n",
        "        print(f\"  {data_type}: {count} samples\")\n",
        "else:\n",
        "    print(\"❌ Pipeline summary not found. Check if pipeline completed successfully.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "download"
      },
      "source": [
        "## Step 5: Download Results\n",
        "\n",
        "Download the processed data and trained models."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "download_code"
      },
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "import zipfile\n",
        "import os\n",
        "\n",
        "# Create zip file of results\n",
        "if os.path.exists(\"colab_results\"):\n",
        "    with zipfile.ZipFile(\"eamcet_results.zip\", \"w\") as zipf:\n",
        "        for root, dirs, files in os.walk(\"colab_results\"):\n",
        "            for file in files:\n",
        "                file_path = os.path.join(root, file)\n",
        "                arcname = os.path.relpath(file_path, \"colab_results\")\n",
        "                zipf.write(file_path, arcname)\n",
        "    \n",
        "    # Download the zip file\n",
        "    files.download(\"eamcet_results.zip\")\n",
        "    print(\"✅ Results downloaded as eamcet_results.zip\")\n",
        "else:\n",
        "    print(\"❌ No results found to download\")"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "name": "EAMCET Zero Manual Pipeline",
      "private_outputs": true,
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}'''
    
    with open("eamcet_colab_notebook.ipynb", "w") as f:
        f.write(notebook_content)
    
    print("✅ Created Colab notebook: eamcet_colab_notebook.ipynb")

def create_requirements_colab():
    """Create a Colab-specific requirements file"""
    colab_requirements = '''# Colab-specific requirements for EAMCET Zero Manual Pipeline
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
PyMuPDF>=1.22.0
opencv-python-headless>=4.7.0  # Use headless version for Colab
pytesseract>=0.3.10
Pillow>=9.5.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
albumentations>=1.3.0
fastapi>=0.100.0
uvicorn>=0.22.0
requests>=2.31.0
tqdm>=4.65.0
wandb>=0.15.0
tensorboard>=2.13.0
# Additional Colab-specific packages
google-colab
ipywidgets
'''
    
    with open("requirements_colab.txt", "w") as f:
        f.write(colab_requirements)
    
    print("✅ Created Colab requirements: requirements_colab.txt")

def create_readme_colab():
    """Create a Colab-specific README"""
    readme_content = '''# EAMCET Zero Manual Pipeline - Google Colab Setup

This repository contains a fully automated EAMCET AI tutor training pipeline that requires **zero manual annotations**.

## 🚀 Quick Start on Google Colab

### Option 1: Use the Colab Notebook
1. Open the `eamcet_colab_notebook.ipynb` file in Google Colab
2. Run all cells sequentially
3. Upload your EAMCET PDFs when prompted
4. Wait for the automated pipeline to complete

### Option 2: Manual Setup
```python
# Clone the repository
!git clone https://github.com/your-username/EAMCET.git
%cd EAMCET

# Install dependencies
!pip install -r requirements_colab.txt

# Install system dependencies
!apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# Run the pipeline
!python eamcet_zero_manual_pipeline.py --input_folder data/raw_pdfs --output_folder results
```

## 📁 File Structure
```
EAMCET/
├── eamcet_zero_manual_pipeline.py    # Main pipeline script
├── requirements_colab.txt             # Colab-specific requirements
├── eamcet_colab_notebook.ipynb      # Ready-to-use Colab notebook
├── data/
│   └── raw_pdfs/                    # Upload your PDFs here
└── colab_results/                   # Pipeline outputs
```

## 🎯 Features
- ✅ **Zero Manual Work**: No annotations required
- ✅ **Intelligent Extraction**: Uses pattern recognition
- ✅ **Automatic Training**: Creates training data automatically
- ✅ **Multi-format Support**: Works with any EAMCET PDF format
- ✅ **Subject Classification**: Automatically categorizes by subject

## 📊 Supported Formats
- EAMCET-AP (Andhra Pradesh)
- EAMCET-TG (Telangana)
- MPC (Mathematics, Physics, Chemistry)
- BiPC (Biology, Physics, Chemistry)
- Question papers and answer keys

## 🔧 Requirements
- Google Colab (free)
- EAMCET PDF files
- Internet connection

## 📈 Output
The pipeline generates:
- Extracted questions with options
- Answer keys
- Training datasets
- Model-ready data
- Confidence scores

## 🚨 Important Notes
1. Update the GitHub repository URL in the notebook
2. Ensure your PDFs are in the correct format
3. The pipeline works best with high-quality PDFs
4. Results are automatically downloaded as a zip file

## 📞 Support
If you encounter issues:
1. Check that all dependencies are installed
2. Verify PDF format compatibility
3. Ensure sufficient Colab runtime memory
4. Check the logs for specific error messages

## 🎉 Success Indicators
- Pipeline completes without errors
- Questions are extracted with confidence > 0.7
- Answer keys are properly matched
- Training data is generated
- Results zip file is downloaded

Happy training! 🚀
'''
    
    with open("README_COLAB.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created Colab README: README_COLAB.md")

def main():
    """Main function to create all Colab setup files"""
    print("🔧 Creating Google Colab Setup Files")
    print("=" * 50)
    
    # Create all necessary files
    create_colab_notebook()
    create_requirements_colab()
    create_readme_colab()
    
    print("\n✅ All Colab setup files created!")
    print("\n📋 Next Steps:")
    print("1. Update the GitHub repository URL in the notebook")
    print("2. Upload the notebook to Google Colab")
    print("3. Run the cells sequentially")
    print("4. Upload your EAMCET PDFs when prompted")
    print("5. Wait for the automated pipeline to complete")
    
    print("\n📁 Created Files:")
    print("- eamcet_colab_notebook.ipynb (Ready-to-use Colab notebook)")
    print("- requirements_colab.txt (Colab-specific dependencies)")
    print("- README_COLAB.md (Setup instructions)")

if __name__ == "__main__":
    main() 
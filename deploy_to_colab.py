#!/usr/bin/env python3
"""
Deploy EAMCET Zero Manual Pipeline to Google Colab
This script prepares all files for GitHub deployment and Colab usage
"""

import os
import subprocess
import json
from pathlib import Path

def check_git_status():
    """Check if we're in a git repository and get status"""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def create_gitignore():
    """Create a .gitignore file for the project"""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
eamcet_env/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/processed_images/
data/annotations/
data/augmented/
logs/
models/*.pth
models/*.pt
*.pkl
*.pickle

# Colab specific
colab_results/
test_output/
eamcet_results.zip

# Temporary files
*.tmp
*.temp
'''
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def create_deployment_script():
    """Create a deployment script for easy GitHub push"""
    deploy_script = '''#!/bin/bash
# EAMCET Zero Manual Pipeline - GitHub Deployment Script

echo "🚀 Deploying EAMCET Zero Manual Pipeline to GitHub"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Add EAMCET Zero Manual Pipeline with Colab support

- Added eamcet_zero_manual_pipeline.py
- Added Colab notebook and requirements
- Added deployment scripts
- Updated documentation"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Please add your GitHub repository as remote:"
    echo "   git remote add origin https://github.com/jaganthoutam/EAMCET.git"
    echo "   Then run: git push -u origin main"
else
    echo "🚀 Pushing to GitHub..."
    git push origin main
fi

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps for Google Colab:"
echo "1. Go to https://colab.research.google.com"
echo "2. Upload eamcet_colab_notebook.ipynb"
echo "3. Update the GitHub URL in the notebook"
echo "4. Run all cells sequentially"
echo "5. Upload your EAMCET PDFs when prompted"
'''
    
    with open("deploy.sh", "w") as f:
        f.write(deploy_script)
    
    # Make it executable
    os.chmod("deploy.sh", 0o755)
    
    print("✅ Created deployment script: deploy.sh")

def create_colab_quick_start():
    """Create a quick start guide for Colab"""
    quick_start = '''# EAMCET Zero Manual Pipeline - Quick Start Guide

## 🚀 Google Colab Setup (5 minutes)

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com
2. Click "Upload" and select `eamcet_colab_notebook.ipynb`

### Step 2: Update Repository URL
In the first code cell, update the GitHub URL:
```python
!git clone https://github.com/jaganthoutam/EAMCET.git
```

### Step 3: Run Setup
1. Run the first cell (Setup Environment)
2. Wait for dependencies to install
3. Verify PyMuPDF import works

### Step 4: Upload PDFs
1. Run the second cell (Upload PDFs)
2. Click "Choose Files" when prompted
3. Select your EAMCET PDF files
4. Wait for upload to complete

### Step 5: Run Pipeline
1. Run the third cell (Run Pipeline)
2. Wait for extraction to complete (5-15 minutes)
3. Check the results in the fourth cell
4. Download results in the fifth cell

## 📊 Expected Results

After successful completion, you should see:
- ✅ Questions extracted with confidence scores
- ✅ Answer keys matched automatically
- ✅ Training data generated
- ✅ Results downloaded as zip file

## 🚨 Troubleshooting

### If PyMuPDF fails to import:
```python
!pip uninstall fitz PyMuPDF -y
!pip install PyMuPDF==1.26.3
```

### If Tesseract fails:
```python
!apt-get update
!apt-get install -y tesseract-ocr tesseract-ocr-eng
```

### If pipeline fails:
1. Check PDF format compatibility
2. Ensure sufficient Colab memory
3. Try with fewer PDFs first

## 📞 Support

For issues:
1. Check the logs in Colab output
2. Verify PDF format matches EAMCET standards
3. Ensure all dependencies installed correctly

Happy training! 🎉
'''
    
    with open("COLAB_QUICK_START.md", "w") as f:
        f.write(quick_start)
    
    print("✅ Created quick start guide: COLAB_QUICK_START.md")

def update_main_readme():
    """Update the main README with Colab information"""
    readme_content = '''# EAMCET AI Tutor - Zero Manual Pipeline

A fully automated EAMCET AI tutor training pipeline that requires **zero manual annotations**. Uses intelligent pattern recognition to extract questions and answers from EAMCET PDFs automatically.

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
1. Open `eamcet_colab_notebook.ipynb` in Google Colab
2. Update the GitHub repository URL
3. Run all cells sequentially
4. Upload your EAMCET PDFs when prompted
5. Wait for automated processing

### Option 2: Local Setup
```bash
# Clone repository
git clone https://github.com/your-username/EAMCET.git
cd EAMCET

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python eamcet_zero_manual_pipeline.py --input_folder data/raw_pdfs --output_folder results
```

## 🎯 Features

- ✅ **Zero Manual Work**: No annotations required
- ✅ **Intelligent Extraction**: Uses pattern recognition
- ✅ **Automatic Training**: Creates training data automatically
- ✅ **Multi-format Support**: Works with any EAMCET PDF format
- ✅ **Subject Classification**: Automatically categorizes by subject

## 📊 Supported Formats

- **States**: EAMCET-AP (Andhra Pradesh), EAMCET-TG (Telangana)
- **Streams**: MPC (Mathematics, Physics, Chemistry), BiPC (Biology, Physics, Chemistry)
- **Types**: Question papers, answer keys, solutions

## 📁 Project Structure

```
EAMCET/
├── eamcet_zero_manual_pipeline.py    # Main pipeline script
├── eamcet_colab_notebook.ipynb      # Ready-to-use Colab notebook
├── requirements.txt                   # Python dependencies
├── requirements_colab.txt            # Colab-specific requirements
├── data/
│   └── raw_pdfs/                    # Upload your PDFs here
├── colab_results/                   # Pipeline outputs
└── README_COLAB.md                  # Colab setup instructions
```

## 🔧 Requirements

- Python 3.8+
- PyMuPDF (for PDF processing)
- OpenCV (for image processing)
- Tesseract OCR (for text extraction)
- Google Colab (for cloud processing)

## 📈 Output

The pipeline generates:
- Extracted questions with options
- Answer keys with confidence scores
- Training datasets for AI models
- Subject-wise categorization
- Model-ready data structures

## 🚨 Important Notes

1. **PDF Quality**: Works best with high-quality, text-based PDFs
2. **Format Compatibility**: Designed for standard EAMCET formats
3. **Processing Time**: 5-15 minutes depending on PDF count and quality
4. **Memory Requirements**: Google Colab provides sufficient resources

## 📞 Support

- Check `COLAB_QUICK_START.md` for detailed setup instructions
- Review logs for specific error messages
- Ensure PDF format matches EAMCET standards

## 🎉 Success Indicators

- Pipeline completes without errors
- Questions extracted with confidence > 0.7
- Answer keys properly matched
- Training data generated successfully
- Results downloaded as zip file

## 📄 License

This project is open source and available under the MIT License.

---

**Happy training! 🚀**

*Built with ❤️ for EAMCET students and educators*
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Updated main README.md")

def main():
    """Main deployment function"""
    print("🚀 Preparing EAMCET Zero Manual Pipeline for Google Colab")
    print("=" * 60)
    
    # Check git status
    git_status = check_git_status()
    if git_status:
        print(f"📝 Git status: {len(git_status.split())} files modified")
    else:
        print("⚠️  Not in a git repository")
    
    # Create necessary files
    create_gitignore()
    create_deployment_script()
    create_colab_quick_start()
    update_main_readme()
    
    print("\n✅ All deployment files created!")
    print("\n📋 Deployment Steps:")
    print("1. Run: ./deploy.sh")
    print("2. Add your GitHub repository as remote")
    print("3. Push to GitHub")
    print("4. Upload notebook to Google Colab")
    print("5. Update repository URL in notebook")
    print("6. Run the pipeline!")
    
    print("\n📁 Created Files:")
    print("- .gitignore (Git ignore rules)")
    print("- deploy.sh (Deployment script)")
    print("- COLAB_QUICK_START.md (Quick start guide)")
    print("- Updated README.md (Main documentation)")
    
    print("\n🎯 Ready for Google Colab deployment!")

if __name__ == "__main__":
    main() 
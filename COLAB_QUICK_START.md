# EAMCET Zero Manual Pipeline - Quick Start Guide

## 🚀 Google Colab Setup (5 minutes)

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com
2. Click "Upload" and select `eamcet_colab_notebook.ipynb`

### Step 2: Update Repository URL
In the first code cell, update the GitHub URL:
```python
!git clone https://github.com/YOUR_USERNAME/eamcet_ai_tutor.git
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

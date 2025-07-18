# Google Colab Troubleshooting Guide

## 🚨 Common Issues and Solutions

### 1. **AttributeError: 'list' object has no attribute 'download'**

**Problem:** The download cell fails with this error.

**Solution:** 
- ✅ **Fixed in the updated notebook** - The download cell has been corrected
- If it still fails, use the standalone download script:
  ```python
  # Copy and paste this into a new cell
  from google.colab import files
  import zipfile
  import os
  
  # Create zip file
  if os.path.exists("colab_results"):
      with zipfile.ZipFile("eamcet_results.zip", "w") as zipf:
          for root, dirs, files_list in os.walk("colab_results"):
              for file in files_list:
                  file_path = os.path.join(root, file)
                  arcname = os.path.relpath(file_path, "colab_results")
                  zipf.write(file_path, arcname)
      
      files.download("eamcet_results.zip")
      print("✅ Download successful!")
  else:
      print("❌ No results found")
  ```

### 2. **PyMuPDF Import Error**

**Problem:** `ModuleNotFoundError: No module named 'frontend'`

**Solution:**
```python
# Run this in a new cell
!pip uninstall fitz PyMuPDF -y
!pip install PyMuPDF==1.26.3
import fitz
print("✅ PyMuPDF fixed!")
```

### 3. **Tesseract OCR Not Found**

**Problem:** OCR fails to work

**Solution:**
```python
# Run this in a new cell
!apt-get update
!apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev
print("✅ Tesseract installed!")
```

### 4. **Memory Issues**

**Problem:** Colab runs out of memory

**Solution:**
- Restart runtime: Runtime → Restart runtime
- Use GPU runtime: Runtime → Change runtime type → GPU
- Process fewer PDFs at once

### 5. **Git Clone Fails**

**Problem:** Repository not found

**Solution:**
- Update the GitHub URL in the first cell
- Replace `YOUR_USERNAME` with your actual GitHub username
- Make sure the repository is public or you have access

### 6. **Pipeline Takes Too Long**

**Problem:** Processing is slow

**Solution:**
- Use GPU runtime for faster processing
- Process fewer PDFs at once
- Check PDF quality (text-based PDFs work better)

### 7. **No Results Generated**

**Problem:** Pipeline runs but no output

**Solution:**
- Check if PDFs are in the correct format
- Verify PDFs are uploaded to `data/raw_pdfs/`
- Look for error messages in the pipeline output
- Try with a single PDF first

### 8. **Download Fails**

**Problem:** Can't download results

**Solution:**
- Use the file browser: Click the folder icon on the left
- Navigate to `colab_results/` folder
- Right-click files and select "Download"
- Or use the standalone download script

## 🔧 Quick Fixes

### Reset Environment
```python
# Run this to reset everything
import os
import shutil

# Clean up
if os.path.exists("eamcet_ai_tutor"):
    shutil.rmtree("eamcet_ai_tutor")
if os.path.exists("colab_results"):
    shutil.rmtree("colab_results")

print("✅ Environment reset!")
```

### Check Installation
```python
# Verify all dependencies
import fitz
import cv2
import pytesseract
import torch
import transformers

print("✅ All dependencies working!")
```

### Manual Download
```python
# If automatic download fails, list files manually
import os

if os.path.exists("colab_results"):
    print("📁 Available files:")
    for root, dirs, files in os.walk("colab_results"):
        for file in files:
            print(f"  📄 {os.path.join(root, file)}")
else:
    print("❌ No results found")
```

## 📞 Getting Help

1. **Check the logs** - Look for error messages in cell outputs
2. **Restart runtime** - Runtime → Restart runtime
3. **Use GPU** - Runtime → Change runtime type → GPU
4. **Check file browser** - Look for your files in the left panel
5. **Try with fewer PDFs** - Start with 1-2 PDFs to test

## 🎯 Success Checklist

- ✅ PyMuPDF imports without errors
- ✅ Tesseract OCR is installed
- ✅ PDFs are uploaded to `data/raw_pdfs/`
- ✅ Pipeline completes without errors
- ✅ Results are in `colab_results/` folder
- ✅ Files can be downloaded

## 🚀 Pro Tips

1. **Use GPU runtime** for faster processing
2. **Upload high-quality PDFs** for better extraction
3. **Start with a few PDFs** to test the pipeline
4. **Check the file browser** for your results
5. **Save your notebook** to Google Drive for backup

Happy troubleshooting! 🎉 
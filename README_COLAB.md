# EAMCET Zero Manual Pipeline - Google Colab Setup

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
!git clone https://github.com/your-username/eamcet_ai_tutor.git
%cd eamcet_ai_tutor

# Install dependencies
!pip install -r requirements_colab.txt

# Install system dependencies
!apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# Run the pipeline
!python eamcet_zero_manual_pipeline.py --input_folder data/raw_pdfs --output_folder results
```

## 📁 File Structure
```
eamcet_ai_tutor/
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

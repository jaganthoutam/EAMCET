# EAMCET AI Tutor - Zero Manual Pipeline

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

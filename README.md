# EAMCET AI Tutor Training Pipeline

This project trains custom AI models for EAMCET question extraction and tutoring.

## Quick Start

1. **Setup Environment**
   ```bash
   # Run the setup script (only once)
   ./setup.sh
   
   # Activate virtual environment
   source eamcet_env/bin/activate
   ```

2. **Prepare Your Data**
   - Copy your EAMCET PDFs to a folder
   - Organize them in this structure:
   ```
   your_pdfs/
   ├── EAMCET-AP/
   │   ├── MPC/
   │   └── BiPC/
   └── EAMCET-TG/
       ├── MPC/
       └── BiPC/
   ```

3. **Extract Data from PDFs**
   ```bash
   cd eamcet_ai_tutor
   python train.py --data_path /path/to/your/pdfs --stage extract
   ```

4. **Review Extracted Data**
   - Check the `processed_data/` folder
   - Review `processing_report.txt` for statistics
   - Examine extracted images and text

## Project Structure

```
eamcet_ai_tutor/
├── config.py              # Configuration settings
├── train.py               # Main training script
├── eamcet_training_starter.py  # PDF processing pipeline
├── requirements.txt       # Python dependencies
├── data/                  # Data storage
│   ├── raw_pdfs/         # Original PDFs
│   ├── processed_images/ # Extracted page images
│   ├── annotations/      # Manual annotations
│   └── augmented/        # Augmented training data
├── models/               # Trained models
├── training/             # Training scripts
├── inference/            # Inference pipeline
└── logs/                # Training logs
```

## Features

- ✅ Multi-state support (AP & TG)
- ✅ Multi-stream support (MPC & BiPC)
- ✅ Automatic PDF scanning and organization
- ✅ High-resolution image extraction
- ✅ OCR text extraction
- ✅ Question pattern detection
- ✅ Answer key format detection
- ✅ Color-coded answer detection
- 🔄 Custom model training (coming next)
- 🔄 AI tutoring system (coming next)

## Next Steps

1. Run data extraction on your PDFs
2. Review and verify extracted data
3. Create manual annotations for training
4. Train custom models
5. Build the tutoring application

## Support

For issues or questions, check the logs in `processed_data/logs/` or review the processing report.

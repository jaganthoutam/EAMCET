# 🎓 EAMCET AI Tutor - Complete Training Guide

## 📋 Overview

This guide provides step-by-step instructions to train the EAMCET AI tutor models using real EAMCET data in Google Colab.

## 🚀 Quick Start (Google Colab)

### Step 1: Setup Colab Environment

```python
# Install required packages
!pip install transformers torch torchvision torchaudio
!pip install pandas numpy matplotlib seaborn
!pip install scikit-learn nltk
!pip install tqdm
!pip install PyMuPDF opencv-python pytesseract
!pip install albumentations fastapi uvicorn
```

```python
# Clone your repository
!git clone https://github.com/your-username/eamcet_ai_tutor.git
%cd eamcet_ai_tutor
```

### Step 2: Upload EAMCET PDFs

1. **Upload your EAMCET PDFs** to the `data/raw_pdfs/` folder
2. **Organize by state and stream**:
   ```
   data/raw_pdfs/
   ├── EAMCET-AP/
   │   ├── MPC/
   │   │   ├── question-papers/
   │   │   └── answer-keys/
   │   └── BiPC/
   │       ├── question-papers/
   │       └── answer-keys/
   └── EAMCET-TG/
       ├── MPC/
       └── BiPC/
   ```

### Step 3: Run Data Extraction

```python
# Extract questions and answers from PDFs
!python eamcet_zero_manual_pipeline.py --input_folder data/raw_pdfs --output_folder eamcet_results
```

### Step 4: Train Models

```python
# Train subject classification model
!python eamcet_model_trainer.py --model_type subject_classification --epochs 10 --batch_size 16

# Train answer prediction model  
!python eamcet_model_trainer.py --model_type answer_prediction --epochs 10 --batch_size 16

# Train question parsing model
!python eamcet_model_trainer.py --model_type question_parsing --epochs 10 --batch_size 16
```

### Step 5: Download Trained Models

```python
# Download the trained models
from google.colab import files
import zipfile

!zip -r trained_models.zip trained_models/
files.download('trained_models.zip')
```

## 🔧 Manual Answer Key Processing

Since the current PDFs have embedded answer keys as visual elements, you may need to manually extract answers:

### Option 1: Manual Answer Extraction

1. **Open answer key PDFs** in a PDF viewer
2. **Look for green/red colored options** in each question
3. **Create a CSV file** with format:
   ```csv
   question_number,correct_answer
   1,A
   2,B
   3,C
   ...
   ```

### Option 2: Use OCR with Color Detection

```python
# Enhanced answer extraction script
import cv2
import numpy as np
import fitz
from PIL import Image

def extract_answers_with_color_detection(pdf_path):
    """Extract answers using color detection"""
    doc = fitz.open(pdf_path)
    answers = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to image
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define green and red color ranges
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        red_lower = np.array([0, 40, 40])
        red_upper = np.array([25, 255, 255])
        
        # Create masks
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        
        # Find contours of colored regions
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process green contours (correct answers)
        for contour in green_contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                # Map position to option (A, B, C, D)
                # This requires knowing the layout of options
                pass
    
    doc.close()
    return answers
```

## 📊 Expected Results

After successful training, you should see:

```
📊 EXTRACTION RESULTS:
   Total questions extracted: 8000+
   Questions with answers: 8000+
   Answer keys processed: 12+

📚 SUBJECT BREAKDOWN:
   Mathematics: 4000+ questions
   Physics: 2000+ questions  
   Chemistry: 2000+ questions

🎯 TRAINING DATA CREATED:
   text_detection: 15000+ samples
   question_parsing: 8000+ samples
   answer_detection: 8000+ samples
```

## 🎯 Model Training Details

### Subject Classification Model
- **Input**: Question text
- **Output**: Subject (Mathematics, Physics, Chemistry, Biology)
- **Architecture**: BERT-based classifier
- **Training Data**: 8000+ labeled questions

### Answer Prediction Model
- **Input**: Question text + options
- **Output**: Correct answer (A, B, C, D)
- **Architecture**: BERT-based sequence classifier
- **Training Data**: 8000+ question-answer pairs

### Question Parsing Model
- **Input**: Raw question text
- **Output**: Structured question data
- **Architecture**: BERT-based sequence labeling
- **Training Data**: 8000+ parsed questions

## 🔍 Troubleshooting

### Issue: No answers extracted
**Solution**: 
1. Check if PDFs contain answer keys
2. Use manual answer extraction
3. Verify color detection settings

### Issue: Low question extraction
**Solution**:
1. Check PDF quality and resolution
2. Verify OCR settings
3. Adjust confidence thresholds

### Issue: Training fails
**Solution**:
1. Ensure sufficient GPU memory
2. Reduce batch size
3. Use smaller model variants

## 📁 File Structure

```
eamcet_ai_tutor/
├── data/raw_pdfs/          # Upload your PDFs here
├── eamcet_results/         # Extracted data
├── trained_models/         # Trained models
├── eamcet_zero_manual_pipeline.py
├── eamcet_model_trainer.py
├── eamcet_inference.py
└── requirements.txt
```

## 🚀 Next Steps

1. **Upload your EAMCET PDFs** to the data folder
2. **Run the extraction pipeline** to process all PDFs
3. **Train the models** using the extracted data
4. **Download trained models** for local use
5. **Test the inference** with real questions

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify PDF format and quality
3. Ensure sufficient Colab resources
4. Contact for additional support

---

**🎓 Ready to build your EAMCET AI tutor!** 
# 🎓 EAMCET AI Tutor - Complete Training Guide

## 📋 Overview

This guide provides step-by-step instructions to train the EAMCET AI tutor models using real EAMCET data in Google Colab.

## ⚠️ Important Note: Answer Extraction Limitation

**Current Challenge**: EAMCET answer keys use visual elements (green checkmarks ✓ and red X marks ✗) that are embedded as images rather than text. This makes automated extraction very difficult.

**Solution**: Use the automated question extraction pipeline and manually extract answers from answer key PDFs.

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
!git clone https://github.com/jaganthoutam/EAMCET.git
%cd EAMCET
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

**Expected Output**: 
- ✅ Questions extracted successfully (8000+ questions)
- ⚠️ Answers extracted: 0 (due to visual elements)

### Step 4: Manual Answer Extraction

Since automated answer extraction is limited, follow these steps:

#### Option A: Manual Answer Key Creation

1. **Open answer key PDFs** in a PDF viewer
2. **Look for green checkmarks ✓** (correct answers) and **red X marks ✗** (incorrect answers)
3. **Create a CSV file** with format:
   ```csv
   question_number,correct_answer
   1,A
   2,B
   3,C
   4,D
   5,A
   ...
   ```

#### Option B: Enhanced Answer Extraction Script

```python
# Create enhanced answer extraction script
import pandas as pd
import fitz
import cv2
import numpy as np

def extract_answers_manual(pdf_path, output_csv):
    """Manual answer extraction with visual guidance"""
    doc = fitz.open(pdf_path)
    answers = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"Processing page {page_num + 1}")
        
        # Convert page to image for visual analysis
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.samples
        img = np.frombuffer(img_data, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Save image for manual inspection
        cv2.imwrite(f"page_{page_num + 1}.png", img)
        
        # Extract text for question numbers
        text = page.get_text()
        
        # Find question numbers and their positions
        import re
        question_matches = re.finditer(r'(\d+)', text)
        
        for match in question_matches:
            question_num = int(match.group(1))
            if 1 <= question_num <= 160:  # Valid EAMCET question range
                print(f"Question {question_num}: Look for green checkmark ✓ or red X ✗")
                # Manual inspection required for each question
                # You can enhance this with image processing
    
    doc.close()
    
    # Create CSV template
    df = pd.DataFrame(answers, columns=['question_number', 'correct_answer'])
    df.to_csv(output_csv, index=False)
    print(f"Answer template saved to {output_csv}")
    print("Please manually fill in the correct answers based on visual inspection")

# Usage
extract_answers_manual("answer_key.pdf", "manual_answers.csv")
```

### Step 5: Train Models with Manual Answers

```python
# Load manual answers
import pandas as pd
manual_answers = pd.read_csv("manual_answers.csv")

# Create training data with manual answers
training_data = {
    'questions': extracted_questions,  # From pipeline
    'answers': dict(zip(manual_answers['question_number'], manual_answers['correct_answer']))
}

# Train models
!python eamcet_model_trainer.py --training_data training_data.pkl --epochs 10
```

### Step 6: Download Trained Models

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
   Questions with answers: 8000+ (with manual extraction)
   Answer keys processed: 12+

📚 SUBJECT BREAKDOWN:
   Mathematics: 4000+ questions
   Physics: 2000+ questions  
   Chemistry: 2000+ questions

🎯 TRAINING DATA CREATED:
   text_detection: 15000+ samples
   question_parsing: 8000+ samples
   answer_detection: 8000+ samples (with manual answers)
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
- **Training Data**: 8000+ question-answer pairs (with manual answers)

### Question Parsing Model
- **Input**: Raw question text
- **Output**: Structured question data
- **Architecture**: BERT-based sequence labeling
- **Training Data**: 8000+ parsed questions

## 🔍 Troubleshooting

### Issue: No answers extracted
**Solution**: 
1. ✅ This is expected - answers are visual elements
2. Use manual answer extraction process above
3. Create CSV file with question_number,correct_answer format

### Issue: Low question extraction
**Solution**:
1. Check PDF quality and resolution
2. Verify OCR settings
3. Ensure PDFs are not password protected

### Issue: Training fails due to insufficient data
**Solution**:
1. Complete manual answer extraction first
2. Ensure you have at least 1000 question-answer pairs
3. Use the enhanced training script with manual data

## 📋 Next Steps

1. **Run the pipeline** to extract questions
2. **Manually extract answers** from answer key PDFs
3. **Create training dataset** with manual answers
4. **Train models** using the complete dataset
5. **Test and deploy** the AI tutor application

---

**Note**: The automated question extraction works well, but answer extraction requires manual processing due to the visual nature of EAMCET answer keys. 
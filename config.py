"""
EAMCET AI Tutor Configuration
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
LOGS_ROOT = PROJECT_ROOT / "logs"

# Data paths
RAW_PDFS_PATH = DATA_ROOT / "raw_pdfs"
PROCESSED_IMAGES_PATH = DATA_ROOT / "processed_images"
ANNOTATIONS_PATH = DATA_ROOT / "annotations"
AUGMENTED_DATA_PATH = DATA_ROOT / "augmented"

# Model paths
TEXT_DETECTION_MODEL_PATH = MODELS_ROOT / "text_detection"
TEXT_RECOGNITION_MODEL_PATH = MODELS_ROOT / "text_recognition"
QUESTION_PARSING_MODEL_PATH = MODELS_ROOT / "question_parsing"
ANSWER_DETECTION_MODEL_PATH = MODELS_ROOT / "answer_detection"

# EAMCET Configuration
EAMCET_CONFIG = {
    'states': ['EAMCET-AP', 'EAMCET-TG'],
    'streams': {
        'MPC': {
            'subjects': ['Mathematics', 'Physics', 'Chemistry'],
            'question_distribution': {'Mathematics': 80, 'Physics': 40, 'Chemistry': 40}
        },
        'BiPC': {
            'subjects': ['Biology', 'Physics', 'Chemistry'],
            'question_distribution': {'Biology': 80, 'Physics': 40, 'Chemistry': 40}
        }
    },
    'total_questions': 160,
    'total_marks': 160,
    'duration_minutes': 180
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 5e-5,
    'epochs': 100,
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_seed': 42,
    'device': 'cuda' if os.environ.get('CUDA_AVAILABLE') else 'cpu',
    'mixed_precision': True,
    'gradient_accumulation_steps': 4
}

# OCR Configuration
OCR_CONFIG = {
    'image_dpi': 300,
    'confidence_threshold': 30,
    'tesseract_config': '--oem 3 --psm 6 -c preserve_interword_spaces=1',
    'supported_languages': ['eng', 'tel']  # English and Telugu
}

# Answer Detection Configuration
ANSWER_DETECTION_CONFIG = {
    'color_ranges': {
        'green_correct': {
            'lower_hsv': [40, 50, 50],
            'upper_hsv': [80, 255, 255]
        },
        'red_incorrect': {
            'lower_hsv': [0, 50, 50],
            'upper_hsv': [20, 255, 255]
        }
    },
    'color_threshold': 0.05,  # 5% of region must be colored
    'icon_confidence_threshold': 0.6
}

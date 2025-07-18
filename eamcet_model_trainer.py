#!/usr/bin/env python3
"""
EAMCET Model Trainer - Complete Training Pipeline
Trains actual neural networks for EAMCET AI Tutor application
"""

import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for EAMCET model training"""
    
    # Model configurations
    TEXT_DETECTION_MODEL = "microsoft/layoutlm-base-uncased"
    QUESTION_PARSING_MODEL = "distilbert-base-uncased"
    ANSWER_DETECTION_MODEL = "distilbert-base-uncased"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    MAX_LENGTH = 512
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # EAMCET specific
    SUBJECTS = ['Mathematics', 'Physics', 'Chemistry', 'Biology']
    ANSWER_OPTIONS = ['A', 'B', 'C', 'D']
    
    # Output paths
    MODEL_OUTPUT_DIR = "trained_models"
    LOGS_DIR = "training_logs"

class EAMCETDataset(Dataset):
    """Custom dataset for EAMCET training data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EAMCETModelTrainer:
    """Complete trainer for EAMCET AI models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config.MODEL_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path(config.LOGS_DIR)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Training history
        self.training_history = {
            'subject_classification': {},
            'answer_prediction': {},
            'question_parsing': {}
        }
    
    def prepare_subject_classification_data(self, training_data: Dict) -> Tuple[List, List]:
        """Prepare data for subject classification model"""
        logger.info("Preparing subject classification data...")
        
        texts = []
        labels = []
        
        for question in training_data.get('question_parsing', []):
            if 'structured_output' in question:
                question_text = question['structured_output'].get('question_text', '')
                subject = question['structured_output'].get('subject', 'Mathematics')
                
                if question_text and subject in self.config.SUBJECTS:
                    texts.append(question_text)
                    labels.append(self.config.SUBJECTS.index(subject))
        
        logger.info(f"Prepared {len(texts)} samples for subject classification")
        return texts, labels
    
    def prepare_answer_prediction_data(self, training_data: Dict) -> Tuple[List, List]:
        """Prepare data for answer prediction model"""
        logger.info("Preparing answer prediction data...")
        
        texts = []
        labels = []
        
        for question in training_data.get('question_parsing', []):
            if 'structured_output' in question:
                question_text = question['structured_output'].get('question_text', '')
                correct_answer = question['structured_output'].get('correct_answer', '')
                
                if question_text and correct_answer in self.config.ANSWER_OPTIONS:
                    texts.append(question_text)
                    labels.append(self.config.ANSWER_OPTIONS.index(correct_answer))
        
        logger.info(f"Prepared {len(texts)} samples for answer prediction")
        return texts, labels
    
    def train_subject_classification_model(self, training_data: Dict) -> Dict:
        """Train model to classify questions by subject"""
        logger.info("🚀 Training Subject Classification Model...")
        
        # Prepare data
        texts, labels = self.prepare_subject_classification_data(training_data)
        
        if len(texts) < 10:
            logger.warning("Insufficient data for subject classification training")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.QUESTION_PARSING_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.QUESTION_PARSING_MODEL,
            num_labels=len(self.config.SUBJECTS)
        )
        
        # Create datasets
        train_dataset = EAMCETDataset(train_texts, train_labels, tokenizer)
        val_dataset = EAMCETDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "subject_classification"),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            logging_dir=str(self.logs_dir / "subject_classification"),
            logging_steps=10,
            save_total_limit=3,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        # Train model
        try:
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Save model
            model_path = self.output_dir / "subject_classification"
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save training history
            self.training_history['subject_classification'] = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'model_path': str(model_path)
            }
            
            logger.info(f"✅ Subject classification model saved to {model_path}")
            logger.info(f"   Final accuracy: {eval_result['eval_accuracy']:.4f}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'accuracy': eval_result['eval_accuracy'],
                'eval_loss': eval_result['eval_loss']
            }
            
        except Exception as e:
            logger.error(f"❌ Error training subject classification model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_answer_prediction_model(self, training_data: Dict) -> Dict:
        """Train model to predict correct answers"""
        logger.info("🚀 Training Answer Prediction Model...")
        
        # Prepare data
        texts, labels = self.prepare_answer_prediction_data(training_data)
        
        if len(texts) < 10:
            logger.warning("Insufficient data for answer prediction training")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.ANSWER_DETECTION_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.ANSWER_DETECTION_MODEL,
            num_labels=len(self.config.ANSWER_OPTIONS)
        )
        
        # Create datasets
        train_dataset = EAMCETDataset(train_texts, train_labels, tokenizer)
        val_dataset = EAMCETDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "answer_prediction"),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            logging_dir=str(self.logs_dir / "answer_prediction"),
            logging_steps=10,
            save_total_limit=3,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        # Train model
        try:
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Save model
            model_path = self.output_dir / "answer_prediction"
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save training history
            self.training_history['answer_prediction'] = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'model_path': str(model_path)
            }
            
            logger.info(f"✅ Answer prediction model saved to {model_path}")
            logger.info(f"   Final accuracy: {eval_result['eval_accuracy']:.4f}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'accuracy': eval_result['eval_accuracy'],
                'eval_loss': eval_result['eval_loss']
            }
            
        except Exception as e:
            logger.error(f"❌ Error training answer prediction model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_question_parsing_model(self, training_data: Dict) -> Dict:
        """Train model for question parsing (sequence-to-sequence)"""
        logger.info("🚀 Training Question Parsing Model...")
        
        # For now, we'll use a simpler approach with BERT for question parsing
        # In a full implementation, you might use T5 or BART for sequence-to-sequence
        
        texts = []
        labels = []  # We'll use subject classification as a proxy
        
        for question in training_data.get('question_parsing', []):
            if 'input_text' in question and 'structured_output' in question:
                input_text = question['input_text']
                subject = question['structured_output'].get('subject', 'Mathematics')
                
                if input_text and subject in self.config.SUBJECTS:
                    texts.append(input_text)
                    labels.append(self.config.SUBJECTS.index(subject))
        
        if len(texts) < 10:
            logger.warning("Insufficient data for question parsing training")
            return {'success': False, 'error': 'Insufficient data'}
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.QUESTION_PARSING_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.QUESTION_PARSING_MODEL,
            num_labels=len(self.config.SUBJECTS)
        )
        
        # Create datasets
        train_dataset = EAMCETDataset(train_texts, train_labels, tokenizer)
        val_dataset = EAMCETDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "question_parsing"),
            num_train_epochs=self.config.NUM_EPOCHS,
            per_device_train_batch_size=self.config.BATCH_SIZE,
            per_device_eval_batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            warmup_steps=self.config.WARMUP_STEPS,
            weight_decay=self.config.WEIGHT_DECAY,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            logging_dir=str(self.logs_dir / "question_parsing"),
            logging_steps=10,
            save_total_limit=3,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
        )
        
        # Train model
        try:
            train_result = trainer.train()
            eval_result = trainer.evaluate()
            
            # Save model
            model_path = self.output_dir / "question_parsing"
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Save training history
            self.training_history['question_parsing'] = {
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result['eval_loss'],
                'eval_accuracy': eval_result['eval_accuracy'],
                'model_path': str(model_path)
            }
            
            logger.info(f"✅ Question parsing model saved to {model_path}")
            logger.info(f"   Final accuracy: {eval_result['eval_accuracy']:.4f}")
            
            return {
                'success': True,
                'model_path': str(model_path),
                'accuracy': eval_result['eval_accuracy'],
                'eval_loss': eval_result['eval_loss']
            }
            
        except Exception as e:
            logger.error(f"❌ Error training question parsing model: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def train_all_models(self, training_data: Dict) -> Dict:
        """Train all EAMCET models"""
        logger.info("🤖 Starting Complete EAMCET Model Training...")
        
        results = {
            'subject_classification': {},
            'answer_prediction': {},
            'question_parsing': {},
            'overall_success': True
        }
        
        # Train subject classification model
        results['subject_classification'] = self.train_subject_classification_model(training_data)
        
        # Train answer prediction model
        results['answer_prediction'] = self.train_answer_prediction_model(training_data)
        
        # Train question parsing model
        results['question_parsing'] = self.train_question_parsing_model(training_data)
        
        # Check overall success
        for model_name, result in results.items():
            if model_name != 'overall_success' and not result.get('success', False):
                results['overall_success'] = False
        
        # Save training summary
        self.save_training_summary(results)
        
        return results
    
    def save_training_summary(self, results: Dict):
        """Save comprehensive training summary"""
        summary = {
            'training_date': datetime.now().isoformat(),
            'config': {
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'num_epochs': self.config.NUM_EPOCHS,
                'subjects': self.config.SUBJECTS,
                'answer_options': self.config.ANSWER_OPTIONS
            },
            'results': results,
            'training_history': self.training_history
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Training summary saved to {summary_path}")
    
    def create_training_visualizations(self, results: Dict):
        """Create training visualizations"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Model accuracy comparison
            model_names = []
            accuracies = []
            
            for model_name, result in results.items():
                if model_name != 'overall_success' and result.get('success'):
                    model_names.append(model_name.replace('_', ' ').title())
                    accuracies.append(result.get('accuracy', 0))
            
            if accuracies:
                axes[0, 0].bar(model_names, accuracies, color=['#ff9999', '#66b3ff', '#99ff99'])
                axes[0, 0].set_title('Model Accuracy Comparison')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].set_ylim(0, 1)
                for i, v in enumerate(accuracies):
                    axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            # Training losses
            losses = []
            for model_name, result in results.items():
                if model_name != 'overall_success' and result.get('success'):
                    losses.append(result.get('eval_loss', 0))
            
            if losses:
                axes[0, 1].bar(model_names, losses, color=['#ffcc99', '#cc99ff', '#99ccff'])
                axes[0, 1].set_title('Model Evaluation Loss')
                axes[0, 1].set_ylabel('Loss')
            
            # Success rate
            successful_models = sum(1 for r in results.values() if r.get('success', False))
            total_models = len([k for k in results.keys() if k != 'overall_success'])
            
            axes[1, 0].pie([successful_models, total_models - successful_models], 
                          labels=['Successful', 'Failed'], 
                          colors=['#66b3ff', '#ff9999'], autopct='%1.1f%%')
            axes[1, 0].set_title('Training Success Rate')
            
            # Model status
            status_data = []
            status_labels = []
            for model_name, result in results.items():
                if model_name != 'overall_success':
                    status_data.append(1 if result.get('success') else 0)
                    status_labels.append(model_name.replace('_', ' ').title())
            
            if status_data:
                axes[1, 1].bar(status_labels, status_data, color=['green' if x else 'red' for x in status_data])
                axes[1, 1].set_title('Model Training Status')
                axes[1, 1].set_ylabel('Status (1=Success, 0=Failed)')
                axes[1, 1].set_ylim(0, 1)
            
            plt.tight_layout()
            
            viz_path = self.output_dir / "training_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Training visualization saved to {viz_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")

def main():
    """Main function for model training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EAMCET Model Trainer')
    parser.add_argument('--training_data', required=True, help='Path to training data pickle file')
    parser.add_argument('--output_dir', default='trained_models', help='Output directory for models')
    parser.add_argument('--config', help='Path to custom config file')
    
    args = parser.parse_args()
    
    print("🚀 EAMCET Model Trainer")
    print("=" * 40)
    
    # Load training data
    if not os.path.exists(args.training_data):
        print(f"❌ Training data file not found: {args.training_data}")
        return
    
    try:
        with open(args.training_data, 'rb') as f:
            training_data = pickle.load(f)
        print(f"✅ Loaded training data from {args.training_data}")
        
    except Exception as e:
        print(f"❌ Error loading training data: {str(e)}")
        return
    
    # Initialize trainer
    config = TrainingConfig()
    if args.output_dir:
        config.MODEL_OUTPUT_DIR = args.output_dir
    
    trainer = EAMCETModelTrainer(config)
    
    # Train all models
    print("🤖 Starting model training...")
    results = trainer.train_all_models(training_data)
    
    # Create visualizations
    trainer.create_training_visualizations(results)
    
    # Print summary
    print("\n📊 Training Summary:")
    print("=" * 40)
    
    for model_name, result in results.items():
        if model_name != 'overall_success':
            status = "✅ Success" if result.get('success') else "❌ Failed"
            accuracy = f"{result.get('accuracy', 0):.2%}" if result.get('accuracy') else "N/A"
            print(f"   {model_name.replace('_', ' ').title()}: {status} (Accuracy: {accuracy})")
    
    overall_status = "✅ All models trained successfully!" if results['overall_success'] else "❌ Some models failed to train"
    print(f"\n{overall_status}")
    
    if results['overall_success']:
        print(f"📁 Models saved to: {args.output_dir}")
        print("🚀 Ready for EAMCET AI Tutor application!")

if __name__ == "__main__":
    main()

# Usage:
# python eamcet_model_trainer.py --training_data training_data.pkl --output_dir trained_models 
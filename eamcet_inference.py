#!/usr/bin/env python3
"""
EAMCET AI Tutor - Inference Module
Provides intelligent tutoring using trained models
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for EAMCET inference"""
    
    # Model paths
    SUBJECT_CLASSIFICATION_MODEL = "trained_models/subject_classification"
    ANSWER_PREDICTION_MODEL = "trained_models/answer_prediction"
    QUESTION_PARSING_MODEL = "trained_models/question_parsing"
    
    # EAMCET specific
    SUBJECTS = ['Mathematics', 'Physics', 'Chemistry', 'Biology']
    ANSWER_OPTIONS = ['A', 'B', 'C', 'D']
    
    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.7
    MAX_LENGTH = 512

class EAMCETTutor:
    """EAMCET AI Tutor with inference capabilities"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.models = {}
        self.tokenizers = {}
        self.load_models()
        
        # Initialize pipelines
        self.subject_classifier = None
        self.answer_predictor = None
        self.question_parser = None
        self.initialize_pipelines()
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        model_configs = {
            'subject_classification': {
                'path': self.config.SUBJECT_CLASSIFICATION_MODEL,
                'num_labels': len(self.config.SUBJECTS)
            },
            'answer_prediction': {
                'path': self.config.ANSWER_PREDICTION_MODEL,
                'num_labels': len(self.config.ANSWER_OPTIONS)
            },
            'question_parsing': {
                'path': self.config.QUESTION_PARSING_MODEL,
                'num_labels': len(self.config.SUBJECTS)
            }
        }
        
        for model_name, config in model_configs.items():
            model_path = Path(config['path'])
            
            if model_path.exists():
                try:
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    self.tokenizers[model_name] = tokenizer
                    
                    # Load model
                    model = AutoModelForSequenceClassification.from_pretrained(
                        str(model_path),
                        num_labels=config['num_labels']
                    )
                    model.to(self.device)
                    model.eval()
                    self.models[model_name] = model
                    
                    logger.info(f"✅ Loaded {model_name} model")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {model_name} model: {str(e)}")
            else:
                logger.warning(f"⚠️  {model_name} model not found at {model_path}")
    
    def initialize_pipelines(self):
        """Initialize inference pipelines"""
        try:
            if 'subject_classification' in self.models:
                self.subject_classifier = pipeline(
                    "text-classification",
                    model=self.models['subject_classification'],
                    tokenizer=self.tokenizers['subject_classification'],
                    device=0 if torch.cuda.is_available() else -1
                )
            
            if 'answer_prediction' in self.models:
                self.answer_predictor = pipeline(
                    "text-classification",
                    model=self.models['answer_prediction'],
                    tokenizer=self.tokenizers['answer_prediction'],
                    device=0 if torch.cuda.is_available() else -1
                )
            
            logger.info("✅ Inference pipelines initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing pipelines: {str(e)}")
    
    def classify_subject(self, question_text: str) -> Dict[str, Any]:
        """Classify question by subject"""
        if not self.subject_classifier:
            return {'subject': 'Unknown', 'confidence': 0.0, 'error': 'Model not loaded'}
        
        try:
            result = self.subject_classifier(question_text)
            
            if result and len(result) > 0:
                prediction = result[0]
                subject_idx = int(prediction['label'].split('_')[-1])
                subject = self.config.SUBJECTS[subject_idx]
                confidence = prediction['score']
                
                return {
                    'subject': subject,
                    'confidence': confidence,
                    'all_predictions': result
                }
            else:
                return {'subject': 'Unknown', 'confidence': 0.0, 'error': 'No prediction'}
                
        except Exception as e:
            logger.error(f"Error in subject classification: {str(e)}")
            return {'subject': 'Unknown', 'confidence': 0.0, 'error': str(e)}
    
    def predict_answer(self, question_text: str) -> Dict[str, Any]:
        """Predict the most likely correct answer"""
        if not self.answer_predictor:
            return {'predicted_answer': 'Unknown', 'confidence': 0.0, 'error': 'Model not loaded'}
        
        try:
            result = self.answer_predictor(question_text)
            
            if result and len(result) > 0:
                prediction = result[0]
                answer_idx = int(prediction['label'].split('_')[-1])
                predicted_answer = self.config.ANSWER_OPTIONS[answer_idx]
                confidence = prediction['score']
                
                # Get all predictions for analysis
                all_predictions = []
                for pred in result:
                    idx = int(pred['label'].split('_')[-1])
                    all_predictions.append({
                        'option': self.config.ANSWER_OPTIONS[idx],
                        'confidence': pred['score']
                    })
                
                return {
                    'predicted_answer': predicted_answer,
                    'confidence': confidence,
                    'all_predictions': all_predictions,
                    'reasoning': self.generate_reasoning(question_text, predicted_answer, confidence)
                }
            else:
                return {'predicted_answer': 'Unknown', 'confidence': 0.0, 'error': 'No prediction'}
                
        except Exception as e:
            logger.error(f"Error in answer prediction: {str(e)}")
            return {'predicted_answer': 'Unknown', 'confidence': 0.0, 'error': str(e)}
    
    def generate_reasoning(self, question_text: str, predicted_answer: str, confidence: float) -> str:
        """Generate reasoning for the predicted answer"""
        subject_result = self.classify_subject(question_text)
        subject = subject_result.get('subject', 'Unknown')
        
        reasoning_templates = {
            'Mathematics': f"Based on mathematical analysis, the most likely answer is {predicted_answer} with {confidence:.1%} confidence. This appears to be a {subject.lower()} problem.",
            'Physics': f"Using physics principles and formulas, the predicted answer is {predicted_answer} with {confidence:.1%} confidence. This is a {subject.lower()} question.",
            'Chemistry': f"Based on chemical concepts and reactions, the answer is likely {predicted_answer} with {confidence:.1%} confidence. This involves {subject.lower()}.",
            'Biology': f"Using biological concepts and processes, the predicted answer is {predicted_answer} with {confidence:.1%} confidence. This is a {subject.lower()} question."
        }
        
        return reasoning_templates.get(subject, f"The predicted answer is {predicted_answer} with {confidence:.1%} confidence.")
    
    def analyze_question(self, question_text: str) -> Dict[str, Any]:
        """Complete question analysis"""
        logger.info(f"Analyzing question: {question_text[:100]}...")
        
        # Classify subject
        subject_result = self.classify_subject(question_text)
        
        # Predict answer
        answer_result = self.predict_answer(question_text)
        
        # Generate comprehensive analysis
        analysis = {
            'question_text': question_text,
            'subject_analysis': subject_result,
            'answer_analysis': answer_result,
            'overall_confidence': (subject_result.get('confidence', 0) + answer_result.get('confidence', 0)) / 2,
            'analysis_timestamp': datetime.now().isoformat(),
            'recommendations': self.generate_recommendations(subject_result, answer_result)
        }
        
        return analysis
    
    def generate_recommendations(self, subject_result: Dict, answer_result: Dict) -> List[str]:
        """Generate study recommendations based on analysis"""
        recommendations = []
        
        subject = subject_result.get('subject', 'Unknown')
        subject_confidence = subject_result.get('confidence', 0)
        answer_confidence = answer_result.get('confidence', 0)
        
        # Subject-based recommendations
        if subject_confidence < self.config.CONFIDENCE_THRESHOLD:
            recommendations.append(f"Review {subject} fundamentals - the model is uncertain about the subject classification.")
        
        # Answer confidence recommendations
        if answer_confidence < self.config.CONFIDENCE_THRESHOLD:
            recommendations.append("This question appears challenging - consider reviewing related concepts.")
        
        # Subject-specific recommendations
        subject_recommendations = {
            'Mathematics': [
                "Practice similar mathematical problems",
                "Review relevant formulas and theorems",
                "Focus on problem-solving techniques"
            ],
            'Physics': [
                "Review physical laws and principles",
                "Practice applying formulas",
                "Understand the underlying concepts"
            ],
            'Chemistry': [
                "Review chemical reactions and equations",
                "Practice balancing equations",
                "Understand molecular structures"
            ],
            'Biology': [
                "Review biological processes",
                "Understand cell structures and functions",
                "Practice with diagrams and processes"
            ]
        }
        
        if subject in subject_recommendations:
            recommendations.extend(subject_recommendations[subject])
        
        return recommendations
    
    def provide_tutoring_feedback(self, question_text: str, user_answer: str = None) -> Dict[str, Any]:
        """Provide comprehensive tutoring feedback"""
        analysis = self.analyze_question(question_text)
        
        feedback = {
            'question_analysis': analysis,
            'tutoring_feedback': {
                'subject': analysis['subject_analysis']['subject'],
                'predicted_answer': analysis['answer_analysis']['predicted_answer'],
                'confidence_level': analysis['overall_confidence'],
                'explanation': analysis['answer_analysis'].get('reasoning', ''),
                'study_recommendations': analysis['recommendations']
            }
        }
        
        # If user provided an answer, compare and provide feedback
        if user_answer:
            predicted = analysis['answer_analysis']['predicted_answer']
            is_correct = user_answer.upper() == predicted
            
            feedback['answer_comparison'] = {
                'user_answer': user_answer.upper(),
                'predicted_answer': predicted,
                'is_correct': is_correct,
                'feedback': self.generate_answer_feedback(user_answer, predicted, is_correct)
            }
        
        return feedback
    
    def generate_answer_feedback(self, user_answer: str, predicted_answer: str, is_correct: bool) -> str:
        """Generate specific feedback for user's answer"""
        if is_correct:
            return "✅ Excellent! Your answer is correct. You've demonstrated good understanding of this concept."
        else:
            return f"❌ Your answer ({user_answer}) differs from the predicted answer ({predicted_answer}). Consider reviewing the related concepts and try similar problems."
    
    def batch_analyze_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple questions at once"""
        logger.info(f"Batch analyzing {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"Analyzing question {i+1}/{len(questions)}")
            analysis = self.analyze_question(question)
            results.append(analysis)
        
        return results
    
    def generate_study_plan(self, questions: List[str]) -> Dict[str, Any]:
        """Generate a personalized study plan based on question analysis"""
        logger.info("Generating personalized study plan...")
        
        # Analyze all questions
        analyses = self.batch_analyze_questions(questions)
        
        # Group by subject
        subject_groups = {}
        for analysis in analyses:
            subject = analysis['subject_analysis']['subject']
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(analysis)
        
        # Calculate confidence levels per subject
        subject_stats = {}
        for subject, analyses in subject_groups.items():
            avg_confidence = np.mean([a['overall_confidence'] for a in analyses])
            subject_stats[subject] = {
                'count': len(analyses),
                'avg_confidence': avg_confidence,
                'needs_review': avg_confidence < self.config.CONFIDENCE_THRESHOLD
            }
        
        # Generate study plan
        study_plan = {
            'total_questions': len(questions),
            'subject_breakdown': subject_stats,
            'recommended_focus': [],
            'study_schedule': self.generate_study_schedule(subject_stats),
            'practice_recommendations': self.generate_practice_recommendations(subject_stats)
        }
        
        # Identify areas needing focus
        for subject, stats in subject_stats.items():
            if stats['needs_review']:
                study_plan['recommended_focus'].append(subject)
        
        return study_plan
    
    def generate_study_schedule(self, subject_stats: Dict) -> Dict[str, List[str]]:
        """Generate a study schedule based on subject performance"""
        schedule = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for subject, stats in subject_stats.items():
            if stats['needs_review']:
                schedule['high_priority'].append(subject)
            elif stats['avg_confidence'] < 0.8:
                schedule['medium_priority'].append(subject)
            else:
                schedule['low_priority'].append(subject)
        
        return schedule
    
    def generate_practice_recommendations(self, subject_stats: Dict) -> Dict[str, List[str]]:
        """Generate practice recommendations for each subject"""
        recommendations = {}
        
        subject_practices = {
            'Mathematics': [
                "Practice calculus problems",
                "Review algebra fundamentals",
                "Work on geometry problems",
                "Practice trigonometry"
            ],
            'Physics': [
                "Practice mechanics problems",
                "Review thermodynamics",
                "Work on electromagnetism",
                "Practice optics problems"
            ],
            'Chemistry': [
                "Practice chemical equations",
                "Review organic chemistry",
                "Work on physical chemistry",
                "Practice inorganic chemistry"
            ],
            'Biology': [
                "Review cell biology",
                "Practice genetics problems",
                "Work on physiology",
                "Review ecology concepts"
            ]
        }
        
        for subject in subject_stats.keys():
            recommendations[subject] = subject_practices.get(subject, [f"Practice {subject} problems"])
        
        return recommendations

def main():
    """Main function for EAMCET AI Tutor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EAMCET AI Tutor')
    parser.add_argument('--question', help='Single question to analyze')
    parser.add_argument('--questions_file', help='File containing multiple questions')
    parser.add_argument('--user_answer', help='User\'s answer for comparison')
    parser.add_argument('--output', default='tutoring_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    print("🎓 EAMCET AI Tutor")
    print("=" * 40)
    
    # Initialize tutor
    config = InferenceConfig()
    tutor = EAMCETTutor(config)
    
    if args.question:
        # Analyze single question
        print(f"📝 Analyzing question: {args.question[:100]}...")
        
        if args.user_answer:
            feedback = tutor.provide_tutoring_feedback(args.question, args.user_answer)
        else:
            feedback = tutor.provide_tutoring_feedback(args.question)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        print(f"📄 Results saved to {args.output}")
        
        # Print summary
        print("\n📊 Analysis Summary:")
        print(f"   Subject: {feedback['tutoring_feedback']['subject']}")
        print(f"   Predicted Answer: {feedback['tutoring_feedback']['predicted_answer']}")
        print(f"   Confidence: {feedback['tutoring_feedback']['confidence_level']:.2%}")
        
        if 'answer_comparison' in feedback:
            comp = feedback['answer_comparison']
            print(f"   Your Answer: {comp['user_answer']}")
            print(f"   Correct: {'✅' if comp['is_correct'] else '❌'}")
    
    elif args.questions_file:
        # Analyze multiple questions
        with open(args.questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        print(f"📚 Analyzing {len(questions)} questions...")
        
        # Generate study plan
        study_plan = tutor.generate_study_plan(questions)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(study_plan, f, indent=2)
        
        print(f"📄 Study plan saved to {args.output}")
        
        # Print summary
        print("\n📊 Study Plan Summary:")
        print(f"   Total Questions: {study_plan['total_questions']}")
        print(f"   Subjects Covered: {list(study_plan['subject_breakdown'].keys())}")
        print(f"   Focus Areas: {study_plan['recommended_focus']}")
    
    else:
        print("❌ Please provide either --question or --questions_file")
        print("\nUsage examples:")
        print("  python eamcet_inference.py --question 'What is the derivative of x²?'")
        print("  python eamcet_inference.py --question 'What is the derivative of x²?' --user_answer 'A'")
        print("  python eamcet_inference.py --questions_file questions.txt")

if __name__ == "__main__":
    main()

# Usage examples:
# python eamcet_inference.py --question "What is the derivative of x²?" --user_answer "A"
# python eamcet_inference.py --questions_file practice_questions.txt 
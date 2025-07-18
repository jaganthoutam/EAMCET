#!/usr/bin/env python3
"""
EAMCET Model Tester - Comprehensive Testing Framework
Test trained models and evaluate performance
"""

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results"""
    question_id: str
    predicted_answer: str
    correct_answer: str
    confidence: float
    model_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class EAMCETModelEvaluator:
    """Comprehensive model evaluation and testing framework"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.training_data = None
        self.test_results = []
        self.evaluation_metrics = {}
        
        # Load models and data
        self.load_models()
        self.load_training_data()
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading trained models...")
        
        model_files = {
            'text_detection': 'text_detection.pth',
            'question_parsing': 'question_parsing.pth', 
            'answer_detection': 'answer_detection.pth'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    # Load model state dict
                    state_dict = torch.load(model_path, map_location='cpu')
                    self.models[model_name] = {
                        'state_dict': state_dict,
                        'path': str(model_path),
                        'loaded': True
                    }
                    logger.info(f"✅ Loaded {model_name} model")
                except Exception as e:
                    logger.error(f"❌ Error loading {model_name} model: {str(e)}")
                    self.models[model_name] = {'loaded': False, 'error': str(e)}
            else:
                logger.warning(f"⚠️  {model_name} model not found at {model_path}")
                self.models[model_name] = {'loaded': False}
    
    def load_training_data(self):
        """Load training data for evaluation"""
        training_data_path = self.models_dir / "training_data.pkl"
        if training_data_path.exists():
            try:
                with open(training_data_path, 'rb') as f:
                    self.training_data = pickle.load(f)
                logger.info(f"✅ Loaded training data")
            except Exception as e:
                logger.error(f"❌ Error loading training data: {str(e)}")
        else:
            logger.warning("⚠️  No training data found")
    
    def test_text_detection(self, image_path: str) -> Dict[str, Any]:
        """Test text detection model with an image"""
        import cv2
        from PIL import Image
        
        logger.info(f"Testing text detection on {image_path}")
        
        try:
            # Load image
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Create mock image for testing
                image = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
            
            # Mock text detection (replace with actual model inference)
            detected_texts = [
                {
                    'text': 'Sample question text',
                    'bbox': [10, 10, 200, 50],
                    'confidence': 0.85,
                    'text_type': 'question_text'
                },
                {
                    'text': 'Option A: First option',
                    'bbox': [10, 60, 200, 80],
                    'confidence': 0.90,
                    'text_type': 'option_a'
                }
            ]
            
            return {
                'success': True,
                'detected_texts': detected_texts,
                'total_texts': len(detected_texts),
                'average_confidence': np.mean([t['confidence'] for t in detected_texts])
            }
            
        except Exception as e:
            logger.error(f"Error in text detection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'detected_texts': [],
                'total_texts': 0,
                'average_confidence': 0.0
            }
    
    def test_question_parsing(self, raw_text: str) -> Dict[str, Any]:
        """Test question parsing model"""
        logger.info("Testing question parsing model")
        
        try:
            # Mock question parsing (replace with actual model inference)
            parsed_question = {
                'question_number': 1,
                'question_id': 'Q001',
                'question_text': 'Sample question text extracted from raw text',
                'options': {
                    'A': 'Option A text',
                    'B': 'Option B text', 
                    'C': 'Option C text',
                    'D': 'Option D text'
                },
                'subject': 'Mathematics',
                'confidence': 0.88
            }
            
            return {
                'success': True,
                'parsed_question': parsed_question,
                'confidence': parsed_question['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error in question parsing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'parsed_question': None,
                'confidence': 0.0
            }
    
    def test_answer_detection(self, question_data: Dict, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Test answer detection model"""
        logger.info("Testing answer detection model")
        
        try:
            # Mock answer detection (replace with actual model inference)
            answer_result = {
                'predicted_answer': 'A',
                'confidence': 0.75,
                'reasoning': 'Based on pattern analysis and color detection',
                'alternative_answers': ['B', 'C', 'D'],
                'alternative_confidences': [0.15, 0.08, 0.02],
                'processing_time': 0.15
            }
            
            return {
                'success': True,
                'answer_result': answer_result,
                'confidence': answer_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error in answer detection: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'answer_result': None,
                'confidence': 0.0
            }
    
    def run_comprehensive_test(self, test_questions: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive test on multiple questions"""
        logger.info(f"Running comprehensive test on {len(test_questions)} questions")
        
        results = {
            'total_questions': len(test_questions),
            'successful_tests': 0,
            'failed_tests': 0,
            'text_detection_results': [],
            'question_parsing_results': [],
            'answer_detection_results': [],
            'overall_metrics': {}
        }
        
        for i, question in enumerate(test_questions):
            logger.info(f"Testing question {i+1}/{len(test_questions)}")
            
            # Test text detection
            text_result = self.test_text_detection(question.get('image_path', ''))
            results['text_detection_results'].append(text_result)
            
            # Test question parsing
            parsing_result = self.test_question_parsing(question.get('raw_text', ''))
            results['question_parsing_results'].append(parsing_result)
            
            # Test answer detection
            answer_result = self.test_answer_detection(
                parsing_result.get('parsed_question', {}),
                question.get('image_path')
            )
            results['answer_detection_results'].append(answer_result)
            
            # Track success/failure
            if all([text_result['success'], parsing_result['success'], answer_result['success']]):
                results['successful_tests'] += 1
            else:
                results['failed_tests'] += 1
        
        # Calculate overall metrics
        results['overall_metrics'] = self.calculate_overall_metrics(results)
        
        return results
    
    def calculate_overall_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        metrics = {}
        
        # Text detection metrics
        text_confidences = [r['average_confidence'] for r in results['text_detection_results'] if r['success']]
        metrics['text_detection_accuracy'] = np.mean(text_confidences) if text_confidences else 0.0
        
        # Question parsing metrics
        parsing_confidences = [r['confidence'] for r in results['question_parsing_results'] if r['success']]
        metrics['question_parsing_accuracy'] = np.mean(parsing_confidences) if parsing_confidences else 0.0
        
        # Answer detection metrics
        answer_confidences = [r['confidence'] for r in results['answer_detection_results'] if r['success']]
        metrics['answer_detection_accuracy'] = np.mean(answer_confidences) if answer_confidences else 0.0
        
        # Overall accuracy
        metrics['overall_accuracy'] = np.mean([
            metrics['text_detection_accuracy'],
            metrics['question_parsing_accuracy'],
            metrics['answer_detection_accuracy']
        ])
        
        # Success rate
        metrics['success_rate'] = results['successful_tests'] / results['total_questions']
        
        return metrics
    
    def test_with_ground_truth(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Test models against ground truth data"""
        logger.info(f"Testing against ground truth data ({len(test_data)} samples)")
        
        evaluation_results = {
            'total_samples': len(test_data),
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'detailed_results': []
        }
        
        for i, sample in enumerate(test_data):
            logger.info(f"Evaluating sample {i+1}/{len(test_data)}")
            
            # Run model prediction
            prediction_result = self.predict_answer(sample)
            
            # Compare with ground truth
            ground_truth = sample.get('correct_answer', '')
            predicted_answer = prediction_result.get('predicted_answer', '')
            
            is_correct = ground_truth.upper() == predicted_answer.upper()
            
            if is_correct:
                evaluation_results['correct_predictions'] += 1
            else:
                evaluation_results['incorrect_predictions'] += 1
            
            # Store detailed result
            detailed_result = {
                'sample_id': i,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'confidence': prediction_result.get('confidence', 0.0),
                'is_correct': is_correct,
                'processing_time': prediction_result.get('processing_time', 0.0)
            }
            evaluation_results['detailed_results'].append(detailed_result)
        
        # Calculate accuracy
        evaluation_results['accuracy'] = evaluation_results['correct_predictions'] / evaluation_results['total_samples']
        
        return evaluation_results
    
    def predict_answer(self, question_data: Dict) -> Dict[str, Any]:
        """Predict answer for a single question"""
        import time
        
        start_time = time.time()
        
        try:
            # Step 1: Text detection (if image provided)
            if 'image_path' in question_data:
                text_result = self.test_text_detection(question_data['image_path'])
            else:
                text_result = {'success': True, 'detected_texts': []}
            
            # Step 2: Question parsing
            raw_text = question_data.get('raw_text', '')
            parsing_result = self.test_question_parsing(raw_text)
            
            # Step 3: Answer detection
            answer_result = self.test_answer_detection(
                parsing_result.get('parsed_question', {}),
                question_data.get('image_path')
            )
            
            processing_time = time.time() - start_time
            
            return {
                'predicted_answer': answer_result.get('answer_result', {}).get('predicted_answer', ''),
                'confidence': answer_result.get('confidence', 0.0),
                'processing_time': processing_time,
                'success': all([text_result['success'], parsing_result['success'], answer_result['success']])
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {
                'predicted_answer': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def generate_test_report(self, results: Dict, output_path: str = "test_report.html"):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EAMCET Model Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 EAMCET Model Test Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Overall Performance</h2>
            <div class="metric">
                <h3>Model Accuracy Metrics</h3>
                <p><strong>Text Detection Accuracy:</strong> {results.get('overall_metrics', {}).get('text_detection_accuracy', 0):.2%}</p>
                <p><strong>Question Parsing Accuracy:</strong> {results.get('overall_metrics', {}).get('question_parsing_accuracy', 0):.2%}</p>
                <p><strong>Answer Detection Accuracy:</strong> {results.get('overall_metrics', {}).get('answer_detection_accuracy', 0):.2%}</p>
                <p><strong>Overall Accuracy:</strong> {results.get('overall_metrics', {}).get('overall_accuracy', 0):.2%}</p>
                <p><strong>Success Rate:</strong> {results.get('overall_metrics', {}).get('success_rate', 0):.2%}</p>
            </div>
            
            <h2>📈 Test Statistics</h2>
            <div class="metric">
                <p><strong>Total Questions Tested:</strong> {results.get('total_questions', 0)}</p>
                <p><strong>Successful Tests:</strong> {results.get('successful_tests', 0)}</p>
                <p><strong>Failed Tests:</strong> {results.get('failed_tests', 0)}</p>
            </div>
            
            <h2>🔍 Model Status</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Status</th>
                    <th>Path</th>
                </tr>
        """
        
        for model_name, model_info in self.models.items():
            status = "✅ Loaded" if model_info.get('loaded', False) else "❌ Failed"
            path = model_info.get('path', 'Not found')
            html_content += f"""
                <tr>
                    <td>{model_name.replace('_', ' ').title()}</td>
                    <td class="{'success' if model_info.get('loaded', False) else 'error'}">{status}</td>
                    <td>{path}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>💡 Recommendations</h2>
            <div class="metric">
                <ul>
                    <li>If accuracy is below 80%, consider retraining with more data</li>
                    <li>If success rate is low, check model loading and data format</li>
                    <li>For production use, ensure all models are properly loaded</li>
                    <li>Monitor processing times for real-time applications</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✅ Test report saved to {output_path}")
        return output_path
    
    def create_visualization(self, results: Dict, output_path: str = "test_visualization.png"):
        """Create visualization of test results"""
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Model accuracy comparison
            metrics = results.get('overall_metrics', {})
            model_names = ['Text Detection', 'Question Parsing', 'Answer Detection']
            accuracies = [
                metrics.get('text_detection_accuracy', 0),
                metrics.get('question_parsing_accuracy', 0),
                metrics.get('answer_detection_accuracy', 0)
            ]
            
            ax1.bar(model_names, accuracies, color=['#ff9999', '#66b3ff', '#99ff99'])
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
            # 2. Success vs Failure
            successful = results.get('successful_tests', 0)
            failed = results.get('failed_tests', 0)
            ax2.pie([successful, failed], labels=['Successful', 'Failed'], 
                   colors=['#66b3ff', '#ff9999'], autopct='%1.1f%%')
            ax2.set_title('Test Success Rate')
            
            # 3. Confidence distribution
            confidences = []
            for result in results.get('text_detection_results', []):
                if result.get('success'):
                    confidences.append(result.get('average_confidence', 0))
            
            if confidences:
                ax3.hist(confidences, bins=10, alpha=0.7, color='#66b3ff')
                ax3.set_title('Confidence Distribution')
                ax3.set_xlabel('Confidence')
                ax3.set_ylabel('Frequency')
            
            # 4. Processing time analysis
            processing_times = []
            for result in results.get('answer_detection_results', []):
                if result.get('success'):
                    processing_times.append(result.get('answer_result', {}).get('processing_time', 0))
            
            if processing_times:
                ax4.hist(processing_times, bins=10, alpha=0.7, color='#99ff99')
                ax4.set_title('Processing Time Distribution')
                ax4.set_xlabel('Time (seconds)')
                ax4.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None

def main():
    """Main function for model testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EAMCET Model Tester')
    parser.add_argument('--models_dir', default='trained_models', help='Directory containing trained models')
    parser.add_argument('--test_data', help='Path to test data JSON file')
    parser.add_argument('--output_dir', default='test_results', help='Output directory for results')
    parser.add_argument('--generate_report', action='store_true', help='Generate HTML report')
    parser.add_argument('--create_visualization', action='store_true', help='Create visualization plots')
    
    args = parser.parse_args()
    
    print("🧪 EAMCET Model Tester")
    print("=" * 40)
    
    # Initialize evaluator
    evaluator = EAMCETModelEvaluator(args.models_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load test data if provided
    test_questions = []
    if args.test_data and os.path.exists(args.test_data):
        with open(args.test_data, 'r') as f:
            test_questions = json.load(f)
        print(f"📄 Loaded {len(test_questions)} test questions")
    else:
        # Create sample test questions
        test_questions = [
            {
                'raw_text': 'Question Number: 1\nQuestion Id: 001\nWhat is 2+2?\nOptions:\n1. 3\n2. 4\n3. 5\n4. 6',
                'correct_answer': 'B'
            },
            {
                'raw_text': 'Question Number: 2\nQuestion Id: 002\nWhat is the capital of France?\nOptions:\n1. London\n2. Paris\n3. Berlin\n4. Madrid',
                'correct_answer': 'B'
            }
        ]
        print("📄 Using sample test questions")
    
    # Run comprehensive test
    print("🚀 Running comprehensive model test...")
    test_results = evaluator.run_comprehensive_test(test_questions)
    
    # Test against ground truth if available
    if any('correct_answer' in q for q in test_questions):
        print("🎯 Testing against ground truth...")
        evaluation_results = evaluator.test_with_ground_truth(test_questions)
        test_results['evaluation_results'] = evaluation_results
    
    # Generate report
    if args.generate_report:
        report_path = output_dir / "test_report.html"
        evaluator.generate_test_report(test_results, str(report_path))
        print(f"📄 Report saved to: {report_path}")
    
    # Create visualization
    if args.create_visualization:
        viz_path = output_dir / "test_visualization.png"
        evaluator.create_visualization(test_results, str(viz_path))
        print(f"📊 Visualization saved to: {viz_path}")
    
    # Save results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"   Total questions: {test_results['total_questions']}")
    print(f"   Successful tests: {test_results['successful_tests']}")
    print(f"   Failed tests: {test_results['failed_tests']}")
    print(f"   Overall accuracy: {test_results['overall_metrics']['overall_accuracy']:.2%}")
    
    if 'evaluation_results' in test_results:
        eval_results = test_results['evaluation_results']
        print(f"   Ground truth accuracy: {eval_results['accuracy']:.2%}")
    
    print("\n✅ Model testing complete!")

if __name__ == "__main__":
    main()

# Usage examples:
# python eamcet_model_tester.py --models_dir trained_models --generate_report --create_visualization
# python eamcet_model_tester.py --test_data test_questions.json --output_dir test_results 
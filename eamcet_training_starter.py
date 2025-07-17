#!/usr/bin/env python3
"""
EAMCET AI Tutor - Training Pipeline Step 1
PDF Processing and Initial Data Extraction

This script processes your EAMCET PDFs and prepares them for model training.
Run this first to extract and organize your training data.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import argparse

class EAMCETDataExtractor:
    """Extract and organize data from EAMCET PDFs"""
    
    def __init__(self, input_folder: str, output_folder: str = "eamcet_processed_data"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_folder / "images").mkdir(exist_ok=True)
        (self.output_folder / "raw_text").mkdir(exist_ok=True)
        (self.output_folder / "annotations").mkdir(exist_ok=True)
        (self.output_folder / "logs").mkdir(exist_ok=True)
        
        # Initialize stats
        self.processing_stats = {
            'total_pdfs': 0,
            'pages_extracted': 0,
            'questions_found': 0,
            'answer_keys_found': 0,
            'errors': []
        }
    
    def scan_and_organize_pdfs(self) -> Dict:
        """Scan input folder and organize PDFs by type and metadata"""
        print("🔍 Scanning PDF files...")
        
        pdf_catalog = {
            'EAMCET-AP': {'MPC': [], 'BiPC': []},
            'EAMCET-TG': {'MPC': [], 'BiPC': []}
        }
        
        # Recursively find all PDFs
        for pdf_path in self.input_folder.rglob("*.pdf"):
            try:
                pdf_info = self.extract_pdf_metadata(pdf_path)
                
                # Categorize by path structure
                parts = pdf_path.parts
                state = None
                stream = None
                
                # Identify state
                if 'EAMCET-AP' in str(pdf_path) or 'ap-eamcet' in pdf_path.name.lower():
                    state = 'EAMCET-AP'
                elif 'EAMCET-TG' in str(pdf_path) or 'tg-eamcet' in pdf_path.name.lower():
                    state = 'EAMCET-TG'
                else:
                    state = 'EAMCET-AP'  # Default assumption
                
                # Identify stream
                if 'MPC' in str(pdf_path) or 'engineering' in pdf_path.name.lower():
                    stream = 'MPC'
                elif 'BiPC' in str(pdf_path) or any(word in pdf_path.name.lower() 
                    for word in ['agriculture', 'pharmacy', 'bipc']):
                    stream = 'BiPC'
                else:
                    stream = 'MPC'  # Default assumption
                
                pdf_info['state'] = state
                pdf_info['stream'] = stream
                pdf_catalog[state][stream].append(pdf_info)
                
                self.processing_stats['total_pdfs'] += 1
                
            except Exception as e:
                self.processing_stats['errors'].append(f"Error processing {pdf_path}: {str(e)}")
                print(f"❌ Error processing {pdf_path.name}: {str(e)}")
        
        # Save catalog
        with open(self.output_folder / "pdf_catalog.json", 'w') as f:
            json.dump(pdf_catalog, f, indent=2, default=str)
        
        self.print_catalog_summary(pdf_catalog)
        return pdf_catalog
    
    def extract_pdf_metadata(self, pdf_path: Path) -> Dict:
        """Extract metadata from PDF filename and properties"""
        filename = pdf_path.name
        
        # Extract year
        year_match = re.search(r'(20\d{2})', filename)
        year = year_match.group(1) if year_match else 'unknown'
        
        # Extract paper type
        paper_type = 'unknown'
        if 'question-paper' in filename.lower():
            paper_type = 'question_paper'
        elif 'answer' in filename.lower() and 'key' in filename.lower():
            paper_type = 'answer_key'
        elif 'solution' in filename.lower():
            paper_type = 'solution'
        
        # Extract shift info
        shift_match = re.search(r'shift[_-]?(\d+)', filename, re.IGNORECASE)
        shift = shift_match.group(1) if shift_match else None
        
        # Get file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        
        return {
            'filename': filename,
            'path': str(pdf_path),
            'year': year,
            'paper_type': paper_type,
            'shift': shift,
            'size_mb': round(file_size_mb, 2)
        }
    
    def extract_pages_from_pdf(self, pdf_path: str, output_prefix: str) -> List[str]:
        """Extract pages from PDF as high-resolution images"""
        print(f"📄 Extracting pages from {Path(pdf_path).name}...")
        
        doc = fitz.open(pdf_path)
        page_paths = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # High resolution matrix for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.samples
                img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # Save image
                img_filename = f"{output_prefix}_page_{page_num:03d}.png"
                img_path = self.output_folder / "images" / img_filename
                img.save(img_path, "PNG", quality=95)
                
                page_paths.append(str(img_path))
                self.processing_stats['pages_extracted'] += 1
                
            except Exception as e:
                error_msg = f"Error extracting page {page_num} from {pdf_path}: {str(e)}"
                self.processing_stats['errors'].append(error_msg)
                print(f"❌ {error_msg}")
        
        doc.close()
        return page_paths
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text from image using OCR"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'text': '', 'error': 'Could not load image'}
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # OCR with detailed output
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Extract text
            text = pytesseract.image_to_string(img_rgb, config=custom_config)
            
            # Extract word-level data for bounding boxes
            data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
            
            # Process OCR data
            words_data = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    word_text = data['text'][i].strip()
                    if word_text:
                        words_data.append({
                            'text': word_text,
                            'bbox': [data['left'][i], data['top'][i], 
                                   data['left'][i] + data['width'][i], 
                                   data['top'][i] + data['height'][i]],
                            'confidence': data['conf'][i]
                        })
            
            return {
                'text': text.strip(),
                'words': words_data,
                'image_path': image_path,
                'image_size': img.shape[:2]
            }
            
        except Exception as e:
            return {'text': '', 'error': str(e)}
    
    def detect_question_patterns(self, text: str) -> List[Dict]:
        """Detect question patterns in extracted text"""
        questions = []
        
        # Pattern for question numbers
        question_pattern = r'Question Number\s*:\s*(\d+)'
        question_id_pattern = r'Question Id\s*:\s*(\d+)'
        
        # Find all question matches
        question_matches = list(re.finditer(question_pattern, text, re.IGNORECASE))
        id_matches = list(re.finditer(question_id_pattern, text, re.IGNORECASE))
        
        for i, match in enumerate(question_matches):
            question_num = int(match.group(1))
            start_pos = match.end()
            
            # Find end position (next question or end of text)
            if i + 1 < len(question_matches):
                end_pos = question_matches[i + 1].start()
            else:
                end_pos = len(text)
            
            question_text = text[start_pos:end_pos].strip()
            
            # Extract question ID if available
            question_id = None
            for id_match in id_matches:
                if start_pos <= id_match.start() < end_pos:
                    question_id = id_match.group(1)
                    break
            
            # Detect options
            options = self.extract_options_from_text(question_text)
            
            questions.append({
                'number': question_num,
                'id': question_id,
                'text': question_text,
                'options': options,
                'raw_position': (start_pos, end_pos)
            })
            
            self.processing_stats['questions_found'] += 1
        
        return questions
    
    def extract_options_from_text(self, question_text: str) -> Dict[str, str]:
        """Extract options A, B, C, D from question text"""
        options = {'A': '', 'B': '', 'C': '', 'D': ''}
        
        # Look for "Options :" section
        options_match = re.search(r'Options\s*:\s*(.*?)(?=Question|$)', question_text, re.DOTALL | re.IGNORECASE)
        
        if options_match:
            options_text = options_match.group(1)
            
            # Pattern to match numbered options
            option_lines = re.split(r'\n\s*(?=[1-4]\.)', options_text)
            
            for line in option_lines:
                line = line.strip()
                if re.match(r'^[1-4]\.', line):
                    option_num = int(line[0])
                    option_text = line[2:].strip()
                    
                    if 1 <= option_num <= 4:
                        option_letter = chr(64 + option_num)  # 1->A, 2->B, etc.
                        options[option_letter] = option_text
        
        return options
    
    def detect_answer_key_indicators(self, text: str) -> Dict:
        """Detect answer key format indicators"""
        indicators = {
            'has_color_coding': False,
            'correct_pattern': None,
            'incorrect_pattern': None,
            'format_type': 'unknown'
        }
        
        # Check for color coding patterns
        green_correct_pattern = r'Options shown in green color and with.*icon are correct'
        red_incorrect_pattern = r'Options shown in red color and with.*icon are incorrect'
        
        if re.search(green_correct_pattern, text, re.IGNORECASE):
            indicators['has_color_coding'] = True
            indicators['correct_pattern'] = 'green_with_icon'
            indicators['format_type'] = 'color_coded'
        
        if re.search(red_incorrect_pattern, text, re.IGNORECASE):
            indicators['has_color_coding'] = True
            indicators['incorrect_pattern'] = 'red_with_icon'
            indicators['format_type'] = 'color_coded'
        
        self.processing_stats['answer_keys_found'] += 1
        return indicators
    
    def process_single_pdf(self, pdf_info: Dict) -> Dict:
        """Process a single PDF file completely"""
        pdf_path = pdf_info['path']
        filename = Path(pdf_path).stem
        
        print(f"\n🔄 Processing: {pdf_info['filename']}")
        print(f"   Type: {pdf_info['paper_type']} | Year: {pdf_info['year']} | Stream: {pdf_info.get('stream', 'unknown')}")
        
        # Extract pages
        page_paths = self.extract_pages_from_pdf(pdf_path, filename)
        
        # Process each page
        processed_pages = []
        for page_path in page_paths:
            page_data = self.extract_text_from_image(page_path)
            
            if 'error' not in page_data:
                # Detect questions or answers based on type
                if pdf_info['paper_type'] == 'question_paper':
                    questions = self.detect_question_patterns(page_data['text'])
                    page_data['questions'] = questions
                elif pdf_info['paper_type'] == 'answer_key':
                    answer_indicators = self.detect_answer_key_indicators(page_data['text'])
                    page_data['answer_indicators'] = answer_indicators
                
                processed_pages.append(page_data)
        
        # Save processed data
        output_data = {
            'pdf_info': pdf_info,
            'pages': processed_pages,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        # Save to JSON file
        output_file = self.output_folder / "raw_text" / f"{filename}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return output_data
    
    def process_all_pdfs(self, pdf_catalog: Dict) -> Dict:
        """Process all PDFs in the catalog"""
        print("\n🚀 Starting PDF processing...")
        
        all_processed_data = {
            'EAMCET-AP': {'MPC': [], 'BiPC': []},
            'EAMCET-TG': {'MPC': [], 'BiPC': []}
        }
        
        total_files = sum(len(files) for state in pdf_catalog.values() 
                         for files in state.values())
        current_file = 0
        
        for state, streams in pdf_catalog.items():
            for stream, pdf_list in streams.items():
                print(f"\n📚 Processing {state} - {stream} ({len(pdf_list)} files)")
                
                for pdf_info in pdf_list:
                    current_file += 1
                    print(f"Progress: {current_file}/{total_files}")
                    
                    try:
                        processed_data = self.process_single_pdf(pdf_info)
                        all_processed_data[state][stream].append(processed_data)
                    except Exception as e:
                        error_msg = f"Error processing {pdf_info['filename']}: {str(e)}"
                        self.processing_stats['errors'].append(error_msg)
                        print(f"❌ {error_msg}")
        
        return all_processed_data
    
    def create_training_dataset_structure(self, processed_data: Dict) -> Dict:
        """Create structured dataset for model training"""
        print("\n🎯 Creating training dataset structure...")
        
        training_dataset = {
            'questions': [],
            'answer_keys': [],
            'question_answer_pairs': [],
            'subjects': {
                'Mathematics': [],
                'Physics': [], 
                'Chemistry': [],
                'Biology': []
            }
        }
        
        for state, streams in processed_data.items():
            for stream, pdf_data_list in streams.items():
                for pdf_data in pdf_data_list:
                    pdf_type = pdf_data['pdf_info']['paper_type']
                    
                    for page in pdf_data['pages']:
                        if 'questions' in page:
                            # Process questions
                            for question in page['questions']:
                                question_entry = {
                                    'state': state,
                                    'stream': stream,
                                    'year': pdf_data['pdf_info']['year'],
                                    'question_number': question['number'],
                                    'question_id': question['id'],
                                    'question_text': question['text'],
                                    'options': question['options'],
                                    'image_path': page['image_path'],
                                    'subject': self.determine_subject(question['number'], stream)
                                }
                                
                                training_dataset['questions'].append(question_entry)
                                
                                # Add to subject-specific list
                                subject = question_entry['subject']
                                if subject in training_dataset['subjects']:
                                    training_dataset['subjects'][subject].append(question_entry)
                        
                        elif 'answer_indicators' in page:
                            # Process answer keys
                            answer_entry = {
                                'state': state,
                                'stream': stream,
                                'year': pdf_data['pdf_info']['year'],
                                'format_indicators': page['answer_indicators'],
                                'image_path': page['image_path'],
                                'raw_text': page['text']
                            }
                            
                            training_dataset['answer_keys'].append(answer_entry)
        
        return training_dataset
    
    def determine_subject(self, question_number: int, stream: str) -> str:
        """Determine subject based on question number and stream"""
        if stream == 'MPC':
            if 1 <= question_number <= 80:
                return 'Mathematics'
            elif 81 <= question_number <= 120:
                return 'Physics'
            elif 121 <= question_number <= 160:
                return 'Chemistry'
        elif stream == 'BiPC':
            if 1 <= question_number <= 80:
                return 'Biology'
            elif 81 <= question_number <= 120:
                return 'Physics'
            elif 121 <= question_number <= 160:
                return 'Chemistry'
        
        return 'Unknown'
    
    def generate_summary_report(self, training_dataset: Dict) -> str:
        """Generate a comprehensive summary report"""
        report = []
        report.append("=" * 60)
        report.append("EAMCET DATA PROCESSING SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic stats
        report.append("📊 PROCESSING STATISTICS:")
        report.append(f"   Total PDFs processed: {self.processing_stats['total_pdfs']}")
        report.append(f"   Pages extracted: {self.processing_stats['pages_extracted']}")
        report.append(f"   Questions found: {self.processing_stats['questions_found']}")
        report.append(f"   Answer keys found: {self.processing_stats['answer_keys_found']}")
        report.append(f"   Errors: {len(self.processing_stats['errors'])}")
        report.append("")
        
        # Dataset composition
        report.append("📚 DATASET COMPOSITION:")
        report.append(f"   Total questions: {len(training_dataset['questions'])}")
        report.append(f"   Total answer keys: {len(training_dataset['answer_keys'])}")
        report.append("")
        
        # Subject breakdown
        report.append("📖 SUBJECT BREAKDOWN:")
        for subject, questions in training_dataset['subjects'].items():
            if questions:  # Only show subjects with questions
                report.append(f"   {subject}: {len(questions)} questions")
        report.append("")
        
        # Year distribution
        years = {}
        for q in training_dataset['questions']:
            year = q['year']
            years[year] = years.get(year, 0) + 1
        
        report.append("📅 YEAR DISTRIBUTION:")
        for year in sorted(years.keys()):
            report.append(f"   {year}: {years[year]} questions")
        report.append("")
        
        # Stream distribution
        streams = {}
        for q in training_dataset['questions']:
            stream = q['stream']
            streams[stream] = streams.get(stream, 0) + 1
        
        report.append("🎯 STREAM DISTRIBUTION:")
        for stream, count in streams.items():
            report.append(f"   {stream}: {count} questions")
        report.append("")
        
        # Error summary
        if self.processing_stats['errors']:
            report.append("❌ ERRORS ENCOUNTERED:")
            for error in self.processing_stats['errors'][:5]:  # Show first 5 errors
                report.append(f"   - {error}")
            if len(self.processing_stats['errors']) > 5:
                report.append(f"   ... and {len(self.processing_stats['errors']) - 5} more errors")
        else:
            report.append("✅ NO ERRORS ENCOUNTERED")
        
        report.append("")
        report.append("🎯 NEXT STEPS:")
        report.append("   1. Review the extracted data in 'processed_data' folder")
        report.append("   2. Use the annotation tool to label questions and answers")
        report.append("   3. Run the model training pipeline")
        report.append("   4. Evaluate model performance")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def print_catalog_summary(self, pdf_catalog: Dict):
        """Print a summary of the PDF catalog"""
        print("\n📋 PDF CATALOG SUMMARY:")
        print("-" * 40)
        
        total_files = 0
        for state, streams in pdf_catalog.items():
            state_total = 0
            print(f"\n🏛️  {state}:")
            
            for stream, files in streams.items():
                print(f"   📚 {stream}: {len(files)} files")
                state_total += len(files)
                
                # Show file types
                types = {}
                for file_info in files:
                    file_type = file_info['paper_type']
                    types[file_type] = types.get(file_type, 0) + 1
                
                for file_type, count in types.items():
                    print(f"      - {file_type}: {count}")
            
            print(f"   Total: {state_total} files")
            total_files += state_total
        
        print(f"\n📊 OVERALL TOTAL: {total_files} PDF files")
        print("-" * 40)


def main():
    """Main function to run the EAMCET data extraction pipeline"""
    parser = argparse.ArgumentParser(description='EAMCET PDF Processing Pipeline')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing EAMCET PDFs')
    parser.add_argument('--output', '-o', default='eamcet_processed_data', help='Output folder for processed data')
    parser.add_argument('--skip-ocr', action='store_true', help='Skip OCR processing (faster for testing)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"❌ Error: Input folder '{args.input}' does not exist!")
        sys.exit(1)
    
    print("🚀 Starting EAMCET PDF Processing Pipeline")
    print(f"📁 Input folder: {args.input}")
    print(f"📁 Output folder: {args.output}")
    
    # Initialize extractor
    extractor = EAMCETDataExtractor(args.input, args.output)
    
    try:
        # Step 1: Scan and organize PDFs
        pdf_catalog = extractor.scan_and_organize_pdfs()
        
        if not args.skip_ocr:
            # Step 2: Process all PDFs
            processed_data = extractor.process_all_pdfs(pdf_catalog)
            
            # Step 3: Create training dataset structure
            training_dataset = extractor.create_training_dataset_structure(processed_data)
            
            # Step 4: Save training dataset
            dataset_file = extractor.output_folder / "training_dataset.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(training_dataset, f, indent=2, ensure_ascii=False)
            
            # Step 5: Generate and save report
            report = extractor.generate_summary_report(training_dataset)
            report_file = extractor.output_folder / "processing_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(report)
            
            print(f"\n✅ Processing complete!")
            print(f"📁 All data saved to: {extractor.output_folder}")
            print(f"📄 Training dataset: {dataset_file}")
            print(f"📄 Report: {report_file}")
        else:
            print("⏭️  Skipped OCR processing (use --skip-ocr flag)")
            print(f"📁 PDF catalog saved to: {extractor.output_folder / 'pdf_catalog.json'}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# Quick test function
def quick_test(input_folder: str):
    """Quick test function for Jupyter notebook or interactive use"""
    extractor = EAMCETDataExtractor(input_folder)
    
    # Just scan PDFs without processing
    pdf_catalog = extractor.scan_and_organize_pdfs()
    
    print(f"Found {extractor.processing_stats['total_pdfs']} PDF files")
    return pdf_catalog, extractor

# Usage examples:
"""
# Command line usage:
python eamcet_training_starter.py --input /path/to/your/EAMCET/pdfs --output processed_data

# Interactive usage:
catalog, extractor = quick_test("/path/to/your/EAMCET/pdfs")
"""
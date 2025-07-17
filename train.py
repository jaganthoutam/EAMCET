#!/usr/bin/env python3
"""
EAMCET AI Tutor - Main Training Script
Usage: python train.py --data_path /path/to/your/pdfs
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from eamcet_training_starter import EAMCETDataExtractor
from config import *

def main():
    parser = argparse.ArgumentParser(description='EAMCET AI Tutor Training Pipeline')
    parser.add_argument('--data_path', required=True, help='Path to folder containing EAMCET PDFs')
    parser.add_argument('--output_path', default='processed_data', help='Output path for processed data')
    parser.add_argument('--stage', choices=['extract', 'train', 'all'], default='all', 
                       help='Which stage to run')
    
    args = parser.parse_args()
    
    print("🚀 EAMCET AI Tutor Training Pipeline")
    print("=" * 50)
    
    if args.stage in ['extract', 'all']:
        print("📊 Stage 1: Data Extraction")
        extractor = EAMCETDataExtractor(args.data_path, args.output_path)
        
        # Extract and process PDFs
        pdf_catalog = extractor.scan_and_organize_pdfs()
        processed_data = extractor.process_all_pdfs(pdf_catalog)
        training_dataset = extractor.create_training_dataset_structure(processed_data)
        
        # Save training dataset
        import json
        dataset_file = Path(args.output_path) / "training_dataset.json"
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(training_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Data extraction complete! Dataset saved to: {dataset_file}")
    
    if args.stage in ['train', 'all']:
        print("🤖 Stage 2: Model Training")
        print("⚠️  Model training implementation coming in next step...")
        print("   For now, use the extracted data to create annotations")
    
    print("\n🎉 Pipeline execution complete!")

if __name__ == "__main__":
    main()

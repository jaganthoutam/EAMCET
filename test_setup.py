#!/usr/bin/env python3
"""
Test script to verify setup and dependencies
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        import fitz
        print(f"✅ PyMuPDF: {fitz.version}")
    except ImportError:
        print("❌ PyMuPDF not installed")
        return False
    
    try:
        import pytesseract
        print(f"✅ Pytesseract: {pytesseract.__version__}")
        
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        print(f"   Tesseract: {version}")
    except ImportError:
        print("❌ Pytesseract not installed")
        return False
    except Exception as e:
        print(f"⚠️  Tesseract issue: {e}")
        print("   Please install Tesseract OCR system package")
    
    try:
        from PIL import Image
        print(f"✅ Pillow (PIL): {Image.__version__}")
    except ImportError:
        print("❌ Pillow not installed")
        return False
    
    print("\n🎉 All dependencies are correctly installed!")
    return True

def test_project_structure():
    """Test if project structure is correct"""
    from pathlib import Path
    
    required_dirs = [
        'data', 'models', 'training', 'inference',
        'data/raw_pdfs', 'data/processed_images', 
        'data/annotations', 'data/augmented'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ Directory exists: {dir_name}")
        else:
            print(f"❌ Missing directory: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_name}")
    
    print("\n✅ Project structure is ready!")

def main():
    print("🧪 EAMCET AI Tutor Setup Test")
    print("=" * 40)
    
    print("\n📦 Testing Dependencies:")
    if not test_imports():
        print("\n❌ Setup incomplete. Please run setup.sh again.")
        return
    
    print("\n📁 Testing Project Structure:")
    test_project_structure()
    
    print("\n🎯 Next Steps:")
    print("1. Copy your EAMCET PDFs to data/raw_pdfs/")
    print("2. Run: python train.py --data_path data/raw_pdfs --stage extract")
    print("3. Review the results in processed_data/")

if __name__ == "__main__":
    main()

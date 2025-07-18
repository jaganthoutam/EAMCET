#!/usr/bin/env python3
"""
Standalone download script for Google Colab
Run this cell if the main download cell fails
"""

from google.colab import files
import zipfile
import os
import glob

def download_eamcet_results():
    """Download EAMCET pipeline results"""
    print("📦 Preparing results for download...")
    
    # Check if results exist
    if not os.path.exists("colab_results"):
        print("❌ No results found!")
        print("💡 Make sure you ran the pipeline first")
        return False
    
    try:
        # Create zip file
        zip_filename = "eamcet_results.zip"
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            # Add all files from colab_results
            for root, dirs, files_list in os.walk("colab_results"):
                for file in files_list:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, "colab_results")
                    zipf.write(file_path, arcname)
                    print(f"📁 Added: {arcname}")
        
        print(f"✅ Created {zip_filename}")
        
        # Download using Colab's method
        files.download(zip_filename)
        print("✅ Download initiated!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating zip: {str(e)}")
        
        # Fallback: list files for manual download
        print("📁 Available files for manual download:")
        for root, dirs, files_list in os.walk("colab_results"):
            for file in files_list:
                file_path = os.path.join(root, file)
                print(f"  📄 {file_path}")
        
        print("💡 You can manually download these files from the file browser")
        return False

# Run the download
download_eamcet_results()
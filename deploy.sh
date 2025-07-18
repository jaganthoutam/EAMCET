#!/bin/bash
# EAMCET Zero Manual Pipeline - GitHub Deployment Script

echo "🚀 Deploying EAMCET Zero Manual Pipeline to GitHub"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Add EAMCET Zero Manual Pipeline with Colab support

- Added eamcet_zero_manual_pipeline.py
- Added Colab notebook and requirements
- Added deployment scripts
- Updated documentation"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Please add your GitHub repository as remote:"
    echo "   git remote add origin https://github.com/jaganthoutam/EAMCET.git"
    echo "   Then run: git push -u origin main"
else
    echo "🚀 Pushing to GitHub..."
    git push origin main
fi

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps for Google Colab:"
echo "1. Go to https://colab.research.google.com"
echo "2. Upload eamcet_colab_notebook.ipynb"
echo "3. Update the GitHub URL in the notebook"
echo "4. Run all cells sequentially"
echo "5. Upload your EAMCET PDFs when prompted"

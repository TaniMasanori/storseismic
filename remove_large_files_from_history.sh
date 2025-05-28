#!/bin/bash

# Script to remove large files from Git history
# This will completely rewrite Git history

set -e

echo "=== Removing large files from Git history ==="

# Check if git filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "Installing git-filter-repo..."
    pip install git-filter-repo
fi

# List of large files to remove from history
LARGE_FILES=(
    "nb1_original_DAS_256input.ipynb"
    "data/pretrain/test_data.pt"
    "data/pretrain/train_data.pt"
)

echo "Files to remove from Git history:"
for file in "${LARGE_FILES[@]}"; do
    echo "  - $file"
done

# Create backup
echo "Creating backup..."
cp -r .git .git.backup

# Remove files from Git history
echo "Removing files from Git history..."
for file in "${LARGE_FILES[@]}"; do
    echo "Removing $file from history..."
    git filter-repo --path "$file" --invert-paths --force
done

echo "=== Large files removed from Git history ==="
echo "Next steps:"
echo "1. Verify repository state"
echo "2. Force push: git push --force-with-lease origin main"
echo "3. All collaborators need to re-clone the repository"

# Show repository size reduction
echo ""
echo "Repository size after cleanup:"
du -sh .git 
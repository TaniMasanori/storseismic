#!/bin/bash

echo "=== Creating clean repository ==="

# Current directory backup
CURRENT_DIR=$(pwd)
BACKUP_DIR="${CURRENT_DIR}_backup_$(date +%Y%m%d-%H%M%S)"

echo "=== Backing up current repository to $BACKUP_DIR ==="
cp -r "$CURRENT_DIR" "$BACKUP_DIR"

echo "=== Creating clean version ==="
# Remove Git history
rm -rf .git

# Remove large files
rm -f nb1_original_DAS_256input.ipynb
rm -rf data/pretrain/*.pt

# Create clean data structure
mkdir -p data/pretrain
touch data/pretrain/.gitkeep

# Initialize new Git repository
git init
git branch -m main

# Add clean files
git add .
git commit -m "Initial commit: Clean repository without large files

- Add memory optimization modules
- Add development guidelines
- Configure proper .gitignore
- Exclude large data files"

echo "=== Clean repository created ==="
echo "Backup saved at: $BACKUP_DIR"
echo "Next: git remote add origin <your-repo-url>"
echo "Then: git push -u origin main" 
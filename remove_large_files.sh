#!/bin/bash

# Remove large files from Git history
echo "Removing large files from Git history..."

# Remove data files
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch data/pretrain/train_data.pt data/pretrain/test_data.pt' \
  --prune-empty --tag-name-filter cat -- --all

# Remove large notebook
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch nb1_original_DAS_256input.ipynb' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Large files removed from Git history" 
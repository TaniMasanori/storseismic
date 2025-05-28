#!/bin/bash

set -e

echo "=== Starting Git history cleanup ==="

# Backup current branch
git branch backup-$(date +%Y%m%d-%H%M%S)

echo "=== Step 1: Remove large files using git filter-branch ==="

# Remove specific large files
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch \
    nb1_original_DAS_256input.ipynb \
    "data/pretrain/train_data.pt" \
    "data/pretrain/test_data.pt" \
    data/pretrain/* \
    || true' \
  --prune-empty --tag-name-filter cat -- --all

echo "=== Step 2: Clean up Git repository ==="
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "=== Step 3: Verify cleanup ==="
git log --oneline -10
git ls-files | grep -E '\.(pt|ipynb)$' || echo "No .pt or large .ipynb files found in Git"

echo "=== Step 4: Check repository size ==="
du -sh .git/

echo "=== Cleanup completed! ==="
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Remove large files from history'"
echo "3. git push --force-with-lease origin main" 
#!/bin/bash

echo "=== Checking current Git status ==="
git status

echo "=== Checking large files in Git ==="
git ls-files | xargs -I {} sh -c 'echo "$(git log --oneline {} | wc -l) {}"' | sort -nr | head -20

echo "=== Removing large files from working directory and index ==="
# Remove from working directory and Git index
rm -f nb1_original_DAS_256input.ipynb
rm -rf data/pretrain/
git rm --cached nb1_original_DAS_256input.ipynb 2>/dev/null || true
git rm --cached data/pretrain/train_data.pt 2>/dev/null || true
git rm --cached data/pretrain/test_data.pt 2>/dev/null || true
git rm -r --cached data/ 2>/dev/null || true

echo "=== Creating clean data directory structure ==="
mkdir -p data/pretrain
touch data/pretrain/.gitkeep

echo "=== Files removed from working directory ===" 
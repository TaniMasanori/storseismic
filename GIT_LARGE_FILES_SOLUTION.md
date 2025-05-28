# Git Large Files Issue - Solution Documentation

## Problem
GitHub rejected the push due to large files exceeding size limits:
- `nb1_original_DAS_256input.ipynb` (98.10 MB) - exceeded 50MB warning threshold
- `data/pretrain/test_data.pt` (234.14 MB / 558.31 MB) - exceeded 100MB limit
- `data/pretrain/train_data.pt` (2233.25 MB / 468.28 MB) - exceeded 100MB limit

## Solution Applied

### 1. Git History Cleanup
Used `git filter-repo` to completely remove large files from Git history:
```bash
./remove_large_files_from_history.sh
```

### 2. Enhanced .gitignore
Updated `.gitignore` to prevent future large file commits:
- Data files (*.pt, *.pth)
- Large notebooks (*_original_*.ipynb)
- Model files and checkpoints
- Additional file types prone to being large

### 3. Git LFS Configuration
Updated `.gitattributes` to handle necessary large files via Git LFS:
- PyTorch model files (*.pt, *.pth)
- HDF5 files (*.hdf5, *.h5)
- NumPy archives (*.npz)
- Pickle files (*.pkl, *.pickle)

### 4. Repository Reset
- Removed problematic files from entire Git history
- Force pushed cleaned repository
- Repository size reduced significantly

## Results
- ✅ Repository successfully pushed to GitHub
- ✅ Large files removed from Git history
- ✅ Future large file commits prevented
- ✅ Git LFS configured for necessary large files

## Important Notes

### For Collaborators
**All collaborators must re-clone the repository** because Git history was completely rewritten:
```bash
git clone git@github.com:TaniMasanori/storseismic.git
```

### For Large Data Files
Going forward, large data files should be:
1. Excluded via `.gitignore` (recommended for datasets)
2. Stored externally (cloud storage, data repositories)
3. Used with Git LFS only when necessary for version control

### Repository Size
- Before cleanup: >2GB due to large files
- After cleanup: ~400MB (significant reduction)

## Prevention Guidelines
1. Always check file sizes before committing
2. Use `git status` and `du -sh` to monitor repository size
3. Store large datasets outside of Git
4. Use descriptive commit messages to track file additions
5. Regular repository size audits

## Scripts Created
- `remove_large_files_from_history.sh`: Script used for this cleanup
- Enhanced `.gitignore` and `.gitattributes` files

This solution ensures the repository complies with GitHub's file size policies while maintaining project functionality. 
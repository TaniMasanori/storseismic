# Development Guidelines

## Large File Management

1. **Never commit large files directly**:
   - Use `.gitignore` for data files > 50MB
   - Use Git LFS for essential large files
   - Document data sources in README

2. **Notebook best practices**:
   - Clear outputs before committing
   - Use lightweight versions for Git
   - Store large notebooks externally

3. **Data storage options**:
   - Git LFS (up to repository limits)
   - External storage (Google Drive, etc.)
   - Data version control (DVC)
   - Cloud storage services

## Memory Optimization

1. **Always use optimized configurations**
2. **Monitor memory usage during training**
3. **Implement gradient accumulation for large effective batch sizes**
4. **Use mixed precision training when possible** 
#!/bin/bash

echo "=== Setting up Git hooks to prevent large file commits ==="

# Pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash

# Check for large files (>50MB)
large_files=$(git diff --cached --name-only | xargs -I {} find {} -size +50M 2>/dev/null || true)

if [ -n "$large_files" ]; then
    echo "Error: Large files detected!"
    echo "$large_files"
    echo "Please add these files to .gitignore or use Git LFS"
    exit 1
fi

# Check for specific file patterns
forbidden_patterns="*.pt *.pth *_original_*.ipynb"
for pattern in $forbidden_patterns; do
    if git diff --cached --name-only | grep -q "$pattern"; then
        echo "Error: Forbidden file pattern detected: $pattern"
        echo "Please check your .gitignore configuration"
        exit 1
    fi
done

echo "Pre-commit check passed"
EOF

chmod +x .git/hooks/pre-commit

echo "=== Git hooks configured ==="
echo "Large files will now be blocked before commit" 
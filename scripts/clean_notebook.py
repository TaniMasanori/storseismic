#!/usr/bin/env python3
"""
Script to clean Jupyter notebooks by removing outputs and large data
"""

import json
import sys
import argparse
from pathlib import Path

def clean_notebook(notebook_path, output_path=None):
    """Clean a Jupyter notebook by removing outputs"""
    
    if output_path is None:
        output_path = notebook_path.with_name(f"{notebook_path.stem}_clean{notebook_path.suffix}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Clean cells
    for cell in notebook.get('cells', []):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
    
    # Remove metadata that might contain large data
    if 'metadata' in notebook:
        notebook['metadata'].pop('widgets', None)
    
    # Write cleaned notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned notebook saved to: {output_path}")
    
    # Show file sizes
    original_size = notebook_path.stat().st_size / (1024**2)  # MB
    cleaned_size = output_path.stat().st_size / (1024**2)    # MB
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Cleaned size: {cleaned_size:.2f} MB")
    print(f"Size reduction: {((original_size - cleaned_size) / original_size * 100):.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean Jupyter notebook outputs')
    parser.add_argument('notebook', help='Path to notebook file')
    parser.add_argument('-o', '--output', help='Output path (default: add _clean suffix)')
    
    args = parser.parse_args()
    
    notebook_path = Path(args.notebook)
    output_path = Path(args.output) if args.output else None
    
    if not notebook_path.exists():
        print(f"Error: {notebook_path} does not exist")
        sys.exit(1)
    
    clean_notebook(notebook_path, output_path) 
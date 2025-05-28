#!/usr/bin/env python3
"""
Script to insert data cutting and resampling cell into nb0_1_original_DAS.ipynb

This script inserts the data processing cell after the data scaling section
and before the data multiplication section.
"""

import json
import os
from datetime import datetime

def load_notebook(notebook_path):
    """Load the notebook JSON file."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    return notebook

def save_notebook(notebook, notebook_path):
    """Save the notebook JSON file."""
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

def create_resample_cell():
    """Create the data cutting and resampling cell."""
    cell_source = [
        "# =============================================================================\n",
        "# DATA CUTTING AND RESAMPLING\n", 
        "# =============================================================================\n",
        "# This cell cuts data after 1024 samples and resamples to 1/4 (256 samples)\n",
        "# for time domain processing.\n",
        "\n",
        "import torch\n",
        "import torch.nn.functional as F\n",
        "\n",
        "def cut_and_resample_data(data_dict, cut_length=1024, target_samples=256):\n",
        "    \"\"\"\n",
        "    Cut data after specified length and resample to target number of samples.\n",
        "    \n",
        "    Args:\n",
        "        data_dict (dict): Dictionary containing seismic data tensors\n",
        "        cut_length (int): Number of samples to keep from original data (default: 1024)\n",
        "        target_samples (int): Target number of samples after resampling (default: 256)\n",
        "        \n",
        "    Returns:\n",
        "        dict: Dictionary with processed data tensors\n",
        "    \"\"\"\n",
        "    print(f\"Starting data cutting and resampling...\")\n",
        "    print(f\"Cut length: {cut_length}, Target samples: {target_samples}\")\n",
        "    \n",
        "    processed_data = {}\n",
        "    \n",
        "    for key, tensor in data_dict.items():\n",
        "        if key == 'index':\n",
        "            # Keep index unchanged\n",
        "            processed_data[key] = tensor.clone()\n",
        "            continue\n",
        "            \n",
        "        print(f\"Processing {key} with original shape: {tensor.shape}\")\n",
        "        \n",
        "        # Clone the tensor to avoid modifying the original\n",
        "        processed_tensor = tensor.clone()\n",
        "        \n",
        "        # Get the time dimension (assuming it's the last dimension)\n",
        "        original_time_samples = processed_tensor.shape[-1]\n",
        "        \n",
        "        # Step 1: Cut data after cut_length samples\n",
        "        if original_time_samples > cut_length:\n",
        "            print(f\"  Cutting {key} from {original_time_samples} to {cut_length} samples\")\n",
        "            processed_tensor = processed_tensor[..., :cut_length]\n",
        "        else:\n",
        "            print(f\"  Warning: {key} has only {original_time_samples} samples, no cutting needed\")\n",
        "        \n",
        "        # Step 2: Resample to target_samples (1/4 downsampling)\n",
        "        current_samples = processed_tensor.shape[-1]\n",
        "        \n",
        "        if current_samples != target_samples:\n",
        "            print(f\"  Resampling {key} from {current_samples} to {target_samples} samples\")\n",
        "            \n",
        "            # Use interpolation for resampling\n",
        "            # Reshape for interpolation: (batch_size * channels, 1, time_samples)\n",
        "            original_shape = processed_tensor.shape\n",
        "            \n",
        "            if len(original_shape) == 3:  # (batch, channels, time)\n",
        "                batch_size, channels, time_samples = original_shape\n",
        "                reshaped = processed_tensor.view(batch_size * channels, 1, time_samples)\n",
        "            elif len(original_shape) == 2:  # (batch, time) \n",
        "                batch_size, time_samples = original_shape\n",
        "                channels = 1\n",
        "                reshaped = processed_tensor.view(batch_size, 1, time_samples)\n",
        "            else:\n",
        "                raise ValueError(f\"Unsupported tensor shape: {original_shape}\")\n",
        "            \n",
        "            # Interpolate to target samples\n",
        "            resampled = F.interpolate(\n",
        "                reshaped.float(), \n",
        "                size=target_samples, \n",
        "                mode='linear', \n",
        "                align_corners=True\n",
        "            )\n",
        "            \n",
        "            # Reshape back to original format\n",
        "            if len(original_shape) == 3:\n",
        "                processed_tensor = resampled.view(batch_size, channels, target_samples)\n",
        "            else:\n",
        "                processed_tensor = resampled.view(batch_size, target_samples)\n",
        "                \n",
        "            # Convert back to original dtype if needed\n",
        "            processed_tensor = processed_tensor.to(tensor.dtype)\n",
        "        \n",
        "        processed_data[key] = processed_tensor\n",
        "        print(f\"  Finished processing {key}, new shape: {processed_tensor.shape}\")\n",
        "    \n",
        "    print(\"Data cutting and resampling completed!\")\n",
        "    return processed_data\n",
        "\n",
        "# Apply cutting and resampling\n",
        "print(\"=\" * 80)\n",
        "print(\"APPLYING DATA CUTTING AND RESAMPLING\")\n",
        "print(\"=\" * 80)\n",
        "print(\"This will cut data after 1024 samples and resample to 256 samples\")\n",
        "print(\"Original time dimension will change from current size to 256\")\n",
        "print()\n",
        "\n",
        "# Check current shapes\n",
        "print(\"Current data shapes:\")\n",
        "print(f\"Training data: {snist_train_mlm['inputs_embeds'].shape}\")\n",
        "print(f\"Testing data: {snist_test_mlm['inputs_embeds'].shape}\")\n",
        "print()\n",
        "\n",
        "# Apply cutting and resampling to training data\n",
        "print(\"Processing training data...\")\n",
        "snist_train_mlm = cut_and_resample_data(snist_train_mlm, cut_length=1024, target_samples=256)\n",
        "print()\n",
        "\n",
        "# Apply cutting and resampling to testing data  \n",
        "print(\"Processing testing data...\")\n",
        "snist_test_mlm = cut_and_resample_data(snist_test_mlm, cut_length=1024, target_samples=256)\n",
        "print()\n",
        "\n",
        "# Show final shapes\n",
        "print(\"Final data shapes:\")\n",
        "print(f\"Training data: {snist_train_mlm['inputs_embeds'].shape}\")\n",
        "print(f\"Testing data: {snist_test_mlm['inputs_embeds'].shape}\")\n",
        "print()\n",
        "\n",
        "print(\"=\" * 80)\n",
        "print(\"DATA CUTTING AND RESAMPLING COMPLETED SUCCESSFULLY!\")\n",
        "print(\"=\" * 80)\n",
        "print(\"You can now proceed with the rest of your data processing pipeline.\")\n",
        "print(\"The time dimension has been reduced from the original size to 256 samples.\")\n"
    ]
    
    markdown_cell = {
        "cell_type": "markdown",
        "id": "data_resample_title",
        "metadata": {},
        "source": [
            "### Cut data and resample to 256 samples\n",
            "\n",
            "Cut data after 1024 samples and resample to 1/4 (256 samples) for time domain processing. This reduces the time dimension while preserving the essential signal characteristics."
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "data_resample_code",
        "metadata": {},
        "outputs": [],
        "source": cell_source
    }
    
    return [markdown_cell, code_cell]

def find_insertion_point(notebook):
    """Find the insertion point after scaling and before multiplication."""
    cells = notebook['cells']
    
    # Look for the scaling section 
    scale_index = -1
    multiply_index = -1
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown' and 'source' in cell:
            source_text = ''.join(cell['source'])
            if 'Scale the data into the range [-1, 1]' in source_text:
                scale_index = i
            elif 'Multiply the data N-times' in source_text:
                multiply_index = i
                break
    
    if scale_index == -1:
        raise ValueError("Could not find scaling section")
    if multiply_index == -1:
        raise ValueError("Could not find multiplication section")
    
    # Find the end of the scaling section (look for next code cell)
    insertion_point = multiply_index
    for i in range(scale_index + 1, multiply_index):
        if cells[i]['cell_type'] == 'code':
            insertion_point = i + 1
    
    return insertion_point

def insert_resample_cells(notebook_path):
    """Insert the resampling cells into the notebook."""
    print(f"Loading notebook: {notebook_path}")
    
    # Create backup
    backup_path = notebook_path.replace('.ipynb', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.ipynb')
    print(f"Creating backup: {backup_path}")
    
    # Load notebook
    notebook = load_notebook(notebook_path)
    
    # Save backup
    save_notebook(notebook, backup_path)
    
    # Find insertion point
    insertion_point = find_insertion_point(notebook)
    print(f"Inserting cells at position: {insertion_point}")
    
    # Create new cells
    new_cells = create_resample_cell()
    
    # Insert cells
    for i, cell in enumerate(new_cells):
        notebook['cells'].insert(insertion_point + i, cell)
    
    # Save modified notebook
    save_notebook(notebook, notebook_path)
    
    print(f"Successfully inserted {len(new_cells)} cells into notebook")
    print(f"Original notebook backed up to: {backup_path}")
    print("Data cutting and resampling cells have been added!")

if __name__ == "__main__":
    notebook_path = "/home/masa/storseismic/nb0_1_original_DAS.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        exit(1)
    
    try:
        insert_resample_cells(notebook_path)
        print("\n" + "="*60)
        print("SUCCESS: Data cutting and resampling cells added to notebook!")
        print("="*60)
        print("You can now open the notebook and run the new cells.")
        print("The cells are located after the data scaling section.")
    except Exception as e:
        print(f"Error: {e}")
        exit(1) 
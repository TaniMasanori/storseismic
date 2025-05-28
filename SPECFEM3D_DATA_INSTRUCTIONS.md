# Instructions for Converting SPECFEM3D DAS Data to StorSeismic Format

These instructions explain how to convert the DAS data produced in the `/home/masa/DAS_GenericCable/examples/04-DAS-Forward-SPECFEM3D.ipynb` notebook to a format that can be used in the `/home/masa/storseismic/nb0_1_data_prep_pretrain.ipynb` notebook.

## Steps for Data Conversion

### 1. Add Conversion Cells to SPECFEM3D Notebook

Add the following code cells to the end of the `04-DAS-Forward-SPECFEM3D.ipynb` notebook (after the waveform plotting cells):

The code cells are available in the `/home/masa/storseismic/specfem3d_data_converter.py` script. You can either:
- Run the script and copy the output: `python /home/masa/storseismic/specfem3d_data_converter.py`
- Open the script and copy each cell directly

The script contains 4 cells that need to be added to the SPECFEM3D notebook:
1. Cell 1: Converts DAS data to PyTorch tensor and normalizes to [-1, 1]
2. Cell 2: Creates batches of data in the format expected by StorSeismic
3. Cell 3: Saves the prepared data
4. Cell 4: Visualizes samples of the converted data

### 2. Run the SPECFEM3D Notebook

Run the entire notebook, including the new cells. This will:
- Process the DAS data
- Convert it to StorSeismic format
- Save it to `/home/masa/storseismic/data/specfem3d_das/specfem_data.pt`

### 3. Modify the StorSeismic Preprocessing Notebook

Add the code from the `storseismic_load_cell` (in the `specfem3d_data_converter.py` script) to the `/home/masa/storseismic/nb0_1_data_prep_pretrain.ipynb` notebook.

The best place to add this code is after the SNIST data is loaded, but before any data processing steps. Look for the cell after:
```python
snist_train = SNIST('./', train=True, download=True) # Training data
snist_test = SNIST('./', train=False, download=True, noise=0) # Testing data
```

The load cell provides two options:
1. Use the SPECFEM3D DAS data as a separate dataset
2. Combine the SPECFEM3D DAS data with the existing SNIST data

Choose the option that best suits your needs.

## Data Format Details

The converted data follows the StorSeismic format:
- `specfem_data['inputs_embeds']`: Input data tensor with shape [batch, channels, time]
- `specfem_data['labels']`: Same as inputs (for pretraining)
- `specfem_data['mask_label']`: Tensor of zeros with same shape as inputs (will be modified during masking)
- `specfem_data['index']`: Tensor of indices

The data is normalized to the range [-1, 1] to match the scaling used in the StorSeismic preprocessing.

## Notes on Segmentation Parameters

The conversion uses these parameters:
- `segment_length = 20`: Number of channels per segment
- 50% overlap between segments

You may adjust these parameters in Cell 2 to better suit your dataset's characteristics.

## Troubleshooting

1. If you encounter memory issues, try reducing the `segment_length` parameter
2. If the shapes don't match when concatenating with SNIST data, check if the time dimensions are compatible
3. Make sure the `os` module is imported in the StorSeismic notebook before loading the SPECFEM3D data 
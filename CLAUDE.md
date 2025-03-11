# StorSeismic Development Guidelines

## Environment Setup
- Install dependencies: `pip install -r requirements.txt`
- Install package in dev mode: `pip install -e .`

## Running Code
- Run notebooks with: `jupyter notebook`
- Data preparation: `nb0_1_data_prep_pretrain.ipynb`, `nb0_2_data_prep_finetune.ipynb`
- Pre-training: `nb1_pretraining.ipynb`
- Fine-tuning: `nb2_1_finetuning_denoising.ipynb`, `nb2_2_finetuning_velpred.ipynb`

## Code Style Guide
- **Classes**: PascalCase (e.g., `BertEmbeddings`)
- **Functions/Variables**: snake_case (e.g., `run_pretraining`, `batch_size`)
- **Imports**: Group by: 1) PyTorch/transformers 2) Other libraries 3) Standard libs 4) Local modules
- **Model Architecture**: Follow transformer-based design with custom seismic components
- **Error Handling**: Use validation checks for dimensions and configurations
- **Documentation**: Include comments for complex operations, especially custom attention mechanisms
- **Tensors**: Pay careful attention to tensor dimensions, particularly for batch processing

## Model Training
- Support both CPU and GPU: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Use early stopping: `from storseismic.pytorchtools import EarlyStopping`
- Save checkpoints in appropriate results directory
# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

## Dataset

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup

### Conda

Make sure [conda](https://docs.anaconda.com/miniconda/miniconda-install/) has been installed properly

1. Create a new conda environment and activate it:

   ```bash
   conda create --name equihgnn python=3.10
   conda activate equihgnn
   ```

2. Install specific `torch` and `torch-cluster` version. We use torch `2.2.1` with CUDA `12.1` support:

   ```bash
   pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
   ```

3. Install other dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the project as an editable module:
   ```bash
   pip install -e .
   ```

## Development

Please ensure you run the code formatter before committing or opening a pull request. This helps maintain clean and consistent code throughout the project.

1. Install pre-commit

   ```bash
   pip install pre-commit
   ```

2. Install Git Hook
   ```bash
   pre-commit install
   ```

## Run

Visit the [./scripts](./scripts) directory to customize parameters such as: model name, dataset type, hyperparameters (learning rate, epochs, batch size, etc.)

## Results


## New Installation
```
conda create --prefix ./.venv python=3.10
conda activate ./.venv

uv pip install -e .

uv lock
```
bash setup_env.sh 
equihgnn-train configs/ehgnn.yaml 
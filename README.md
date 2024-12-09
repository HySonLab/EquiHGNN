# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

## Dataset

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup

Make sure [conda](https://docs.anaconda.com/miniconda/miniconda-install/) is properly installed.

1. Create a new conda environment and activate it:
   ```bash
   conda create --prefix ./.venv python=3.10
   conda activate ./.venv
   ```

2. Install the project and dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```

3. Run the environment setup script for PyG dependencies:
   ```bash
   bash setup_env.sh
   ```

## Development

Please ensure you run the code formatter before committing or opening a pull request. This helps maintain clean and consistent code throughout the project.

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install Git Hook:
   ```bash
   pre-commit install
   ```

## Run

1. Modify the `.env` file for environment configuration.

2. Customize parameters such as model name, dataset type, and hyperparameters (learning rate, epochs, batch size, etc.) in the [./configs](./configs) directory.

3. Train the model:
   ```bash
   equihgnn-train configs/ehgnn.yaml
   ```


## Results

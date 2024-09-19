# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

## Dataset

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup

### Docker

Make sure the following tools have been installed properly:

- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- CMake

### Conda

Make sure [conda](https://docs.anaconda.com/miniconda/miniconda-install/) has been installed properly

1. Create a new conda environment and activate it:

   ```shell
   conda create --name equihgnn python=3.10
   conda activate equihgnn
   ```

2. Install specific `torch` and `torch-cluster` version. We use torch `2.2.1` with CUDA `12.1` support:

   ```shell
   pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
   pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
   ```

3. Install other dependencies:

   ```shell
   pip install -r requirements.txt
   ```

4. Install the project as an editable module:
   ```shell
   pip install -e .
   ```

## Development

Please ensure you run the code formatter before committing or opening a pull request. This helps maintain clean and consistent code throughout the project.

1. Install pre-commit

   ```shell
   pip install pre-commit
   ```

2. Install Git Hook
   ```shell
   pre-commit install`
   ```

## Run

Visit the [./scripts](./scripts) directory to customize parameters such as: model name, dataset type, hyperparameters (learning rate, epochs, batch size, etc.)

## Run on Docker

To build the Docker image:
`shell make build `

To clean the unused containers:
`shell make clean `

## Train

To use the Comet logging:

```bash
export COMET_API_KEY=<YOUR-API-KEY>
```

### OPV dataset

- To train and evaluate the _Molecular Hypergraph Neural Network_:

  - On a specific task:

    ```bash
    # molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
    # polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
    make train_opv $TASK
    ```

  - On all tasks:
    ```bash
    make train_opv_all
    ```

- To train and evaluate the _Equivariant Molecular Hypergraph Neural Network_:

  - On a specific task:

    ```bash
    # molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
    # polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
    make train_opv3d $TASK
    ```

  - On all tasks:
    ```bash
    make train_opv3d_all
    ```

### QM9 dataset

- To train and evaluate the _Molecular Hypergraph Neural Network_:

  - On a specific task:

    ```bash
    # 0-alpha, 1-gap, 2-homo, 3-lumo, 4-mu, 5-cv
    make train_qm9 $TASK
    ```

  - On all tasks:
    ```bash
    make train_qm9_all
    ```

- To train and evaluate the _Equivariant Molecular Hypergraph Neural Network_:

  - On a specific task:

    ```bash
    # 0-alpha, 1-gap, 2-homo, 3-lumo, 4-mu, 5-cv
    make train_qm9_3d $TASK
    ```

  - On all tasks:
    ```bash
    make train_qm9_3d_all
    ```

## Results

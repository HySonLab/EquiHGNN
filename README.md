# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

EquiHGNN is a deep learning framework for molecular property prediction using hypergraph neural networks with rotational equivariance.

## Datasets

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup

First, create and activate a Conda environment:

```shell
conda create --name equihgnn python=3.10
conda activate equihgnn
make
```

## Train the Model

Training parameters, including model type, dataset selection, and hyperparameters, are configurable within the `./scripts` directory. A flexible interface allows easy model selection using the `--method` flag. The following models are supported:

- `mhnnm`: Molecular Hypergraph Neural Network (baseline).
- `egnn_equihnns`: Equivariant Graph Neural Network (EGNN) integration for geometric feature extraction.
- `se3_transformer_equihnns`: SE(3) Transformer integration for geometric feature extraction.
- `equiformer_equihnns`: Equiformer integration for geometric feature extraction.
- `visnet_equihnns`: VisNet integration for geometric feature extraction.
- `faformer_equihnns`: Frame Averaging Transformer (FAFormer) integration for geometric feature extraction.

(Optional) Enable Comet Logging to track experiments with Comet, set up your API key:

```shell
export COMET_API_KEY=<YOUR-API-KEY>
```

### OPV Dataset Tasks

OPV dataset task IDs:

- **Molecular**: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
- **Polymer**: 4-homo, 5-lumo, 6-gap, 7-optical_lumo

Train without geometric information:

```shell
bash scripts/run_opv.sh $TASK_ID
```

Train with geometric information:

```shell
bash scripts/run_opv_3d.sh $TASK_ID
```

### QM9 Dataset Tasks

QM9 dataset task IDs: 0-alpha, 1-gap, 2-homo, 3-lumo, 4-mu, 5-cv

Train without geometric information:

```shell
bash scripts/run_qm9.sh $TASK_ID
```

Train with geometric information:

```shell
bash scripts/run_qm9_3d.sh $TASK_ID
```

## Training with Docker

Build the Docker image:

```shell
docker build -t equihgnn .
```

Run training inside a Docker container:

```shell
docker run \
  --gpus all \
  -v ./datasets:/module/datasets \
  -v ./logs:/module/logs \
  -v ./scripts:/module/scripts \
  -e COMET_API_KEY=$(COMET_API_KEY) \
  equihgnn bash scripts/*.sh $TASK_ID
```

## Acknowledgements

This project utilizes code and inspiration from the following open-source repositories:

- **MHNN Baseline:** [schwallergroup/mhnn](https://github.com/schwallergroup/mhnn)
- **EGNNs:** [lucidrains/egnn-pytorch](https://github.com/lucidrains/egnn-pytorch)
- **SE(3) Transformers:** [lucidrains/se3-transformer-pytorch](https://github.com/lucidrains/se3-transformer-pytorch)
- **Equiformer:** [lucidrains/equiformer-pytorch](https://github.com/lucidrains/equiformer-pytorch)
- **Frame Averaging Transformer:** [Graph-and-Geometric-Learning/Frame-Averaging-Transformer](https://github.com/Graph-and-Geometric-Learning/Frame-Averaging-Transformer)
- **VisNet:** [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric/blob/2f1e4f2e666db65056d001650488be9b31f8dd0f/torch_geometric/nn/models/visnet.py)

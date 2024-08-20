# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

## Dataset

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup


### Conda
Create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

### Docker

To build the image
```bash
docker build -t equihgnn .
```
## Models


## Run Code

To train and evaluate the models, use the following commands:

1. **Training**:

```bash
# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
bash scripts/run_opv.sh $TASK
```

Train with Docker:
```bash
docker run \
    --gpus all \
    --name equihgnn_container \
    -v $(pwd)/datasets:/equihgnn/datasets \
    -v $(pwd)/logs:/equihgnn/logs \
    equihgnn bash scripts/run_opv.sh <TASK-ID>
```

## Results


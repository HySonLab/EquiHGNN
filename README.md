# Rotationally Equivariant Hypergraph Neural Networks (EquiHGNN)

## Dataset

This project currently utilizes two main datasets:

- **OPV Dataset**: The Organic Photovoltaic (OPV) dataset contains molecular structures and their corresponding photovoltaic properties.
- **QM9 Dataset**: The QM9 dataset consists of small molecules with geometric, energetic, electronic, and thermodynamic properties.

## Setup

Make sure the following tools have been installed properly:

* [Docker](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* CMake

The data will be downloaded automatically.

## Run

### Build
To build the Docker image:
```shell
make build
```

### Clean
To clean the unused containers:
```shell
make clean
```

## Train

To use the Comet logging:

```bash
export COMET_API_KEY=<YOUR-API-KEY>
```

### OPV dataset
* To train and evaluate the *Molecular Hypergraph Neural Network*:
    * On a specific task:
        ```bash
        # molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
        # polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
        make train_opv $TASK
        ```

    * On all tasks:
        ```bash
        make train_opv_all
        ```

* To train and evaluate the *Equivariant Molecular Hypergraph Neural Network*:
    * On a specific task:
        ```bash
        # molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
        # polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
        make train_opv3d $TASK
        ```

    * On all tasks:
        ```bash
        make train_opv3d_all
        ```

### QM9 dataset

## Results


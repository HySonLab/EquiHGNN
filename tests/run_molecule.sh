#! /bin/bash
set -euxo pipefail

# 0-dipole x, 1-dipole y, 2-dipole z, 3-homo,
# 4-lumo, 5-homolumogap, 6-scf-energy

TASK=$1

python -u main.py \
    --method gin \
    --data molecule_g \
    --data_dir datasets/molecule3d \
    --runs 1 \
    --target $TASK \
    --All_num_layers 3 \
    --MLP1_num_layers 2 \
    --MLP2_num_layers 2 \
    --MLP3_num_layers 2 \
    --MLP4_num_layers 2 \
    --output_num_layers 3 \
    --MLP_hidden 256 \
    --output_hidden 128 \
    --aggregate mean \
    --lr 0.0001 \
    --min_lr 0.0001 \
    --wd 0 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 1 \
    --epochs 400 \
    --debug

#! /bin/bash
set -euxo pipefail

# 0-alpha, 1-gap, 2-homo, 3-lumo, 4-mu, 5-cv
TASK=$1

# Use default settings
python -u main.py \
    --method mhnnm \
    --data_dir datasets/qm9 \
    --data qm9_hg \
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

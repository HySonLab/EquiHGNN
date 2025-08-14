#! /bin/bash
set -euxo pipefail

# 0-aspirin, 1-benzene, 2-ethanol, 3-malonaldehyde,
# 4-naphthalene, 5-salicylic acid, 6-toluene, 7-u0, 8-uracil,

TASK=$1

# Use default settings
python -u main.py \
    --method egnn \
    --data_dir datasets/md17 \
    --data md17_g_3d \
    --target $TASK \
    --output_num_layers 3 \
    --MLP_hidden 256 \
    --output_hidden 128 \
    --aggregate mean \
    --lr 0.0001 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 768 \
    --epochs 400 \

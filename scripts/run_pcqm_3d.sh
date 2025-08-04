#! /bin/bash
set -euxo pipefail

python -u main.py \
    --method egnn_equihnns \
    --data pcqm_hg_3d \
    --data_dir datasets/pcqm \
    --output_num_layers 3 \
    --MLP_hidden 256 \
    --output_hidden 128 \
    --aggregate mean \
    --lr 0.0001 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 768 \
    --epochs 400

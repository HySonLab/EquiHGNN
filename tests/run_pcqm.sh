#! /bin/bash
set -euxo pipefail

python -u main.py \
    --method gin \
    --data pcqm_g \
    --data_dir datasets/pcqm \
    --runs 1 \
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

#! /bin/bash
set -euxo pipefail

# 0-mu, 1-alpha, 2-homo, 3-lumo, 4-gap, 5-r2, 6-zpve, 7-u0, 8-u298, 
# 9-h298, 10-g298, 11-cv, 12-u0_atom, 13-u298_atom, 14-h298_atom, 15-g298_atom

TASK=$1

# Use default settings
python -u main.py \
    --method mhnnm \
    --data_dir datasets/qm9 \
    --data qm9_hg \
    --use_ring \
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
    --wd 0 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 768 \
    --epochs 400 \

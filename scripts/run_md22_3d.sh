#! /bin/bash
set -euxo pipefail

# 0-Ac_Ala3_NHMe, 1-Docosahexaenoic_acid, 2-Stachyose, 3-AT_AT,
# 4-AT_AT_CG_CG, 5-Buckyball_catcher, 6-Double_walled_nanotube

TASK=$1

# Use default settings
python -u main.py \
    --method egnn \
    --data_dir datasets/md22 \
    --data md22_g_3d \
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

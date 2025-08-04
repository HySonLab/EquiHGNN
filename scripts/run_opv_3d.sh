#! /bin/bash
set -euxo pipefail

# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
TASK=$1

python -u main.py \
    --method egnn_equihnns \
    --data opv_hg_3d \
    --target $TASK \
    --output_num_layers 3 \
    --MLP_hidden 256 \
    --output_hidden 128 \
    --aggregate mean \
    --lr 0.0001 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 768 \
    --epochs 400

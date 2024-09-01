#! /bin/bash
set -euxo pipefail

# 0-gap, 1-homo, 2-lumo
TASK=$1

# Use default settings
python -u train_opv.py \
    --target $TASK
#!/bin/bash

echo "Installing PyG dependencies..."
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
echo "PyG dependencies installed successfully."

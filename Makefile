build:
	pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
	pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
	pip install -r requirements.txt
	pip install -e .

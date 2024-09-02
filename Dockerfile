ARG BASE_IMAGE=pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE}

WORKDIR /equihgnn

COPY ./requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

COPY ./data /equihgnn/data
COPY ./models /equihgnn/models
COPY ./scripts /equihgnn/scripts
COPY ./tests /equihgnn/tests
COPY ./utils /equihgnn/utils
COPY ./common /equihgnn/common
COPY ./main.py /equihgnn/main.py

COPY pyproject.toml .
RUN pip install -e .
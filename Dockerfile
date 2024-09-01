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
COPY ./train_opv.py /equihgnn/train_opv.py
COPY ./train_qm9.py /equihgnn/train_qm9.py

COPY pyproject.toml .
RUN pip install -e .
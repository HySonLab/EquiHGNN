ARG BASE_IMAGE=pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

FROM ${BASE_IMAGE}

WORKDIR /module

COPY ./requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

COPY ./equihgnn /module/equihgnn
COPY ./main.py /module/main.py

COPY pyproject.toml .
RUN pip install -e .

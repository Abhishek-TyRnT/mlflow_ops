FROM tensorflow/tensorflow:2.14.0rc0-gpu

ARG USERNAME=mlflow
ARG UID=1001
WORKDIR /opt
SHELL ["/bin/bash", "-c"]

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install sudo && \
    useradd -m -u $UID $USERNAME && \
    usermod -aG sudo mlflow

RUN mkdir /logs && \ 
    chgrp -R 1001 /logs



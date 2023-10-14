FROM tensorflow/tensorflow:2.14.0-gpu

ARG USERNAME=mlflow
ARG UID=1001
ARG PASS=pass
WORKDIR /opt
SHELL ["/bin/bash", "-c"]

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get install sudo && \
    useradd -m -u $UID $USERNAME && \
    echo "${USERNAME}:${PASS}" | chpasswd && \
    usermod -aG sudo $USERNAME 

RUN mkdir /logs && \ 
    mkdir /datasets && \
    chgrp -R $UID /datasets && \
    chgrp -R $UID /logs && \
    chmod 770 /logs && \
    chmod 770 /datasets

USER $USERNAME
ENV HOME=/home/$USERNAME


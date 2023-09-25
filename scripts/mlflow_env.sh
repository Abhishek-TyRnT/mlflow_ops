#!/bin/bash

HOME_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
container_name=mlflow

build()
{
    #echo $HOME_DIR
    docker build -t mlflow:latest -f $HOME_DIR/envs/Dockerfile --build-arg USERNAME=mlflow --build-arg UID=$(id -u) $HOME_DIR/envs
    docker volume create logs
}

run()
{
    
    result=$(docker ps -a)
    #echo $result
    if [[ $result == *"$container_name"* ]]; 
    then
        docker start -i $container_name
    else
        docker run -it -v logs:/logs \
            -v $HOME_DIR:/home/mlflow/project \
            -w /home/mlflow/project \
            --user $(id -u):$(id -g) \
            --name mlflow \
            mlflow:latest
    fi


}

clean()
{
    result=$(docker ps -a)
    if [[ $result == *"$container_name"* ]]; 
    then
        docker rm $container_name
    fi
}

$1
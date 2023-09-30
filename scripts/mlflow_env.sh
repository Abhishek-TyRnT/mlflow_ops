#!/bin/bash

HOME_DIR=$(dirname "$(dirname "$(readlink -f "$0")")")
container_name=mlflow_container

build()
{
    #echo $HOME_DIR
    docker build \
        -t mlflow:latest \
        -f $HOME_DIR/envs/Dockerfile \
        --build-arg USERNAME=mlflow \
        --build-arg UID=$(id -u)\
        --build-arg PASS=$1\
         $HOME_DIR/envs
    docker volume create logs
    docker volume create datasets
}

run()
{
    
    result=$(docker ps -a)
    if [[ $result == *"$container_name"* ]]; 
    then
        docker start -i $container_name
    else
        docker run -it -v logs:/logs \
            -v datasets:/datasets \
            -v $HOME_DIR:/home/mlflow/project \
            -v $HOME/.kaggle:/home/mlflow/.kaggle \
            --user $(id -u) \
            -w /home/mlflow/project \
            --name $container_name \
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

$1 $2
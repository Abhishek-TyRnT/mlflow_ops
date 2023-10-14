#!/bin/bash

cd /logs
experiment_list=$( mlflow experiments search -v active_only)


if [[ $experiment_list==*"$EXPERIMENT_NAME"* ]]
then
    echo "RUN already created"
else
    mlflow experiments create -n $EXPERIMENT_NAME 
fi

cd /datasets

if [ -f $DATASET_NAME ]
then
    echo "Dataset already present"
else
    kaggle datasets download -d $KAGGLE_NAME
    unzip $DATASET_NAME
fi

cd /logs
mlflow ui --host 0.0.0.0 -p 8080
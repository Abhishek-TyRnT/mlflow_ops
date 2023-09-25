#!/bin/bash

def build()
{
    if [$1 == "windows"]
    then
        docker build -t  mlflow:latest envs/
        
}
#!/bin/bash

docker build --no-cache -t serving_image -f Dockerfile.serving .
docker build --no-cache -t frontend_image -f Dockerfile.frontend .
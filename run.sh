#!/bin/bash

docker run -d -e COMET_API_KEY=${COMET_API_KEY} -p 8000:8000 serving_image
docker run -d -p 8050:8050 frontend_image
#!/bin/bash

docker run -e COMET_API_KEY=${COMET_API_KEY} -p 8000:8000 serving_image
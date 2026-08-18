#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

hf download \
    --repo-type model \
    --repo-id adithyamurali/GraspGenXModel \
    --local-dir /home/robot/hewu/model_zoo/GraspGenXModel

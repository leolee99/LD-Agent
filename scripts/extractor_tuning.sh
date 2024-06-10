#! /usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

DATASET_PATH=dataset/persona_extraction
CONFIG_PATH=Trainer/configs/extractor.yaml
BASE_MODEL_PATH=THUDM/chatglm3-6b

python -u Trainer/lora_tune.py $DATASET_PATH $BASE_MODEL_PATH $CONFIG_PATH

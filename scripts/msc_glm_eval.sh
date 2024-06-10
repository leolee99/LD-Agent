#! /usr/bin/env bash
DATASET_PATH=dataset/MSC

MODEL_PATH=THUDM/chatglm3-6b
SUMMARIZER=logs/models/summarizer
EXTRACTOR=logs/models/extractor
GENERATOR=logs/models/generator

DATESTR=`date +%Y%m%d-%H%M%S`
LOG_NAME=MSC_EVAL-${DATESTR}.log


python -m pdb main.py --dataset msc --data_path ${DATASET_PATH} --data_name sequential_msc.json \
        --client chatglm --model ${MODEL_PATH} \
        --summary_model ${SUMMARIZER} --persona_model ${EXTRACTOR} --generation_model ${GENERATOR} \
        --usr_name SPEAKER_1 --agent_name SPEAKER_2 \
        --test_num 501 --gpus 0


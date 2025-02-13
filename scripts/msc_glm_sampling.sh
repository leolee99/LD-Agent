#! /usr/bin/env bash
DATASET_PATH=dataset/MSC

MODEL_PATH=THUDM/chatglm3-6b
SUMMARIZER=logs/models/summarizer
EXTRACTOR=logs/models/extractor
GENERATOR=logs/models/generator

SAMPLING_PATH=logs/sampling/
SAMPLING_FILE=test.json

DATESTR=`date +%Y%m%d-%H%M%S`
LOG_NAME=MSC_SAMPLING-${DATESTR}.log


python -u main.py --dataset msc --data_path ${DATASET_PATH} --data_name sequential_test.json \
        --sampling --sampling_step 10 --sampling_path ${SAMPLING_PATH} --sampling_file_name ${SAMPLING_FILE}\
        --client chatglm --model ${MODEL_PATH} \
        --summary_model ${SUMMARIZER} --persona_model ${EXTRACTOR} --generation_model ${GENERATOR} \
        --usr_name SPEAKER_1 --agent_name SPEAKER_2 --min_session 1 --max_session 5 --gpus 0


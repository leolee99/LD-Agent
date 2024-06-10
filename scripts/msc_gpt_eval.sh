#! /usr/bin/env bash
DATASET_PATH=dataset/MSC

MODEL_PATH=gpt-3.5-turbo-1106
API_KEY=your_openai_api_key

DATESTR=`date +%Y%m%d-%H%M%S`
LOG_NAME=MSC_EVAL-${DATESTR}.log

python -u main.py --dataset msc --data_path ${DATASET_PATH} --data_name sequential_msc.json\
        --client chatgpt --model ${MODEL_PATH} --api_key ${API_KEY}\
        --usr_name SPEAKER_1 --agent_name SPEAKER_2 \
        --test_num 501


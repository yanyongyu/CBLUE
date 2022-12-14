#!/usr/bin/env bash
DATA_DIR="CBLUEDatasets"

TASK_NAME="qic"
MODEL_TYPE="domain-enhance"
MODEL_DIR="data/model_data"
MODEL_NAME="domain-enhance-101000"
OUTPUT_DIR="data/output"
RESULT_OUTPUT_DIR="data/result_output/${MODEL_NAME}"

MAX_LENGTH=50

SEED=${SEED:-2022}

echo "Start running"

if [ $# == 0 ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_dir=${MODEL_DIR} \
        --model_name=${MODEL_NAME} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_train \
        --max_length=${MAX_LENGTH} \
        --train_batch_size=128 \
        --eval_batch_size=128 \
        --learning_rate=2e-5 \
        --epochs=3 \
        --warmup_proportion=0.1 \
        --earlystop_patience=100 \
        --logging_steps=20 \
        --save_steps=20 \
        --seed=${SEED}
elif [ $1 == "predict" ]; then
    python baselines/run_classifier.py \
        --data_dir=${DATA_DIR} \
        --model_type=${MODEL_TYPE} \
        --model_name=${MODEL_NAME} \
        --model_dir=${MODEL_DIR} \
        --task_name=${TASK_NAME} \
        --output_dir=${OUTPUT_DIR} \
        --result_output_dir=${RESULT_OUTPUT_DIR} \
        --do_predict \
        --max_length=${MAX_LENGTH} \
        --eval_batch_size=128 \
        --seed=${SEED}
fi

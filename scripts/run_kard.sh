#!/bin/sh

set -e
set -x

gpus=$1
batchsize=$2
modelsize=$3
dataset=$4
train_samples=${5:-"-1"}  # New parameter with default value -1
port=$(shuf -i 29400-29700 -n 1)

deepspeed --include=localhost:$gpus --master_port $port trainers/run_seq2seq_deepspeed.py \
    --model_id google/flan-t5-$modelsize \
    --dataset_path preprocessed_data/$dataset-cot-wikipedia \
    --epochs 3 \
    --per_device_train_batch_size $batchsize \
    --per_device_eval_batch_size $batchsize \
    --lr 1e-4 \
    --deepspeed "configs/ds_flan_t5_z3_config.json" \
    --output_dir ./save/flan-t5-$modelsize/$dataset/kard_wikipedia \
    --train_samples $train_samples
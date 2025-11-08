#!/usr/bin/env bash
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) epochs=$2; shift 2 ;;
        --batch_size) batch_size=$2; shift 2 ;;
        --learning_rate) learning_rate=$2; shift 2 ;;
        --model_name) model_name=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

epochs=${epochs:-10}
batch_size=${batch_size:-32}
learning_rate=${learning_rate:-0.001}
model_name=${model_name:-"resnet18"}

echo "Launching training: epochs=$epochs, batch_size=$batch_size, learning_rate=$learning_rate, model_name=$model_name"

python train.py \
    --epochs $epochs \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --model_name $model_name

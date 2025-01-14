#!/bin/bash

fractions=(0.2 0.4 0.6 0.8)

# 训练参数
output_dir_base="output/Llama"
train_epoch=1
learning_rate=3e-4
train_batch_size=1
source_length=128
use_bf16="--bf16"  

# 循环训练不同大小的模型
for frac in "${fractions[@]}"; do
    python finetune_llama_frac.py \
        --output_dir ${output_dir_base} \
        --model_name "meta-llama/Llama-2-7b-chat-hf"  \
        --train_epoch ${train_epoch} \
        --learning_rate ${learning_rate} \
        --train_batch_size ${train_batch_size} \
        --source_length ${source_length} \
        --fraction $frac \
        ${use_bf16}
done

#!/bin/bash


models=(
    "meta-llama/Llama-2-7b-chat-hf" 
    "meta-llama/Llama-2-13b-chat-hf" 
    "meta-llama/Llama-2-70b-chat-hf"
    )


output_dir_base="output/Llama"
train_epoch=1
learning_rate=3e-4
train_batch_size=16
source_length=128
use_bf16="--bf16"  


for model_name in "${models[@]}"; do
    echo "Fine-tuning model: ${model_name}"
    
    python finetune_llama.py \
        --output_dir ${output_dir_base} \
        --model_name ${model_name} \
        --train_epoch ${train_epoch} \
        --learning_rate ${learning_rate} \
        --train_batch_size ${train_batch_size} \
        --source_length ${source_length} \
        ${use_bf16}

    echo "Finished fine-tuning ${model_name}. Results saved to ${output_dir_base}"
done

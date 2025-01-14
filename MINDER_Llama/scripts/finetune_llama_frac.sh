#!/bin/bash

# Define fractions to iterate over
fractions=(0.2 0.4 0.6 0.8)

output_dir="output/"

for frac in "${fractions[@]}"; do
    # Set the specific output directory for each fraction
    
    # Run the training script with the current fraction
    python finetune_llama_frac.py \
        --data_path ../data/training \
        --output_dir $output_dir \
        --model_name meta-llama/Llama-2-7b-chat-hf \
        --train_epoch 1 \
        --learning_rate 3e-4 \
        --train_batch_size 4 \
        --source_length 128 \
        --save_total_limit 1 \
        --logging_steps 500 \
        --bf16 \
        --fraction $frac
done

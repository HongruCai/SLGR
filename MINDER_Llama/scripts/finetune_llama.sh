python finetune_llama.py \
    --data_path ../data/training \
    --output_dir output/ \
    --model_name meta-llama/Llama-2-7b-chat-hf \
    --train_epoch 1 \
    --learning_rate 3e-4 \
    --train_batch_size 64 \
    --source_length 128 \
    --save_total_limit 1 \
    --logging_steps 500 \
    --bf16 \

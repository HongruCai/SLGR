python train_t5.py \
    --output_dir output/T5 \
    --model_name t5-base \
    --train_epoch 1 \
    --learning_rate 1e-3 \
    --train_batch_size 128 \
    --source_length 64 \
    --bf16 \
    --lora


models=(
    "t5-small"
    "t5-base"
    "t5-large"
    "t5-3b"
    "t5-11b"
)

batch_size=256 
gpus=4 
learning_rate=3e-5
num_epochs=1


for model in "${models[@]}"; do
    echo "Training model: $model"
    
    python -m torch.distributed.launch --nproc_per_node=$gpus T5_sft.py \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --num_epochs $num_epochs \
        --model_size $model
    
    if [ $? -ne 0 ]; then
        echo "Training failed for model: $model"
        exit 1
    fi
    
    echo "Finished training model: $model"
done



python scripts/merge_save.py \
    --base_model_name t5-small \
    --lora_model_path output/t5-small/final_ckpt \
    --output_dir output/t5-small
rm output/t5-small/added_tokens.json

python scripts/merge_save.py \
    --base_model_name t5-base \
    --lora_model_path output/t5-base/final_ckpt \
    --output_dir output/t5-base
rm output/t5-base/added_tokens.json

python scripts/merge_save.py \
    --base_model_name t5-large \
    --lora_model_path output/t5-large/final_ckpt \
    --output_dir output/t5-large
rm output/t5-large/added_tokens.json

python scripts/merge_save.py \
    --base_model_name t5-3b \
    --lora_model_path output/t5-3b/final_ckpt \
    --output_dir output/t5-3b
rm output/t5-3b/added_tokens.json

python scripts/merge_save.py \
    --base_model_name t5-11b \
    --lora_model_path output/t5-11b/final_ckpt \
    --output_dir output/t5-11b
rm output/t5-11b/added_tokens.json
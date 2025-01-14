
models=(
    "t5-small"
    "t5-base"
    "t5-large"
    "t5-3b"
    "t5-11b"
)

# 训练参数
output_dir_base="output/T5"
train_epoch=1
learning_rate=1e-3
train_batch_size=256
source_length=64
use_bf16="--bf16"  # 使用 bf16 模式
use_lora="--lora"  # 使用 LoRA 训练

# 循环训练不同大小的模型
for model_name in "${models[@]}"; do
    echo "Training model: ${model_name}"
    
    python train_t5.py \
        --output_dir ${output_dir_base} \
        --model_name ${model_name} \
        --train_epoch ${train_epoch} \
        --learning_rate ${learning_rate} \
        --train_batch_size ${train_batch_size} \
        --source_length ${source_length} \
        ${use_bf16} \
        ${use_lora}

    echo "Finished training ${model_name}. Results saved to ${output_dir_base}"
done
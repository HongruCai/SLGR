
# 训练

models=(
    "meta-llama/Llama-2-70b-chat-hf" 
    "meta-llama/Llama-2-13b-chat-hf" 
    "meta-llama/Llama-2-7b-chat-hf"
)


data_path="../data/training"
output_dir="output/"
train_epoch=1
learning_rate=3e-4
train_batch_size=256 #可能爆显存，可以根据情况调整，不影响结果
source_length=256
save_total_limit=1
logging_steps=500

for model_name in "${models[@]}"; do
    echo "Starting fine-tuning for model: $model_name"

    python finetune_llama.py \
        --data_path "$data_path" \
        --output_dir "$output_dir" \
        --model_name "$model_name" \
        --train_epoch "$train_epoch" \
        --learning_rate "$learning_rate" \
        --train_batch_size "$train_batch_size" \
        --source_length "$source_length" \
        --save_total_limit "$save_total_limit" \
        --logging_steps "$logging_steps" \
        --bf16

    echo "Finished fine-tuning for model: $model_name"
done


# 记录loss, 单卡多进程，可以调整进程数

# llama-7b
python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20  

# llama-13b
python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-13b-chat-hf/" \
    --base_path "meta-llama/Llama-2-13b-chat-hf" \
    --output_file "result/llama_13b_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20 

# llama-70b
python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-70b-chat-hf/" \
    --base_path "meta-llama/Llama-2-70b-chat-hf" \
    --output_file "result/llama_70b_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20 



accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-7b-chat-hf \
    --base_path meta-llama/Llama-2-7b-chat-hf \
    --output_file result/rllama_7b_loss.json \
    --batch_size 256 \
    --num_processes 10

accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-13b-chat-hf \
    --base_path meta-llama/Llama-2-13b-chat-hf \
    --output_file result/rllama_13b_loss.json \
    --batch_size 256 \
    --num_processes 10

accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-70b-chat-hf \
    --base_path meta-llama/Llama-2-70b-chat-hf \
    --output_file result/rllama_70b_loss.json \
    --batch_size 256 \
    --num_processes 10

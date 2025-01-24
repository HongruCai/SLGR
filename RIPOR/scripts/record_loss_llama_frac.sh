

accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-7b-chat-hf_frac0.2 \
    --base_path meta-llama/Llama-2-7b-chat-hf \
    --output_file result/rllama_7b02_loss.json \
    --batch_size 256 \
    --num_processes 20


accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-7b-chat-hf_frac0.4 \
    --base_path meta-llama/Llama-2-7b-chat-hf \
    --output_file result/rllama_7b04_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-7b-chat-hf_frac0.6 \
    --base_path meta-llama/Llama-2-7b-chat-hf \
    --output_file result/rllama_7b06_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_llama_mp.py \
    --model_path ./output/Llama/Llama-2-7b-chat-hf_frac0.8 \
    --base_path meta-llama/Llama-2-7b-chat-hf \
    --output_file result/rllama_7b08_loss.json \
    --batch_size 256 \
    --num_processes 20
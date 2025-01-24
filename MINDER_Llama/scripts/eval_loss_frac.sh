

python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf_frac0.2/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b02_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20  


python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf_frac0.4/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b04_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20  

python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf_frac0.6/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b06_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20  

python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf_frac0.8/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b08_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20  
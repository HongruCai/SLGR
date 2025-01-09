

# # 记录loss, 单卡多进程，可以调整进程数num_processes

# llama-7b
python scripts/record_loss_llama_mp.py \
    --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
    --csv_file "../data/NQ/nq-test.csv" \
    --model_path "./output/Llama-2-7b-chat-hf/" \
    --base_path "meta-llama/Llama-2-7b-chat-hf" \
    --output_file "result/llama_7b_loss.json" \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 10  

# # llama-13b
# python scripts/record_loss_llama_mp.py \
#     --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
#     --csv_file "../data/NQ/nq-test.csv" \
#     --model_path "./output/Llama-2-13b-chat-hf/" \
#     --base_path "meta-llama/Llama-2-13b-chat-hf" \
#     --output_file "result/llama_13b_loss.json" \
#     --total_docs 21015324 \
#     --num_neg_samples 256 \
#     --num_processes 10 

# # llama-70b
# python scripts/record_loss_llama_mp.py \
#     --index "../data/FMIndex/Llama/llama_psgs_w100.fm_index" \
#     --csv_file "../data/NQ/nq-test.csv" \
#     --model_path "./output/Llama-2-70b-chat-hf/" \
#     --base_path "meta-llama/Llama-2-70b-chat-hf" \
#     --output_file "result/llama_70b_loss.json" \
#     --total_docs 21015324 \
#     --num_neg_samples 256 \
#     --num_processes 10 

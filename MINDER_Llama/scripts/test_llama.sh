CUDA_VISIBLE_DEVICES=2 \
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../zz/data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b_debug.json \
--checkpoint output/Llama-2-7b-chat-hf \
--jobs 5 \
--progress \
--device cuda:0 \
--batch_size 10 \
--beam 15 \
--decode_query False \
--fm_index  data/FMIndex/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores  \
# --debug
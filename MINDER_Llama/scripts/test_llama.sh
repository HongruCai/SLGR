
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b_res.json \
--checkpoint output/Llama-2-7b-chat-hf \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 1 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \


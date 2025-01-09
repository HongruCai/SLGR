CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b_norescoreep1.json \
--checkpoint output/Llama-2-7b-chat-hf \
--jobs 5 \
--progress \
--device cuda:0 \
--batch_size 2 \
--beam 15 \
--decode_query False \
--fm_index  ../data/FMIndex/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \
--debug \
--dont_use_markers


python ./seal/evaluate_output.py --file result/llama_7b_norescoreep1.json
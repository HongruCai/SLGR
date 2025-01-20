
# llama_7b
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b_res.json \
--checkpoint output/Llama-2-7b-chat-hf \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \

#llama_13b
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_13b_res.json \
--checkpoint output/Llama-2-13b-chat-hf \
--backbone meta-llama/Llama-2-13b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \

# llama_70b
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_70b_res.json \
--checkpoint output/Llama-2-70b-chat-hf \
--backbone meta-llama/Llama-2-70b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \

# llama_7b02
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b02_res.json \
--checkpoint output/Llama-2-7b-chat-hf_frac0.2 \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \

# llama_7b04
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b04_res.json \
--checkpoint output/Llama-2-7b-chat-hf_frac0.4 \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \


# llama_7b06
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b06_res.json \
--checkpoint output/Llama-2-7b-chat-hf_frac0.6 \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \


# llama_7b08
TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/llama_7b08_res.json \
--checkpoint output/Llama-2-7b-chat-hf_frac0.8 \
--backbone meta-llama/Llama-2-7b-chat-hf \
--jobs 20 \
--progress \
--device cuda:0 \
--batch_size 1 \
--beam 100 \
--decode_query False \
--fm_index  ../data/FMIndex/Llama/llama_psgs_w100.fm_index \
--dont_decode_title \
--dont_unigram_scores \


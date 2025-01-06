

python scripts/build_fm_index.py \
   ../data/NQ/psgs_w100.tsv  ../data/FMIndex/llama_psgs_w100.fm_index  \
    --hf_model meta-llama/Llama-2-7b-chat-hf \
    --jobs 40 \
    --include_title \
    --pid2query data/NQ/pid2query.pkl \
    --format dpr \
    --include_query \
    --query_format stable \

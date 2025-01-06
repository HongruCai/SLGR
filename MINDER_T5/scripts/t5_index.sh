python scripts/build_fm_index_t5.py \
   data/NQ/psgs_w100.tsv  \
   data/FMIndex/t5/t5_psgs_w100.fm_index  \
    --hf_model 'T5' \
    --jobs 40 \
    --include_title \
    --pid2query data/NQ/pid2query.pkl \
    --format dpr \
    --include_query \
    --query_format stable \
    --delim '<extra_id_99>'
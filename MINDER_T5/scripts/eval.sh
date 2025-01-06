# 推理，记录Recall，可以调整jobs, batch_size

TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/t5_small_res.json \
--checkpoint ./output/t5-small  \
--backbone t5-small \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 32 \
--beam 15 \
--decode_query stable \
--fm_index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
--dont_fairseq_checkpoint \
--max_passage_hits 100 \
--hits 100 \

TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/t5_base_res.json \
--checkpoint ./output/t5-base  \
--backbone t5-base \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 32 \
--beam 15 \
--decode_query stable \
--fm_index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
--dont_fairseq_checkpoint \
--max_passage_hits 100 \
--hits 100 \

TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/t5_large_res.json \
--checkpoint ./output/t5-large  \
--backbone t5-large \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 32 \
--beam 15 \
--decode_query stable \
--fm_index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
--dont_fairseq_checkpoint \
--max_passage_hits 100 \
--hits 100 \

TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/t5_3b_res.json \
--checkpoint ./output/t5-3b  \
--backbone t5-3b \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 32 \
--beam 15 \
--decode_query stable \
--fm_index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
--dont_fairseq_checkpoint \
--max_passage_hits 100 \
--hits 100 \

TOKENIZERS_PARALLELISM=false \
python ./seal/search.py \
--topics_format dpr_qas \
--topics ../data/NQ/nq-test.csv \
--output_format dpr \
--output result/t5_11b_res.json \
--checkpoint ./output/t5-11b  \
--backbone t5-11b \
--jobs 10 \
--progress \
--device cuda:0 \
--batch_size 32 \
--beam 15 \
--decode_query stable \
--fm_index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
--dont_fairseq_checkpoint \
--max_passage_hits 100 \
--hits 100 \
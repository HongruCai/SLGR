

python scripts/record_loss.py \
    --index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
    --csv_file ../data/NQ/nq-test.csv \
    --model_path output/t5-small \
    --output_file result/t5_small_loss.json \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20

python scripts/record_loss.py \
    --index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
    --csv_file ../data/NQ/nq-test.csv \
    --model_path output/t5-base \
    --output_file result/t5_base_loss.json \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20

python scripts/record_loss.py \
    --index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
    --csv_file ../data/NQ/nq-test.csv \
    --model_path output/t5-large \
    --output_file result/t5_large_loss.json \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20

python scripts/record_loss.py \
    --index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
    --csv_file ../data/NQ/nq-test.csv \
    --model_path output/t5-3b \
    --output_file result/t5_3b_loss.json \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20

python scripts/record_loss.py \
    --index ../data/FMIndex/T5/t5_psgs_w100.fm_index \
    --csv_file ../data/NQ/nq-test.csv \
    --model_path output/t5-11b \
    --output_file result/t5_11b_loss.json \
    --total_docs 21015324 \
    --num_neg_samples 256 \
    --num_processes 20


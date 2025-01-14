# batch size最大为256

accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5_/t5-small \
    --base_model_path t5-small \
    --output_path result/rt5_small_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5/t5-base \
    --base_model_path t5-base \
    --output_path result/rt5_base_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5/t5-large \
    --base_model_path t5-large \
    --output_path result/rt5_large_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5/t5-3b \
    --base_model_path t5-3b \
    --output_path result/rt5_3b_loss.json \
    --batch_size 256 \
    --num_processes 20

accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5/t5-11b \
    --base_model_path t5-11b \
    --output_path result/rt5_11b_loss.json \
    --batch_size 128 \
    --num_processes 20

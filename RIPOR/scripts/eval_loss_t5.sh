
accelerate launch --num_processes 8 record_loss_t5_mp.py \
    --model_path ./output/T5_/t5-small \
    --base_model_path t5-small \
    --output_path result/rt5_small_loss.json \
    --batch_size 256 \
    --num_processes 20


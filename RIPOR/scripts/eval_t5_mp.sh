accelerate launch --num_processes 3 test_t5_mp.py \
    --model_path ./output/T5/t5-small \
    --base_model_path t5-small \
    --output_dir result/t5_small_res.json
export HF_DATASETS_OFFLINE=1
export ckpt="/PATH/TO/checkpoints/STAGE3_RUN_NAME"
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port=12341 \
    -m lmms_eval \
    --model llava_vid \
    --tasks videomme,mlvd \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_next \
    --output_path ./logs/ \
    --model_args pretrained=$ckpt,video_decode_backend=decord,max_frames_num=16,mm_spatial_pool_mode=average,mm_newline_position=no_token,mm_resampler_location=after \
    
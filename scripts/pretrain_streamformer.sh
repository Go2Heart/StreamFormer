export CUDA_VISIBLE_DEVICES=1
task_config=all
frame_num=16

mkdir -p logs/frame$frame_num/siglip_multi_task_grad_accu_balance_$task_config

torchrun --master_port 52544 --nnodes 1 --nproc_per_node 1 --node_rank 0  run_finetuning_multi_task.py \
    --multi_tasks_metadata scripts/dataset_metadata/$task_config.yaml \
    --init_vit siglip \
    --use_decord \
    --batch_size 16 \
    --num_workers 8 \
    --pin_mem \
    --pretrained_model /mnt/vision_user/yibinyan/StreamFormer-3D/checkpoints/timesformer-siglip-16 \
    --world_size 1 \
    --local_rank 0 \
    --num_sample 1 \
    --aa rand-m7-n4-mstd0.5-inc1 \
    --train_interpolation bicubic \
    --sampler_type balanced \
    --update_freq 3 \
    --epochs 20 \
    --warmup_lr 0 \
    --lr 2e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 0 \
    --weight_decay 0.05 \
    --output_dir logs/frame$frame_num/siglip_multi_task_grad_accu_balance_$task_config/output \
    --save_ckpt \
    --enable_multitask_collate \
    --enable_lora_spatial \
    --enable_causal_temporal \
    --freeze_text_encoder \
    --log_dir logs/frame$frame_num/siglip_multi_task_grad_accu_balance_$task_config \
    2>&1 | tee logs/frame$frame_num/siglip_multi_task_grad_accu_balance_$task_config/test.log
    # --do_eval \
    # --ckpt_path /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/yanyibin-240108100044/outputs/siglip_multi_task_frame8_youtube_vis_LVVIS/checkpoint-epoch_2.pth \
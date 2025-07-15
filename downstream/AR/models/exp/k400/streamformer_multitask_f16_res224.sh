export OMP_NUM_THREADS=1

JOB_NAME='streamformer_multitask_f16_res224_single_gpu'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/PATH/TO/DATA/Kinetics400'
DATA_PATH='data_list/k400'
MODEL_PATH='/PATH/TO/PRETRAINED/timesformer-siglip-16'
PRETRAINED_CKPT='/PATH/TO/checkpoint-epoch.pth'

CUDA_VISIBLE_DEVICES=0 python main_finetuning.py \
        --model ${MODEL_PATH} \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_SIGLIP' \
        --nb_classes 400 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 16 \
        --num_sample 2 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --num_workers 12 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 30 \
        --lr 2e-4 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --enable_deepspeed \
        --test_best \
        --split ',' \
        --enable_lora_spatial \
        --ckpt_path ${PRETRAINED_CKPT}

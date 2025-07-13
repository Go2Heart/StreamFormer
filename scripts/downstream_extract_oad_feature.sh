#!/bin/bash

# DATASET='THUMOS14'
DATASET='TVSeries' 

MODEL_PATH='/PATH/TO/PRETRAINED/timesformer-siglip-16'
PRETRAINED_CKPT='/PATH/TO/CHECKPOINT/checkpoint-epoch.pth'

# Dataset paths
THUMOS_DATA_PATH='/PATH/TO/thumos-video'
THUMOS_ANNO_PATH='/PATH/TO/metadata/thumos'
THUMOS_SAVE_PATH='/PATH/TO/THUMOS14/streamformer_multitask_feature'

TVSERIES_DATA_PATH='/PATH/TO/tv_series/mkv_videos'
TVSERIES_ANNO_PATH='/PATH/TO/metadata/tvseries' 
TVSERIES_SAVE_PATH='/PATH/TO/TVSeries/streamformer_multitask_feature'

# Set data paths based on dataset
if [ "$DATASET" = "THUMOS14" ]; then
    DATA_PATH=$THUMOS_DATA_PATH
    ANNO_PATH=$THUMOS_ANNO_PATH
    SAVE_PATH=$THUMOS_SAVE_PATH
else
    DATA_PATH=$TVSERIES_DATA_PATH
    ANNO_PATH=$TVSERIES_ANNO_PATH
    SAVE_PATH=$TVSERIES_SAVE_PATH
fi

# Split work into 8 GPUs
interval=0.125
start_points=(0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875)

for i in {0..7}; do
    st=${start_points[$i]}
    # Use awk instead of bc for floating point arithmetic
    ed=$(awk "BEGIN {print $st + $interval}")
    st=$(printf "%.2f" $st)
    ed=$(printf "%.2f" $ed)
    
    CUDA_VISIBLE_DEVICES=$i python extract_oad_feature.py \
        --data_set $DATASET \
        --data_path $DATA_PATH \
        --dataset_annotation_folder $ANNO_PATH \
        --ckpt_path $PRETRAINED_CKPT \
        --save_path $SAVE_PATH \
        --pretrained_model $MODEL_PATH \
        --enable_lora_spatial \
        --start_idx $st \
        --end_idx $ed &
done

wait
"""Extract features for temporal action detection datasets"""
import argparse
import os
import random
import io
import time

import numpy as np
import torch
import json, yaml
import tqdm

from decord import VideoReader, cpu
from datasets.video_transforms import Compose, Resize, CenterCrop, Normalize
from datasets.volume_transforms import ClipToTensor
from models import StreamformerForMultiTaskingSigLIP

def get_args():
    parser = argparse.ArgumentParser(
        'Extract TAD features using the videomae model', add_help=False)

    parser.add_argument('--enable_lora_spatial', action='store_true', help='enable lora spatial')
    parser.add_argument('--data_set', default='THUMOS14', choices=['THUMOS14', 'TVSeries'], type=str, help='dataset')
    parser.add_argument('--data_path', default='YOUR_PATH/thumos14_video', type=str, help='dataset path')
    parser.add_argument('--save_path', default='actionformer_release/data/thumos/test_2', type=str, help='path for saving features')
    parser.add_argument('--pretrained_model', default='timesformer-siglip', type=str, help='pretrained model')
    parser.add_argument('--ckpt_path', default=None, help='load from checkpoint')
    parser.add_argument('--dataset_annotation_folder', default='data/THUMOS14', help='dataset annotation folder')
    parser.add_argument('--start_idx', default=0, type=float, help='start index')
    parser.add_argument('--end_idx', default=1, type=float, help='end index')
    return parser.parse_args()

######### version 2: given 24 fps video, we only forward 6 frames at a time, and thus the frame stride = 24/6 = 4 ###########
def get_start_idx_range(num_frames):
    return np.linspace(0, num_frames, num_frames // 6).astype(int)

def extract_feature(args):
    # preparation
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    data_transform = Compose([
        Resize(224, interpolation='bilinear'),
        CenterCrop(size=(224, 224)),
        ClipToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5])
    ])

    # get video path
    train_file = os.path.join(args.dataset_annotation_folder, "train.json")
    with open(train_file, 'r') as f:
        vid_list = list(json.load(f).keys())
    val_file = os.path.join(args.dataset_annotation_folder, "val.json")
    with open(val_file, 'r') as f:
        vid_list.extend(json.load(f).keys())
        if args.data_set == 'THUMOS14':
            vid_list.extend(['test/video_test_0001292.mp4']) ### we miss this in the val set for OAD, mannually add it 

    test_file = os.path.join(args.dataset_annotation_folder, "test.json")
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            vid_list.extend(json.load(f).keys())
    
    st_idx = int(len(vid_list) * args.start_idx)
    ed_idx = int(len(vid_list) * args.end_idx)
    vid_list = vid_list[st_idx:ed_idx]
    print(f'[{args.start_idx} / {args.end_idx}]: {len(vid_list)} videos to extract')
    
    # get model & load ckpt
    multi_task_config = {}
    model = StreamformerForMultiTaskingSigLIP.from_pretrained(
        args.pretrained_model, multi_task_config, ignore_mismatched_sizes=True)
    
    if args.enable_lora_spatial:
        model.add_lora_spatial()

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)['model']    
        ckpt = {k: v for k, v in ckpt.items() if 'task_heads' not in k}
        res = model.load_state_dict(ckpt, strict=False)
        print('Loading checkpoint:', res)
    
    model.eval()
    model.cuda()

    # extract feature
    num_videos = len(vid_list)
    batch_size = 16

    for idx, vid_name in tqdm.tqdm(enumerate(vid_list)):
        # url = os.path.join(args.save_path, vid_name.split('.')[0] + '.npy')
        url = os.path.join(args.save_path, vid_name.split('/')[-1].split('.')[0] + '.npy')
        
        dirname = os.path.dirname(url)
        if os.path.exists(url):
            continue
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        video_path = os.path.join(args.data_path, vid_name)
        vr = VideoReader(video_path, ctx=cpu(0))

        feature_list = []
        batched_input = []
    
        all_data = vr.get_batch(range(0, len(vr))).asnumpy()        
        transformed_data = data_transform(all_data).permute(1, 0, 2, 3)
        print(f'Begin processing {transformed_data.shape}')

        # get original fps
        original_fps = vr.get_avg_fps()
        target_fps = 24
        new_indices = np.linspace(0, len(transformed_data) - 1, int(len(transformed_data) / original_fps * target_fps)).astype(int)
        
        # resample to 24 fps
        transformed_data = transformed_data[new_indices]
        print(f'After fps conversion: {transformed_data.shape}, from original fps: {original_fps} to target fps: {target_fps}')

        st_time = time.time()

        total_range = get_start_idx_range(len(transformed_data))
        
        for start_idx in total_range:
            if start_idx + 6 > len(transformed_data):
                frame_q = transformed_data[len(transformed_data) - 6:]
            else:
                frame_q = transformed_data[start_idx: start_idx + 6]
                
            input_data = frame_q.unsqueeze(0).cuda()
            with torch.no_grad():
                feature = model.forward_features(input_data, pooling_method="last")
                feature_list.append(feature.cpu().numpy())
        
        feature_vector = np.vstack(feature_list)
        np.save(url, feature_vector)
        ed_time = time.time()
        print(f'[{idx} / {num_videos}]: save feature on {url} with shape:{feature_vector.shape}, used time {ed_time - st_time}')

if __name__ == '__main__':
    args = get_args()
    extract_feature(args)
    
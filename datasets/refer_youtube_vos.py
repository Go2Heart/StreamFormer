import json
import os
import io
import cv2
import numpy as np
import torch
import pandas as pd
import random
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter, 
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop,
    PairCompose, PairResize, PairRandomCrop, PairRandomHorizontalFlip
)
from PIL import Image
from .volume_transforms import ClipToTensor

import pycocotools
from pycocotools import mask as mask_utils
import ast

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False


class ReferYoutubeVOSDataset(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train',
                 crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
                 test_num_crop=3, filename_tmpl='img_{:05}.jpg', args=None):
        self.anno_path = anno_path
        self.prefix = prefix
        self.split = split
        self.mode = mode
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.filename_tmpl = filename_tmpl
        self.args = args
        self.aug = False
        self.rand_erase = False

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

        if self.mode in ['train']:
            self.aug = True
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos.")
        
        self.videos = pd.read_csv(anno_path)
        self.label2id = {} 

        print(f"There are {len(self.videos)} videos")

        if (mode == 'train'):
            self.paired_transform = PairCompose([
                PairResize(self.short_side_size), 
                PairRandomCrop(self.crop_size),
                # PairRandomHorizontalFlip(p=0.5), ### we have left/right descriptions
            ])
            self.frame_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        elif (mode == 'validation'):
            if args.init_vit == 'siglip':
                self.data_transform = Compose([
                    Resize(self.short_side_size, interpolation='bilinear'),
                    CenterCrop(size=(self.crop_size, self.crop_size)),
                    ClipToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
                ])
            else: 
                self.data_transform = Compose([
                    Resize(self.short_side_size,
                                            interpolation='bilinear'),
                    CenterCrop(size=(self.crop_size,
                                                    self.crop_size)),
                    ClipToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
        elif mode == 'test':
            self.data_resize = Compose([
                Resize(size=(short_side_size),
                                        interpolation='bilinear')
            ])
            if args.init_vit == 'siglip':
                self.data_transform = Compose([
                    ClipToTensor(),
                    Normalize(mean=[0.5, 0.5, 0.5],
                                               std=[0.5, 0.5, 0.5])
                ])
            else:
                self.data_transform = Compose([
                    ClipToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                ])
            
    def copy_dataset(self, num_copies):
        if isinstance(self.data_list, pd.DataFrame):
            self.data_list = pd.concat([self.data_list] * num_copies, ignore_index=True)
        else:
            self.data_list = self.data_list * num_copies

    def visualize(self, index, video, mask):
        # video: [T, 3, h, w], [-1~1]
        # mask: [T, h, w]: {0, 1}
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        video[:, 0] = video[:, 0] * std[0] + mean[0]
        video[:, 1] = video[:, 1] * std[1] + mean[1]
        video[:, 2] = video[:, 2] * std[2] + mean[2]
        # video = (video.numpy() * 255).astype(np.uint8)
        for i in range(len(video)):
            image = (video[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image_mask = (mask[i] * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image_mask = Image.fromarray(image_mask)
            image.save(f'visualisation/{index}_{i}_raw.png')
            image_mask.save(f'visualisation/{index}_{i}_mask.png')


    def __getitem__(self, index):
        try:

            video = self.videos.iloc[index]
            video_id = video['video_id']
            text = video['text']
            frame_names = ast.literal_eval(video['frames'])
            total_frame = video['length']
            obj_ids = ast.literal_eval(video['obj_id'])
            anno_ids = ast.literal_eval(video['anno_id'])

            # annotations = self.annotations[video_id]

            # print('video_id', video_id, 'total_frame', total_frame, 'frame_names', frame_names, 'obj_id', obj_ids, 'anno_id', anno_ids)
            # print('text', text)
            # breakpoint()

            buffer, mask_target = self.load_video_and_mask(video_id, frame_names, obj_ids[0], total_frame)  # T H W C

            # print('before transform:')
            # print(buffer.shape, buffer.min(), buffer.max(), buffer.mean())
            # print(mask_target.shape, mask_target.min(), mask_target.max(), mask_target.mean())
                   
            assert mask_target.sum() > 0, 'mask_target does not exist for index: {}, video_id: {}, text: {}'.format(index, video_id, text)

            buffer, mask_target = self.paired_transform(buffer, mask_target)
            buffer = self.frame_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3) # T C H W
            mask_target = torch.from_numpy(np.stack(mask_target, 0)).long()  # [T,H,W]

            # print('after transform:')
            # print(buffer.shape, buffer.min(), buffer.max(), buffer.mean())
            # print(mask_target.shape, torch.unique(mask_target))
            # breakpoint()
                 
            # self.visualize(index, buffer, mask_target)

            return {
                "task_name": "ReferYoutubeVOS",
                "task_input": {
                    "video": buffer,
                    "video_id": video_id,
                    "caption": text,
                    "mask_target": mask_target, # for cross entropy loss
                    "mask_size": (mask_target.shape[1], mask_target.shape[2]),
                }
            }

        except:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)    

    def load_video_and_mask(self, sample, frame_names, obj_id, total_frame):
        """Load video content using Decord"""

        average_duration = total_frame // self.num_segment
        all_index = []
        ### TODO: try a continous sampling strategy, combined with global sampling ###
        ### https://github.com/wjn922/ReferFormer/blob/main/datasets/ytvos.py#L104 ###
        if average_duration > 0:
            all_index = list(
                np.multiply(list(range(self.num_segment)),
                            average_duration) +
                np.random.randint(average_duration, size=self.num_segment)) # TODO: check this, training with random sampling, try uniform sampling later

        else:
            all_index = [0] * (self.num_segment - total_frame) + list(
                range(total_frame))

        all_index = list(np.array(all_index))
        # print('all index:', all_index)

        all_images, all_masks = [], []
        for j in range(self.num_segment):
            frame_indx = all_index[j]
            frame_name = frame_names[j]
            img_path = os.path.join(str(self.prefix), 'JPEGImages', sample, frame_name + '.jpg')
            mask_path = os.path.join(str(self.prefix), 'Annotations', sample, frame_name + '.png')
            # print(img_path, mask_path)
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = Image.open(mask_path).convert('P')
            mask = np.array(mask)
            mask = (mask==obj_id).astype(np.float32)
            # mask = torch.from_numpy(mask)

            # print(img.shape, img.min(), img.max(), img.mean())
            # print(mask.shape, np.unique(mask))
            # breakpoint()

            all_images.append(img)
            all_masks.append(mask)
        
        buffer = np.array(all_images)
        mask_target = np.array(all_masks)

        return buffer, mask_target

    def __len__(self):
        return len(self.videos)
        
        
        

import json
import os
import io
import cv2
import numpy as np
import torch
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
    PairCompose, PairResize, PairRandomCrop, PairRandomHorizontalFlip,
    PairMaskCrop,
)
from PIL import Image
from .volume_transforms import ClipToTensor

import pycocotools
from pycocotools import mask as mask_utils

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

class TaskVISDataset(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(self, anno_path, prefix='', split=' ', mode='train',
                 crop_size=224, short_side_size=256, new_height=256, new_width=340,
                 keep_aspect_ratio=True, num_segment=1, num_crop=1, test_num_segment=10,
                 test_num_crop=3, filename_tmpl='img_{:05}.jpg', 
                 label2id_path=None, data_dict=None,
                 args=None):
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
        if data_dict is not None:
            self.prefix_dict = data_dict['root_dir']
        else:
            self.prefix_dict = {
                "YoutubeVIS": "data/youtube_vis_2019/train/JPEGImages/",
                "LVVIS": "data/lvvis/train/JPEGImages/",
                "COCOPseudoVIS": "data/coco/images/train2017/",
            }
        
        self.label2id = json.load(open(label2id_path, 'r'))
        self.total_videos = {}
        self.total_annotations = {}
        self.accumulated_video_len = []
        
        for per_data_anno in os.listdir(self.anno_path):
            if not (per_data_anno.startswith('train_') and per_data_anno.endswith('.json')):
                continue

            cur_dataset = per_data_anno.split('_')[1].split('.')[0]
            cur_anno_path = os.path.join(self.anno_path, per_data_anno)
            with open(cur_anno_path, 'r') as f:
                data = json.load(f)

            if cur_dataset == 'COCOPseudoVIS':
                videos = data['images']
                categories = data['categories']
                len_coco_videos = len(videos)
                self.id2sequential_id = {cat['id']: idx for idx, cat in enumerate(categories)}
            else:
                videos = data['videos'] 

            annotations = {}
            video_ids = [video['id'] for video in videos]
            video_ids.reverse()
            for anno in data['annotations']:
                if cur_dataset == 'COCOPseudoVIS':
                    video_id = anno['image_id']
                else:
                    video_id = anno['video_id']

                if video_id not in annotations:
                    annotations[video_id] = []
                annotations[video_id].append(anno)
            
            for idx, video_id in enumerate(video_ids):
                if video_id not in annotations:
                    if cur_dataset == 'COCOPseudoVIS':
                        reversed_idx = len_coco_videos - idx - 1
                        videos.pop(reversed_idx)
                    else:
                        videos.pop(video_id)
                        # if video_id in videos:
                        #     videos.pop(video_id)

            self.total_videos[cur_dataset] = videos
            self.total_annotations[cur_dataset] = annotations
            self.accumulated_video_len.append(len(videos))

            print(f'After processing dataset: {cur_dataset}, there are {len(videos)} videos and {len(annotations)} annotations')

        self.dataset_names = list(self.total_videos.keys())
        self.cumulative_lengths = np.cumsum(self.accumulated_video_len)

        print(f"Total videos: {self.cumulative_lengths[-1]}")
        if (mode == 'train'):
            self.paired_transform = PairCompose([
                PairResize(self.short_side_size),
                PairMaskCrop(self.crop_size), 
                # PairRandomCrop(self.crop_size),
                PairRandomHorizontalFlip(p=0.5),
            ])
            self.frame_transform = Compose([
                ClipToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
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
        for dataset_name in self.total_videos:
            self.total_videos[dataset_name] = self.total_videos[dataset_name] * num_copies
            
        for dataset_name in self.total_annotations:
            self.total_annotations[dataset_name] = {
                k: v for _ in range(num_copies) 
                for k, v in self.total_annotations[dataset_name].items()
            }
        
        self.dataset_names = list(self.total_videos.keys())
        self.accumulated_video_len = [length * num_copies for length in self.accumulated_video_len]
        
        self.cumulative_lengths = np.cumsum(self.accumulated_video_len)
        
        print(f"After copying {num_copies} times, total videos: {self.cumulative_lengths[-1]}")


    def process_coco_pseudo_vis(self, video, annotations, dataset_name):
        fname = video['file_name']
        pseudo_frames, image_shape = self.load_pseudo_frames(fname, dataset_name)
        image_shape = image_shape[:2]
        
        
        # load anno mask
        anno_masks = {}
        for anno in annotations:
            category_id = self.id2sequential_id[anno['category_id']]
            masks = []
            if anno['iscrowd'] and anno['segmentation']is not None:
                # read in RLE objects
                # import ipdb; ipdb.set_trace()
                mask = pycocotools.mask.frPyObjects(anno['segmentation'], video['height'], video['width'])
                # mask = anno['segmentations'][idx]
                mask = pycocotools.mask.decode(mask)
                mask = np.array(mask, dtype=np.uint8)
                # mask = torch.from_numpy(mask)
                masks.append(mask)
            else:
                # read in polygon objects
                if anno['segmentation'] is not None:
                    masks.append(self.polygons_to_mask(image_shape, anno['segmentation']))
                else:
                    masks.append(np.zeros(image_shape))
                # masks.append(torch.zeros(anno['height'], anno['width']))
            masks = np.stack(masks, 0) # [1xHxW]
            if category_id not in anno_masks:
                anno_masks[category_id] = masks
            else:
                anno_masks[category_id] = anno_masks[category_id] | masks
        mask_target = np.zeros_like(masks, dtype=np.int32) # [1xHxW]
        for category_id, mask in anno_masks.items():
            # where [t,x,y] in mask is non-zero, set mask_target to category_id
            # assert mask_target[t,x,y] == 0
            # assert (mask_target[masks != 0] == 0).all() # assert each pixel is valid for one and only one label
            mask_target[mask != 0] = category_id
        # copy mask_target to all frames
        mask_target = np.repeat(mask_target, self.num_segment, axis=0) # [T,H,W]
        pseudo_frames, mask_target = self._random_rotation(pseudo_frames, mask_target)
        
        pseudo_frames, mask_target = self.paired_transform(pseudo_frames, mask_target)
        buffer = self.frame_transform(pseudo_frames)
        mask_target = torch.from_numpy(np.stack(mask_target))
        # buffer = self.data_transform(pseudo_frames)
        # copy frame times
        buffer = buffer.permute(1, 0, 2, 3) # T C H W
        return buffer, mask_target, masks

    def process_youtube_vis(self, video, annotations, dataset_name):
        frame_names = video['file_names']
        total_frame = video['length']
        
        buffer, all_index = self.load_frame(frame_names, dataset_name, total_frame)  # T H W C
        
        # load anno mask
        anno_masks = {}
        for anno in annotations:
            category_id = anno['category_id']
            masks = []
            for idx in all_index:
                if anno['segmentations'][idx] is not None:
                    mask = pycocotools.mask.frPyObjects(anno['segmentations'][idx], anno['height'], anno['width'])
                    mask = pycocotools.mask.decode(mask)
                    mask = np.array(mask, dtype=np.uint8)
                    masks.append(mask)
                else:
                    masks.append(np.zeros((anno['height'], anno['width']), dtype=np.uint8))
            masks = np.stack(masks, 0) # concat along the temporal dimension
            if category_id not in anno_masks:
                anno_masks[category_id] = masks
            else:
                anno_masks[category_id] = anno_masks[category_id] | masks
        mask_target = np.zeros_like(masks, dtype=np.int32) # [T,H,W]
        for category_id, masks in anno_masks.items():
            mask_target[masks != 0] = category_id # [T,H,W] with category_id indicating the label
        
        buffer, mask_target = self.paired_transform(buffer, mask_target)
        buffer = self.frame_transform(buffer)
        buffer = buffer.permute(1, 0, 2, 3) # T C H W
        mask_target = torch.from_numpy(np.stack(mask_target, 0)) # [T,H,W]

        return buffer, mask_target, masks

    def process_lvvis(self, video, annotations, dataset_name):
        frame_names = video['file_names']
        total_frame = video['length']
        
        buffer, all_index = self.load_frame(frame_names, dataset_name, total_frame)  # T H W C
        
        # load anno mask
        anno_masks = {}
        for anno in annotations:
            category_id = anno['category_id']
            masks = []
            for idx in all_index:
                if anno['segmentations'][idx] is not None:
                    mask = anno['segmentations'][idx]
                    mask = pycocotools.mask.decode(mask)
                    mask = np.array(mask, dtype=np.uint8)
                else:
                    mask = np.zeros((anno['height'], anno['width']), dtype=np.uint8)
                masks.append(mask)
            masks = np.stack(masks, 0)
            if category_id not in anno_masks:
                anno_masks[category_id] = masks
            else:
                anno_masks[category_id] = anno_masks[category_id] | masks
            
        mask_target = np.zeros_like(masks, dtype=np.int32)
        for category_id, masks in anno_masks.items():
            mask_target[masks != 0] = category_id
            
        buffer, mask_target = self.paired_transform(buffer, mask_target)
        buffer = self.frame_transform(buffer)
        buffer = buffer.permute(1, 0, 2, 3)  # T C H W
        mask_target = torch.from_numpy(np.stack(mask_target, 0))  # [T,H,W]

        return buffer, mask_target, masks

    def __getitem__(self, index):
        try:
            dataset_index = np.searchsorted(self.cumulative_lengths, index, side='right')
            sample_index = index - (self.cumulative_lengths[dataset_index - 1] if dataset_index > 0 else 0)
            
            dataset_name = self.dataset_names[dataset_index]
            video = self.total_videos[dataset_name][sample_index]
            video_id = video['id']
            annotations = self.total_annotations[dataset_name][video_id]

            if dataset_name == 'COCOPseudoVIS':
                buffer, mask_target, masks = self.process_coco_pseudo_vis(video, annotations, dataset_name)
            elif dataset_name == 'LVVIS':
                buffer, mask_target, masks = self.process_lvvis(video, annotations, dataset_name)
            elif dataset_name == 'YoutubeVIS':
                buffer, mask_target, masks = self.process_youtube_vis(video, annotations, dataset_name)
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")
            
            assert mask_target is not None and mask_target.sum() > 0, f'<task_{dataset_name}> mask_target is None or No foreground'
            # self.visualize(index, buffer, mask_target)

            return {
                "task_name": 'TaskVIS',
                "task_input": {
                    "video": buffer,
                    "video_id": video_id,
                    "mask_target": mask_target, # for cross entropy loss
                    "mask_size": (mask_target.shape[1], mask_target.shape[2]),
                    'dataset': dataset_name,
                }
            }
        except Exception as e:
            print(f'Error loading video {video_id} at index {index}: {e}')
            return self.__getitem__(random.randint(0, len(self) - 1))

    def load_frame(self, sample, dataset_name, num_frames, sample_rate_scale=1):
        """Load video content using Decord"""
        frame_names = [os.path.join(self.prefix_dict[dataset_name], fname) for fname in sample]
        
        assert len(frame_names) == num_frames
        if self.mode == 'test':
            raise NotImplementedError
            tick = num_frames / float(self.num_segment)
            all_index = []
            for t_seg in range(self.test_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_num_segment + tick * x)
                    for x in range(self.num_segment)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            imgs = []
            for idx in all_index:
                frame_fname = os.path.join(fname, self.filename_tmpl.format(idx + 1)) 
                if "s3://" in fname:
                    img_bytes = self.client.get(frame_fname)
                else:
                    with open(frame_fname, 'rb') as f:
                        img_bytes = f.read()
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
                imgs.append(img)
            buffer = np.array(imgs)
            return buffer

        # handle temporal segments
        average_duration = num_frames // self.num_segment
        all_index = []
        if average_duration > 0:
            if self.mode == 'validation':
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.ones(self.num_segment, dtype=int) *
                    (average_duration // 2))
            else:
                all_index = list(
                    np.multiply(list(range(self.num_segment)),
                                average_duration) +
                    np.random.randint(average_duration, size=self.num_segment))

        else:
            all_index = [0] * (self.num_segment - num_frames) + list(
                range(num_frames))
        all_index = list(np.array(all_index))
        imgs = []
        #print(frame_names)
        #breakpoint()
        for idx in all_index:
            frame_fname = frame_names[idx]
            if "s3://" in frame_fname:
                img_bytes = self.client.get(frame_fname)
            else:
                with open(frame_fname, 'rb') as f:
                    img_bytes = f.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            imgs.append(img)
        buffer = np.array(imgs)
        return buffer, all_index
    
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
            image.save(f'{index}_{i}_raw.png')
            image_mask.save(f'{index}_{i}_mask.png')



    def _random_rotation(self, video, mask):
        T, H, W, C = video.shape

        rotated_video = []
        rotated_mask = []
        
        for t in range(T):
            angle = random.uniform(-15, 15)
        
            # Center offset
            center_offset_x = random.uniform(-1, 1) * W // 10 + W//2
            center_offset_y = random.uniform(-1, 1) * H // 10 + H//2
            center = (center_offset_x, center_offset_y)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            frame = video[t]
            m = mask[t]
            
            # rotate image
            rotated_frame = cv2.warpAffine(
                frame, rotation_matrix, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # rotate mask
            rotated_m = cv2.warpAffine(
                m, rotation_matrix, (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            rotated_video.append(rotated_frame)
            rotated_mask.append(rotated_m)
        
        return np.stack(rotated_video), np.stack(rotated_mask)

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        
        if isinstance(polygons, list):
            for polygon in polygons:
                if len(polygon) >= 6:
                    pts = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
        else:
            shape = polygons.shape
            polygons = polygons.reshape(shape[0], -1, 2)
            cv2.fillPoly(mask, polygons, 1)
        # mask = torch.from_numpy(mask)
        return mask

    def load_pseudo_frames(self, fname, dataset_name):
        # copy the frame self.num_segment times to form a pseudo video, using same format as self.load_frame
        frame_fname = os.path.join(self.prefix_dict[dataset_name], fname)
        if 's3://' in frame_fname:
            img_bytes = self.client.get(frame_fname)
        else:
            with open(frame_fname, 'rb') as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        shape = img.shape
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        buffer = np.array([img] * self.num_segment)
        return buffer, shape

    def __len__(self):
        return sum(len(videos) for videos in self.total_videos.values())
        
        


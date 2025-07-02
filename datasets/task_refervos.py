import ast
import io
import json
import os
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import pycocotools
import torch
from decord import VideoReader, cpu
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision import transforms

from .random_erasing import RandomErasing
from .video_transforms import (
    CenterCrop,
    Compose,
    Normalize,
    PairCompose,
    PairMaskCrop,
    PairResize,
    Resize,
)
from .volume_transforms import ClipToTensor

try:
    from petrel_client.client import Client

    has_client = True
except ImportError:
    has_client = False


class TaskReferVOSDataset(Dataset):
    """Load your own raw frame classification dataset."""

    def __init__(
        self,
        anno_path,
        prefix="",
        split=" ",
        mode="train",
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        test_num_segment=10,
        test_num_crop=3,
        filename_tmpl="img_{:05}.jpg",
        data_dict=None,
        args=None,
    ):
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
            self.client = Client("~/petreloss.conf")

        if self.mode in ["train"]:
            self.aug = True
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos."
            )
        if data_dict is not None:
            self.prefix_dict = data_dict["prefix_dict"]
            self.mask_root_dict = data_dict["mask_root_dict"]
        else:
            self.prefix_dict = {
                "MEVIS": "data/mevis/videos",
                "ReferYoutubeVOS": "data/refer_youtube_vos/ICCV2023_VOS/train/JPEGImages",
                "ReferCOCOPseudo": "data/coco/images/train2017",  # TODO: add this
            }

            self.mask_root_dict = {
                # "MEVIS": os.path.join(os.path.dirname(self.prefix_dict["MEVIS"]), "masks"),
                "MEVIS": "data/mevis/masks",
                "ReferYoutubeVOS": os.path.join(
                    os.path.dirname(self.prefix_dict["ReferYoutubeVOS"]), "Annotations"
                ),
                "ReferCOCOPseudo": "",  # TODO: add this
            }

        self.label2id = {}
        self.total_videos = {}
        self.total_annotations = {}
        self.accumulated_video_len = []

        for per_data_anno in os.listdir(self.anno_path):
            if not per_data_anno.startswith("train_"):
                continue

            cur_dataset = per_data_anno.split("_")[1].split(".")[0]
            cur_anno_path = os.path.join(self.anno_path, per_data_anno)

            if per_data_anno.endswith(".csv"):
                data = pd.read_csv(cur_anno_path)
            elif per_data_anno.endswith(".json"):
                with open(cur_anno_path, "r") as f:
                    data_dict = json.load(f)
                self.total_annotations[cur_dataset] = data_dict["annotations"]
                data = data_dict["images"]
            else:
                raise ValueError(f"Unknown annotation file type: {per_data_anno}")

            self.total_videos[cur_dataset] = data
            self.accumulated_video_len.append(len(data))

            print(
                f"After processing dataset: {cur_dataset}, there are {len(data)} videos"
            )

        self.dataset_names = list(self.total_videos.keys())
        self.cumulative_lengths = np.cumsum(self.accumulated_video_len)
        print(f"Total videos: {self.cumulative_lengths[-1]}")

        if mode == "train":
            self.paired_transform = PairCompose(
                [
                    PairResize(self.short_side_size),
                    PairMaskCrop(self.crop_size),
                    # PairRandomCrop(self.crop_size),
                    # PairRandomHorizontalFlip(p=0.5), ### we have left/right descriptions
                ]
            )
            self.frame_transform = Compose(
                [ClipToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
            )

        elif mode == "validation":
            if args.init_vit == "siglip":
                self.data_transform = Compose(
                    [
                        Resize(self.short_side_size, interpolation="bilinear"),
                        CenterCrop(size=(self.crop_size, self.crop_size)),
                        ClipToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
            else:
                self.data_transform = Compose(
                    [
                        Resize(self.short_side_size, interpolation="bilinear"),
                        CenterCrop(size=(self.crop_size, self.crop_size)),
                        ClipToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        elif mode == "test":
            self.data_resize = Compose(
                [Resize(size=(short_side_size), interpolation="bilinear")]
            )
            if args.init_vit == "siglip":
                self.data_transform = Compose(
                    [
                        ClipToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
            else:
                self.data_transform = Compose(
                    [
                        ClipToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def copy_dataset(self, num_copies):
        # 复制total_videos中每个数据集的视频
        for dataset_name in self.total_videos:
            self.total_videos[dataset_name] = (
                self.total_videos[dataset_name] * num_copies
            )

        # 复制total_annotations中每个数据集的标注
        for dataset_name in self.total_annotations:
            self.total_annotations[dataset_name] = {
                k: v
                for _ in range(num_copies)
                for k, v in self.total_annotations[dataset_name].items()
            }

        # 更新dataset_names
        self.dataset_names = list(self.total_videos.keys())
        # 更新accumulated_video_len
        self.accumulated_video_len = [
            length * num_copies for length in self.accumulated_video_len
        ]

        # 更新cumulative_lengths
        self.cumulative_lengths = np.cumsum(self.accumulated_video_len)

        print(
            f"After copying {num_copies} times, total videos: {self.cumulative_lengths[-1]}"
        )

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
            image.save(f"visualisation/{index}_{i}_raw.png")
            image_mask.save(f"visualisation/{index}_{i}_mask.png")

    def process_mevis(self, video):
        video_id = video["video_id"]
        text = video["text"]
        frame_names = ast.literal_eval(video["frames"])
        total_frame = video["length"]
        obj_ids = ast.literal_eval(video["obj_id"])
        anno_ids = ast.literal_eval(video["anno_id"])

        buffer, all_index = self.loadvideo_decord(
            video_id + ".mp4", total_frame
        )  # T H W C

        mask_target = None
        for obj_id, anno_id in zip(obj_ids, anno_ids):
            # print('Loading mask for obj_id', obj_id, 'anno_id', anno_id)
            mask_path = os.path.join(self.mask_root_dict["MEVIS"], f"{anno_id}.json")
            if "s3://" in mask_path:
                f = io.BytesIO(self.client.get(mask_path))
                mask_data = json.load(f)
            else:
                with open(mask_path, "r") as f:
                    mask_data = json.load(f)

            mask_data_indexed = [mask_data[i] for i in all_index]
            # print('mask_data', len(mask_data), 'mask_data_indexed', len(mask_data_indexed))

            masks = []
            has_mask_index = -1
            no_mask_index = []
            for idx, per_frame_mask in enumerate(mask_data_indexed):
                if per_frame_mask is not None:
                    mask = mask_utils.decode(per_frame_mask)
                    mask = np.array(mask, dtype=np.uint8)
                    # mask = torch.from_numpy(mask)
                    masks.append(mask)
                    has_mask_index = idx
                else:
                    # masks.append(torch.zeros(self.crop_size, self.crop_size))
                    masks.append(np.zeros((self.crop_size, self.crop_size)))
                    no_mask_index.append(idx)

            for idx in no_mask_index:
                # masks[idx] = torch.zeros_like(masks[has_mask_index])
                masks[idx] = np.zeros_like(masks[has_mask_index])

            # masks = torch.stack(masks, 0).long() # concat along the temporal dimension
            masks = np.stack(masks, 0)

            if masks.sum() == 0:
                ### no masks here, skipping ###
                continue

            if mask_target is None:
                mask_target = masks
            else:
                mask_target = mask_target | masks  ### union of masks ###

        assert (
            mask_target is not None and mask_target.sum() > 0
        ), "<task mevis> mask_target is None or No foreground"

        buffer, mask_target = self.paired_transform(buffer, mask_target)
        buffer = self.frame_transform(buffer)
        buffer = buffer.permute(1, 0, 2, 3)  # T C H W
        mask_target = torch.from_numpy(np.stack(mask_target, 0)).long()  # [T,H,W]

        return buffer, mask_target, text

    def process_youtube_vos(self, video):
        video_id = video["video_id"]
        text = video["text"]
        frame_names = ast.literal_eval(video["frames"])
        total_frame = video["length"]
        obj_ids = ast.literal_eval(video["obj_id"])
        anno_ids = ast.literal_eval(video["anno_id"])

        buffer, mask_target = self.load_video_and_mask(
            video_id, frame_names, obj_ids[0], total_frame
        )  # T H W C

        assert mask_target.sum() > 0, "<task ytvos> mask_target has no foreground"

        buffer, mask_target = self.paired_transform(buffer, mask_target)
        buffer = self.frame_transform(buffer)
        buffer = buffer.permute(1, 0, 2, 3)  # T C H W
        mask_target = torch.from_numpy(np.stack(mask_target, 0)).long()  # [T,H,W]
        return buffer, mask_target, text

    def process_refcoco_pseudo(self, video, annotation, dataset_name):
        fname = video["file_name"].split("_")[-1]
        text = video["caption"]
        pseudo_frames, image_shape = self.load_pseudo_frames(fname, dataset_name)
        image_shape = image_shape[:2]

        ### assume we only have one segment here ###
        anno = annotation

        masks = []
        if anno["iscrowd"] and anno["segmentation"] is not None:
            # read in RLE objects
            # import ipdb; ipdb.set_trace()
            mask = pycocotools.mask.frPyObjects(
                anno["segmentation"], video["height"], video["width"]
            )
            # mask = anno['segmentations'][idx]
            mask = pycocotools.mask.decode(mask)
            mask = np.array(mask, dtype=np.uint8)
            # mask = torch.from_numpy(mask)
            masks.append(mask)
        else:
            # read in polygon objects
            if anno["segmentation"] is not None:
                masks.append(self.polygons_to_mask(image_shape, anno["segmentation"]))
            else:
                masks.append(np.zeros(image_shape))
            # masks.append(torch.zeros(anno['height'], anno['width']))
        mask_target = np.stack(masks, 0)  # [1xHxW]
        # copy mask_target to all frames
        mask_target = np.repeat(mask_target, self.num_segment, axis=0)  # [T,H,W]

        assert (
            mask_target.sum() > 0
        ), "<task refcoco pseudo> mask_target has no foreground"

        pseudo_frames, mask_target = self._random_rotation(pseudo_frames, mask_target)
        pseudo_frames, mask_target = self.paired_transform(pseudo_frames, mask_target)

        buffer = self.frame_transform(pseudo_frames)
        mask_target = torch.from_numpy(np.stack(mask_target))

        # copy frame times
        buffer = buffer.permute(1, 0, 2, 3)  # T C H W
        return buffer, mask_target, text

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

    def _random_rotation(self, video, mask):
        T, H, W, C = video.shape

        rotated_video = []
        rotated_mask = []

        for t in range(T):
            angle = random.uniform(-15, 15)

            # Center offset
            center_offset_x = random.uniform(-1, 1) * W // 10 + W // 2
            center_offset_y = random.uniform(-1, 1) * H // 10 + H // 2
            center = (center_offset_x, center_offset_y)

            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            frame = video[t]
            m = mask[t]

            # rotate image
            rotated_frame = cv2.warpAffine(
                frame,
                rotation_matrix,
                (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # rotate mask
            rotated_m = cv2.warpAffine(
                m,
                rotation_matrix,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            rotated_video.append(rotated_frame)
            rotated_mask.append(rotated_m)

        return np.stack(rotated_video), np.stack(rotated_mask)

    def load_pseudo_frames(self, fname, dataset_name):
        # copy the frame self.num_segment times to form a pseudo video, using same format as self.load_frame
        frame_fname = os.path.join(self.prefix_dict[dataset_name], fname)
        if "s3://" in frame_fname:
            img_bytes = self.client.get(frame_fname)
        else:
            with open(frame_fname, "rb") as f:
                img_bytes = f.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        shape = img.shape
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        buffer = np.array([img] * self.num_segment)
        return buffer, shape

    def __getitem__(self, index):
        try:

            dataset_index = np.searchsorted(
                self.cumulative_lengths, index, side="right"
            )
            sample_index = index - (
                self.cumulative_lengths[dataset_index - 1] if dataset_index > 0 else 0
            )
            dataset_name = self.dataset_names[dataset_index]

            if dataset_name == "MEVIS":
                video = self.total_videos[dataset_name].iloc[sample_index]
                video_id = video["video_id"]
                buffer, mask_target, text = self.process_mevis(video)
            elif dataset_name == "ReferYoutubeVOS":
                video = self.total_videos[dataset_name].iloc[sample_index]
                video_id = video["video_id"]
                buffer, mask_target, text = self.process_youtube_vos(video)
            elif dataset_name == "ReferCOCOPseudo":
                video = self.total_videos[dataset_name][sample_index]
                video_id = video["file_name"]
                annotation = self.total_annotations[dataset_name][sample_index]
                buffer, mask_target, text = self.process_refcoco_pseudo(
                    video, annotation, dataset_name
                )

            assert (
                mask_target is not None and mask_target.sum() > 0
            ), f"{dataset_name} mask_target has no foreground"
            # self.visualize(index, buffer, mask_target)

            return {
                "task_name": "TaskReferVOS",
                "task_input": {
                    "video": buffer,
                    "video_id": video_id,
                    "caption": text,
                    "mask_target": mask_target,  # for cross entropy loss
                    "mask_size": (mask_target.shape[1], mask_target.shape[2]),
                },
            }

        except:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

    def loadvideo_decord(self, sample, total_frame):
        """Load video content using Decord"""
        fname = sample
        fname = os.path.join(self.prefix_dict["MEVIS"], fname)

        # try:
        if self.keep_aspect_ratio:
            if "s3://" in fname:
                video_bytes = self.client.get(fname)
                vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        else:
            if "s3://" in fname:
                video_bytes = self.client.get(fname)
                vr = VideoReader(
                    io.BytesIO(video_bytes),
                    width=self.new_width,
                    height=self.new_height,
                    num_threads=1,
                    ctx=cpu(0),
                )
            else:
                vr = VideoReader(
                    fname,
                    width=self.new_width,
                    height=self.new_height,
                    num_threads=1,
                    ctx=cpu(0),
                )

        assert len(vr) == total_frame
        average_duration = total_frame // self.num_segment
        all_index = []
        ### TODO: try a continous sampling strategy, combined with global sampling ###
        ### https://github.com/wjn922/ReferFormer/blob/main/datasets/ytvos.py#L104 ###
        if average_duration > 0:
            all_index = list(
                np.multiply(list(range(self.num_segment)), average_duration)
                + np.random.randint(average_duration, size=self.num_segment)
            )  # TODO: check this, training with random sampling, try uniform sampling later

        else:
            all_index = [0] * (self.num_segment - total_frame) + list(
                range(total_frame)
            )

        all_index = list(np.array(all_index))
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer, all_index

    def load_video_and_mask(self, sample, frame_names, obj_id, total_frame):
        """Load video content using Decord"""
        ### here we add a sanity check ###
        # print(f'sample: {sample}, total_frame: {total_frame}, len(fnames): {len(frame_names)}')
        total_frame = min(total_frame, len(frame_names))

        average_duration = total_frame // self.num_segment
        all_index = []
        ### TODO: try a continous sampling strategy, combined with global sampling ###
        ### https://github.com/wjn922/ReferFormer/blob/main/datasets/ytvos.py#L104 ###
        if average_duration > 0:
            all_index = list(
                np.multiply(list(range(self.num_segment)), average_duration)
                + np.random.randint(average_duration, size=self.num_segment)
            )  # TODO: check this, training with random sampling, try uniform sampling later

        else:
            all_index = [0] * (self.num_segment - total_frame) + list(
                range(total_frame)
            )

        all_index = list(np.array(all_index))
        # print('all index:', all_index)

        all_images, all_masks = [], []
        for j in range(self.num_segment):
            frame_indx = all_index[j]
            frame_name = frame_names[j]
            img_path = os.path.join(
                self.prefix_dict["ReferYoutubeVOS"], sample, frame_name + ".jpg"
            )
            mask_path = os.path.join(
                self.mask_root_dict["ReferYoutubeVOS"], sample, frame_name + ".png"
            )
            # print(img_path, mask_path)
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = Image.open(mask_path).convert("P")
            mask = np.array(mask)
            mask = (mask == obj_id).astype(np.float32)
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
        return sum(len(videos) for videos in self.total_videos.values())

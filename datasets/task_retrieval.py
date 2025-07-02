import sys

sys.path.append("../")
import ast
import io
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

try:
    from petrel_client.client import Client

    has_client = True
except ImportError:
    has_client = False


# 最简单版本的实现，对于原始的 CLIP4CLIP 的 dataset 实现简化了不少，后续可以进一步优化
class TaskRetrievalDataset(Dataset):
    """TaskRetrieval train dataset implementation."""

    def __init__(
        self,
        anno_path,
        prefix="",
        mode="train",
        clip_len=8,
        num_segment=1,
        num_crop=1,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        data_dict=None,
        args=None,
    ):
        self.dataset_samples = pd.read_csv(anno_path)
        self.clip_len = clip_len
        self.mode = mode
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.args = args
        assert num_segment == 1

        self.transform = self.get_transform(
            mode=self.mode,
            init_model="siglip",
            image_res=self.crop_size,
            random_aug=False,
        )

        if data_dict is not None:
            rootdir = data_dict["root_dir"]
            trimmed30s = data_dict["trimmed30s"]
            is_paragraph_retrieval = data_dict["is_paragraph_retrieval"]
            self.params_dict = {
                "rootdir": rootdir,
                "trimmed30s": trimmed30s,
                "is_paragraph_retrieval": is_paragraph_retrieval,
            }
        else:
            self.params_dict = {
                "rootdir": {
                    "MSRVTT": "data/msr-vtt/MSRVTT_Videos/",
                },
                "trimmed30s": {
                    "MSRVTT": False,
                    "MSVD": False,
                    "ActivityNet": False,
                    "DiDeMo": True,
                    "LSMDC": False,
                    "VATEX": False,
                    "WebVid": False,
                },
                "is_paragraph_retrieval": {
                    "MSRVTT": False,
                    "MSVD": False,
                    "ActivityNet": True,
                    "DiDeMo": True,
                    "LSMDC": False,
                    "VATEX": False,
                    "WebVid": False,
                },
            }
        for retrieval_dataset in [
            "MSRVTT",
            "MSVD",
            "ActivityNet",
            "DiDeMo",
            "LSMDC",
            "VATEX",
            "WebVid",
        ]:
            if retrieval_dataset not in self.params_dict["rootdir"]:
                continue
            print(
                f'{retrieval_dataset} rootdir: {self.params_dict["rootdir"][retrieval_dataset]}, trimmed30s: {self.params_dict["trimmed30s"][retrieval_dataset]}, is_paragraph_retrieval: {self.params_dict["is_paragraph_retrieval"][retrieval_dataset]}'
            )

        self.client = None
        if has_client:
            self.client = Client("~/petreloss.conf")

    def copy_dataset(self, num_copies):
        if isinstance(self.dataset_samples, pd.DataFrame):
            self.dataset_samples = pd.concat(
                [self.dataset_samples] * num_copies, ignore_index=True
            )
        else:
            self.dataset_samples = self.dataset_samples * num_copies

    def __len__(self):
        return len(self.dataset_samples)

    def get_transform(self, mode, init_model, image_res, random_aug=False):
        assert init_model == "siglip"
        if init_model == "siglip":
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        normalize = transforms.Normalize(mean, std)
        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

        if random_aug:
            aug_transform = transforms.RandAugment()
        else:
            aug_transform = transforms.Lambda(lambda x: x)

        if mode == "train":
            transform = transforms.Compose(
                [
                    aug_transform,
                    # transforms.RandomResizedCrop(
                    #     image_res,
                    #     scale=(0.5, 1.0),
                    #     interpolation=InterpolationMode.BICUBIC,
                    #     antialias=True
                    # ),
                    transforms.Resize(self.short_side_size),
                    transforms.RandomCrop(size=(image_res, image_res)),
                    transforms.RandomHorizontalFlip(),
                    type_transform,
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_res, image_res),
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    type_transform,
                    normalize,
                ]
            )

        return transform

    def get_frame_indices(
        self,
        num_frames,
        vlen,
        sample="rand",
        fix_start=None,
        input_fps=1,
        max_num_frames=-1,
    ):
        if sample in ["rand", "middle"]:  # uniform sampling
            acc_samples = min(num_frames, vlen)
            # split the video into `acc_samples` intervals, and sample from each interval.
            intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
            ranges = []
            for idx, interv in enumerate(intervals[:-1]):
                ranges.append((interv, intervals[idx + 1] - 1))
            if sample == "rand":
                try:
                    frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
                except:
                    frame_indices = np.random.permutation(vlen)[:acc_samples]
                    frame_indices.sort()
                    frame_indices = list(frame_indices)
            elif fix_start is not None:
                frame_indices = [x[0] + fix_start for x in ranges]
            elif sample == "middle":
                frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            else:
                raise NotImplementedError

            if len(frame_indices) < num_frames:  # padded with last frame
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[: len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
        elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
            output_fps = float(sample[3:])
            duration = float(vlen) / input_fps
            delta = (
                1 / output_fps
            )  # gap between frames, this is also the clip length each frame represents
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            frame_indices = np.around(frame_seconds * input_fps).astype(int)
            frame_indices = [e for e in frame_indices if e < vlen]
            if max_num_frames > 0 and len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
                # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
        else:
            raise ValueError
        return frame_indices

    def loadvideo_decord(self, sample, dataset):
        """Load video content using Decord"""
        fname = sample
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

        vlen = len(vr)
        fps = float(vr.get_avg_fps())
        duration = vlen / fps

        if self.params_dict["trimmed30s"][dataset] and duration > 30:
            duration = 30
            vlen = int(30 * fps)

        frame_indices = self.get_frame_indices(
            num_frames=self.clip_len,
            vlen=vlen,
            sample="rand",
            fix_start=None,
            input_fps=fps,
            max_num_frames=-1,
        )
        vr.seek(0)
        buffer = vr.get_batch(frame_indices).asnumpy()
        buffer = torch.from_numpy(buffer).float()
        buffer = buffer.permute(0, 3, 1, 2)

        # print('name', fname, 'duration', duration, 'vlen', vlen, 'fps', fps, 'frame indices', frame_indices)
        return buffer

    def get_video_info(self, idx):
        sample = self.dataset_samples.iloc[idx]
        video_id = sample["video_id"]
        # start_second = sample['start_second']
        # end_second = sample['end_second']
        dataset = sample["dataset"]

        if dataset == "WebVid":
            text = sample["text"]
        else:
            texts = ast.literal_eval(sample["text"])
            text = random.choice(texts)

        # if dataset in ['LSMDC']:
        #     prefix_folder = video_id.split(start_second)[0].strip('_')
        #     video_id = f'{prefix_folder}/{video_id}.avi'
        # elif dataset in ["MSVD"]:
        #     video_id = video_id + '.avi'
        # else:
        #     video_id = video_id + '.mp4' if not video_id.endswith('.mp4') else video_id
        if dataset in ["LSMDC", "MSVD"]:
            video_id = video_id + ".avi"
        else:
            video_id = video_id + ".mp4" if not video_id.endswith(".mp4") else video_id

        return video_id, text, dataset

    def __getitem__(self, idx):
        try:
            video_id, text, dataset = self.get_video_info(idx)
            video_path = os.path.join(self.params_dict["rootdir"][dataset], video_id)

            buffer = self.loadvideo_decord(video_path, dataset)
            buffer = self.transform(buffer)

            return {
                "task_name": "TaskRetrieval",
                "task_input": {
                    "video": buffer,
                    "caption": text,
                    "video_id": video_id,
                    "label": 0,  # RESERVED
                },
            }
        except Exception as e:
            print(f"Error loading video {video_id} at index {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

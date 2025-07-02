import ast
import io
import json
import math
import os
import random
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms

from .random_erasing import RandomErasing
from .video_transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    Resize,
)
from .volume_transforms import ClipToTensor

try:
    from petrel_client.client import Client

    has_client = True
except ImportError:
    has_client = False


class TaskLocalizationDataset(Dataset):
    """Load your own video classification dataset.

    Training: Padding Dataset to 2304 per video.
    Validation: Sliding Window.
    Testing: Sliding Window.
    """

    def __init__(
        self,
        anno_path,
        prefix="",
        mode="train",
        clip_len=8,
        frame_sample_rate=2,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        test_num_segment=10,
        test_num_crop=3,
        label2id_path=None,
        sample_type="uniform",
        data_dict=None,
        args=None,
    ):
        self.anno_path = anno_path
        self.prefix = prefix
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        assert num_segment == 1

        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos."
            )

        self.data_list = pd.read_csv(
            self.anno_path
        )  # (vid, start_time, end_time, sentence)
        if label2id_path is not None:
            self.label2id = json.load(open(label2id_path, "r"))
        else:
            self.label2id = None
        if data_dict is not None:
            self.root_dict = data_dict["root_dir"]
        else:
            self.root_dict = {
                "CharadesSTA": "data/Charades_v1_480/",
            }
        self.video_templates = [
            "A video of an action where",
            "A video showing",
            "A video that demonstrates",
            "A video clip of",
            "A video recording of",
            "A video featuring",
            "A video capturing",
            "A video displaying",
            "A video presenting",
            "A video illustrating",
            "Watch this action where",
            "Look at this scene showing",
            "Here is a demonstration of",
            "This clip captures",
            "Observe this recording of",
            "This moment shows",
            "Witness this scene of",
        ]

        self.client = None
        if has_client:
            self.client = Client("~/petreloss.conf")

        self.sample_type = sample_type  ## [fixfps, uniform]

        print(f"Task localization using sample type {self.sample_type}")

        if mode == "train":
            if args.init_vit == "siglip":
                self.data_transform = Compose(
                    [
                        Resize(self.short_side_size, interpolation="bilinear"),
                        RandomCrop(size=(self.crop_size, self.crop_size)),
                        ClipToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
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
        if isinstance(self.data_list, pd.DataFrame):
            self.data_list = pd.concat([self.data_list] * num_copies, ignore_index=True)
        else:
            self.data_list = self.data_list * num_copies

    def __len__(self):
        return len(self.data_list)

    def get_video_info(self, index):
        data = self.data_list.iloc[index]
        video_id = data["video_id"]
        start_time = data["start_second"]
        end_time = data["end_second"]
        text = data["text"]
        relevant_windows = ast.literal_eval(data["relevant_windows"])
        dataset = data["dataset"]

        label = self.label2id[dataset][text]
        prefix = self.root_dict[dataset]

        if dataset == "TaCoS":
            video_id = video_id + "-cam-002.avi"
        elif dataset == "HACS":
            video_id = "v_" + video_id
            video_id = video_id + ".mp4" if not video_id.endswith(".mp4") else video_id
        else:
            video_id = video_id + ".mp4" if not video_id.endswith(".mp4") else video_id

        video_path = os.path.join(prefix, video_id)

        if dataset in ["FineAction", "HACS", "ActivityNet"]:
            template = random.choice(self.video_templates)
            text = f"{template} {text}"
        return video_path, start_time, end_time, text, dataset, relevant_windows, label

    def __getitem__(self, index):
        try:
            video_path, start_time, end_time, text, dataset, relevant_windows, label = (
                self.get_video_info(index)
            )

            if self.sample_type == "fixfps":
                buffer, label, duration, timestamps = self.loadvideo_decord_fixfps(
                    video_path,
                    start_time,
                    end_time,
                    self.clip_len,
                    relevant_windows,
                    label,
                )
            else:
                buffer, label, duration, timestamps = self.loadvideo_decord(
                    video_path,
                    start_time,
                    end_time,
                    self.clip_len,
                    relevant_windows,
                    label,
                )

            buffer = self.data_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)  # T C H W
            return {
                "task_name": "TaskLocalization",
                "task_input": {
                    "video": buffer,
                    "label": label,
                    "caption": str(text),
                    "duration": duration,
                    "dataset": dataset,
                },
            }
        except:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)

    def loadvideo_decord_fixfps(
        self, sample, start_time, end_time, clip_len, relevant_windows, label
    ):
        """Load video content using Decord, calculate duration, and generate time labels based on frame indices."""
        fname = sample

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

        num_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = num_frames / fps

        ### Case 1: end_time - start_time > clip_len, means the window is longer than the clip_len, we need to sample within this window
        if end_time - start_time > clip_len:
            start = start_time
            end = end_time
        ### Case 2: end_time - start_time <= clip_len, means the window is shorter than the clip_len, we need to expand the window to the clip_len
        else:
            window_center = (start_time + end_time) / 2
            half_clip_len = clip_len / 2

            if window_center - half_clip_len < 0:
                start = 0
                end = clip_len
            elif window_center + half_clip_len > duration:
                start = duration - clip_len
                end = duration
            else:
                start = window_center - half_clip_len
                end = window_center + half_clip_len

        # Convert time to frame indices
        start_frame = int(start * fps)
        end_frame = int(end * fps)

        # Ensure we have enough frames to sample
        if end_frame - start_frame < clip_len:
            end_frame = start_frame + clip_len

        # Calculate segment size for uniform sampling
        seg_size = (end_frame - start_frame) / clip_len

        # Randomly sample one frame from each segment
        selected_indices = []
        for i in range(clip_len):
            seg_start = int(start_frame + seg_size * i)
            seg_end = int(start_frame + seg_size * (i + 1))
            if seg_end > seg_start:
                frame_idx = random.randint(seg_start, seg_end)
            else:
                frame_idx = seg_start
            selected_indices.append(min(frame_idx, num_frames - 1))

        timestamps = [round(index / fps, 2) for index in selected_indices]

        vr.seek(0)
        buffer = vr.get_batch(selected_indices).asnumpy()

        if len(relevant_windows) <= 1:
            labels = torch.tensor(
                [label if start_time <= time <= end_time else 0 for time in timestamps],
                dtype=torch.float32,
            )
        else:
            labels = torch.zeros(len(timestamps), dtype=torch.float32)
            for start, end in relevant_windows:
                for time_idx, time in enumerate(timestamps):
                    if start <= time <= end:
                        labels[time_idx] = label

        # print('start: ', start_time, ' end:', end_time)
        # print('timestamps:', timestamps)
        # print('labels: ', labels)
        return buffer, labels, duration, timestamps

    def loadvideo_decord(
        self, sample, start_time, end_time, clip_len, relevant_windows, label
    ):
        """Load video content using Decord, calculate duration, and generate time labels based on frame indices."""
        fname = sample

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

        num_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = num_frames / fps

        if self.mode == "train":
            seg_size = max(0.0, float(num_frames - 1) / clip_len)
            max_frame = int(num_frames) - 1
            selected_indices = []
            for i in range(clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                idx = min(random.randint(start, end), max_frame)
                selected_indices.append(idx)
        else:
            selected_indices = np.linspace(0, num_frames - 1, clip_len, dtype=int)

        timestamps = [round(index / fps, 2) for index in selected_indices]

        vr.seek(0)
        buffer = vr.get_batch(selected_indices).asnumpy()

        if len(relevant_windows) <= 1:
            labels = torch.tensor(
                [
                    label if start_time <= time <= end_time else -1
                    for time in timestamps
                ],
                dtype=torch.float32,
            )
        else:
            labels = -torch.ones(len(timestamps), dtype=torch.float32)
            for start, end in relevant_windows:
                for time_idx, time in enumerate(timestamps):
                    if start <= time <= end:
                        labels[time_idx] = label

        # print('start: ', start_time, ' end:', end_time)
        # print('timestamps:', timestamps)
        # print('labels: ', labels)
        return buffer, labels, duration, timestamps

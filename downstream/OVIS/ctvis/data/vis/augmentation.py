# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC
import copy
import logging
import sys

import numpy as np
from PIL import Image
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from detectron2.data import transforms as T


class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
            self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice",
                                "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(
                    self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0  # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError(
                "Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0  # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


class RandomCropClip(T.Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size, clip_frame_cnt=2):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.
        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range",
                             "relative", "absolute", "absolute_range"]
        self._init(locals())
        self._cnt = 0

    def get_transform(self, image):
        h, w = image.shape[:2]  # 667, 500
        if self._cnt % self.clip_frame_cnt == 0:
            croph, cropw = self.get_crop_size((h, w))
            assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(
                self)

            h0 = np.random.randint(h - croph + 1)  # rand(124) -> 5
            w0 = np.random.randint(w - cropw + 1)  # rand(111) -> 634

            h1 = np.random.randint(h0, h - croph + 1)
            w1 = np.random.randint(w0, w - cropw + 1)

            x = np.sort(np.random.rand(self.clip_frame_cnt))

            h = h0 * x + h1 * (1 - x)
            w = w0 * x + w1 * (1 - x)
            h = np.round_(h).astype(np.int_)
            w = np.round_(w).astype(np.int_)

            if self._rand_range() < 0.5:
                h = h[::-1]
                w = w[::-1]

            self.hw_save = (h, w)
            self.crop_h_save, self.crop_w_save = croph, cropw
            self._cnt = 0  # avoiding overflow
        _h, _w = self.hw_save[0][0], self.hw_save[1][0]
        self._cnt += 1

        return T.CropTransform(_w, _h, self.crop_w_save, self.crop_h_save)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width
        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(
                min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(
                min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError(
                "Unknown crop type {}".format(self.crop_type))


def build_augmentation(cfg, is_train):
    logger = logging.getLogger(__name__)
    aug_list = []
    if is_train:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ms_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM if "by_clip" in cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING else 1
        # Crop
        if cfg.INPUT.CROP.ENABLED:
            aug_list.append(RandomCropClip(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE, clip_frame_cnt=ms_clip_frame_cnt))  # noqa

        aug_list.append(ResizeShortestEdge(min_size, max_size,
                        sample_style, clip_frame_cnt=ms_clip_frame_cnt))

        # Flip
        if cfg.INPUT.RANDOM_FLIP != "none":
            if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
            else:
                flip_clip_frame_cnt = 1

            aug_list.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlip(
                    horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (
                        cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    clip_frame_cnt=flip_clip_frame_cnt,
                )
            )

        # Additional augmentations : brightness, contrast, saturation, rotation
        augmentations = cfg.INPUT.AUGMENTATIONS
        if "brightness" in augmentations:
            aug_list.append(T.RandomBrightness(0.9, 1.1))
        if "contrast" in augmentations:
            aug_list.append(T.RandomContrast(0.9, 1.1))
        if "saturation" in augmentations:
            aug_list.append(T.RandomSaturation(0.9, 1.1))
        if "rotation" in augmentations:
            aug_list.append(
                T.RandomRotation(
                    [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], sample_style="range"
                )
            )
        if not cfg.INPUT.CROP.ENABLED:
            return aug_list
        else:
            aug_with_crop = copy.deepcopy(aug_list)
            del aug_list[0]
            return aug_with_crop, aug_list
    else:
        # Resize
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    return aug_list

# Copyright (c) 2023 ZJU-AIM.
import sys
import copy
import logging
import random
from typing import List, Union

import numpy as np
from PIL import Image
import torch
import pycocotools.mask as mask_util
from pycocotools import mask as coco_mask

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    polygons_to_bitmask,
)
from fvcore.transforms.transform import NoOpTransform

from ..vis.builtin import COCO_TO_YTVIS_2019, COCO_TO_YTVIS_2021, COCO_TO_OVIS
from .augmentation import build_augmentation, build_pseudo_augmentation


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    if len(polygons) == 0:  # noqa
        # coco api does not support empty polygons
        return np.zeros((height, width)).astype(np.bool)
    rles = mask_util.frPyObjects(polygons, height, width)  # noqa
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(np.bool)


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    boxes = (
        np.stack(
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
            self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=2
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
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

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
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
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


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING

    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())  # todo delete flip
        # tfm_gens.append(T.NoOpTransform())  # todo delete flip

    # 最终的形状主要由这个部分形状决定
    tfm_gens.append(ResizeShortestEdge(min_size, max_size, sample_style, clip_frame_cnt=1))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class RotateCOCOVideoDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
            self,
            is_train: bool,
            is_tgt: bool,
            *,
            augmentations: List[Union[T.Augmentation, T.Transform]],
            image_format: str,
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 5,
            src_dataset_name: str = "",
            tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train = is_train
        self.is_tgt = is_tgt
        self.augmentations = T.AugmentationList(augmentations)
        self.image_format = image_format
        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = COCO_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = COCO_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                src2tgt = COCO_TO_OVIS
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = False):
        if is_tgt:
            augs = build_augmentation(cfg, is_train)
        else:
            augs = build_pseudo_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame - self.sampling_frame_range)
            end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
        else:
            video_length = self.sampling_frame_num
            selected_idx = list(range(self.sampling_frame_num))

        dataset_dict["video_len"] = video_length
        dataset_dict["frame_idx"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _img_annos
                if obj.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = annotations_to_instances(annos, image_shape)
            if not self.is_tgt:
                instances.gt_classes = torch.tensor(
                    [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
                )
            instances.gt_ids = torch.tensor(_gt_ids)
            # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
            instances = filter_empty_instances(instances)
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            else:
                instances.gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)

            generate_flag = [False] * len(_gt_ids)
            instances.generate_flag = torch.tensor(generate_flag)
            dataset_dict["instances"].append(instances)

        return dataset_dict

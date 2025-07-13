import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata  # noqa
from detectron2.data.datasets.coco import register_coco_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2021_instances_meta,
    _get_ytvis_2019_instances_meta,
    _get_ovis_instances_meta)

COCO_TO_YTVIS_2019 = {
    1: 1, 2: 21, 3: 6, 4: 21, 5: 28, 7: 17, 8: 29, 9: 34, 17: 14, 18: 8, 19: 18, 21: 15, 22: 32, 23: 20, 24: 30, 25: 22,
    35: 33, 36: 33, 41: 5, 42: 27, 43: 40
}

COCO_TO_YTVIS_2021 = {
    1: 26, 2: 23, 3: 5, 4: 23, 5: 1, 7: 36, 8: 37, 9: 4, 16: 3, 17: 6, 18: 9, 19: 19, 21: 7, 22: 12, 23: 2, 24: 40,
    25: 18, 34: 14, 35: 31, 36: 31, 41: 29, 42: 33, 43: 34
}

COCO_TO_OVIS = {
    1: 1, 2: 21, 3: 25, 4: 22, 5: 23, 6: 25, 8: 25, 9: 24, 17: 3, 18: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 24: 10,
    25: 11,
}

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/train/JPEGImages",
                        "ytvis_2019/train_sub_3_video.json")
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json")
}

# ==== Predefined splits for YTVIS 2022 ===========
_PREDEFINED_SPLITS_YTVIS_2022 = {
    "ytvis_2022_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2022_val": ("ytvis_2022/valid/JPEGImages",
                       "ytvis_2022/valid.json"),
    "ytvis_2022_test": ("ytvis_2022/test/JPEGImages",
                        "ytvis_2022/test.json")
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                   "ovis/annotations/train.json"),
    "ovis_val": ("ovis/valid",
                 "ovis/annotations/valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations/test.json")
}

# ==== Predefined splits for COCO PSEUDO ===========
_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(
                root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(
                root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(
                root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2022(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2022.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(
                root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(
                root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_ytvis_2022(_root)
    register_all_ovis(_root)
    register_all_coco_video(_root)

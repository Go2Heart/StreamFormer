import os

from torchvision import transforms

# from .transforms import *
# from .masking_generator import TubeMaskingGenerator, RandomMaskingGenerator
# from .mae import VideoMAE
# from .mae_multi import VideoMAE_multi
from .kinetics import VideoClsDataset
from .kinetics_sparse import VideoClsDataset_sparse

# from .hmdb import HMDBVideoClsDataset, HMDBRawFrameClsDataset
# from .charades_sta import GroundingDataset
# from .msrvtt_dataset import MSRVTT_TestDataset, MSRVTT_TrainDataset
from .multi_task import MultiTaskDataset

# from .mevis import MEVISDataset
from .refer_youtube_vos import ReferYoutubeVOSDataset

# from .anet import ANetDataset
from .ssv2 import SSRawFrameClsDataset, SSVideoClsDataset

# from .qvhighlights import QVHighlightsGroundingDataset
# from .queryd import QuerYDGroundingDataset
# from .activitynet_captions import ActivityNetCaptionsGroundingDataset
# from .tacos import TaCoSGroundingDataset
# from .didemo import DiDeMoGroundingDataset
from .task_grounding import TaskGroundingDataset

# from .thumos14_grounding import Thumos14GroundingDataset
# from .activitynet_grounding import ActivityNetGroundingDataset
# from .fineaction_grounding import FineActionGroundingDataset
# from .hacs_grounding import HACSGroundingDataset
from .task_localization import TaskLocalizationDataset
from .task_refervos import TaskReferVOSDataset
from .task_retrieval import TaskRetrievalDataset
from .task_vis import TaskVISDataset

# from .thumos14 import Thumos14TALDataset, Thumos14FeatureDataset
# from .activitynet import ActivityNetTALDataset, ActivityNetFeatureDataset
# from .fineaction import FineActionTALDataset
# from .hacs import HACSTALDataset

# from .youtube_vis import YoutubeVisDataset
# from .lvvis import LVVisDataset
# from .coco_pseudo_video import CocoPseudoVISDataset
# from .webvid import WebVid_TestDataset, WebVid_TrainDataset


def build_multi_task_dataset(multitask_datasets, args):
    train_dataset_dict = {}
    test_dataset_dict = {}
    multi_task_config = {}
    for task, dataset in multitask_datasets.items():
        if "SSV2" == task:
            multi_task_config["SSV2"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"], "train.csv")
                elif mode == "test":
                    anno_path = os.path.join(mode_info["data_path"], "test.csv")
                elif mode == "validation":
                    anno_path = os.path.join(mode_info["data_path"], "val.csv")
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                SSv2_dataset = SSVideoClsDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    split=mode_info["split"],
                    mode=mode,
                    clip_len=1,
                    num_segment=mode_info["num_frames"],
                    test_num_segment=mode_info["num_segments"],
                    test_num_crop=mode_info["num_crops"],
                    num_crop=1 if not mode == "test" else 3,
                    keep_aspect_ratio=True,
                    crop_size=mode_info["input_size"],
                    short_side_size=mode_info["short_side_size"],
                    new_height=256,
                    new_width=320,
                    filename_tmpl=None,
                    label2id_path=mode_info["label2id_path"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = SSv2_dataset
                    multi_task_config["SSV2"]["label2id"] = SSv2_dataset.label2id
                elif (
                    mode == "test" or mode == "validation"
                ):  # consider as test set for now
                    test_dataset_dict[task] = SSv2_dataset
        elif "Kinetics" == task:
            multi_task_config["Kinetics"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"], "train.csv")
                elif mode == "test":
                    anno_path = os.path.join(mode_info["data_path"], "test.csv")
                elif mode == "validation":
                    # anno_path = os.path.join(mode_info["data_path"], 'val_subset.csv')
                    anno_path = os.path.join(mode_info["data_path"], "val.csv")
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                Kinetics_dataset = VideoClsDataset_sparse(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    split=mode_info["split"],
                    mode=mode,
                    clip_len=mode_info["num_frames"],
                    num_segment=1,
                    test_num_segment=mode_info["num_segments"],
                    test_num_crop=mode_info["num_crops"],
                    num_crop=1 if not mode == "test" else 3,
                    keep_aspect_ratio=True,
                    crop_size=mode_info["input_size"],
                    short_side_size=mode_info["short_side_size"],
                    new_height=256,
                    new_width=320,
                    label2id_path=mode_info["label2id_path"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = Kinetics_dataset
                    multi_task_config["Kinetics"][
                        "label2id"
                    ] = Kinetics_dataset.label2id
                elif (
                    mode == "test" or mode == "validation"
                ):  # consider as test set for now
                    test_dataset_dict[task] = Kinetics_dataset

        elif "TaskGrounding" == task:
            multi_task_config["TaskGrounding"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"], "train.csv")
                elif mode == "test":
                    anno_path = os.path.join(mode_info["data_path"], "test.csv")
                elif mode == "validation":
                    anno_path = os.path.join(mode_info["data_path"], "val.csv")
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                task_grounding_dataset = TaskGroundingDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    mode=mode,
                    clip_len=mode_info["num_frames"],
                    num_segment=1,
                    test_num_segment=mode_info["test_num_segment"],
                    test_num_crop=mode_info["test_num_crop"],
                    num_crop=1 if not mode == "test" else 3,
                    keep_aspect_ratio=True,
                    crop_size=mode_info["input_size"],
                    short_side_size=mode_info["short_side_size"],
                    new_height=256,
                    new_width=320,
                    sample_type=mode_info["sample_type"],
                    data_dict=mode_info["data_dict"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = task_grounding_dataset
                elif mode == "test" or mode == "validation":
                    test_dataset_dict[task] = task_grounding_dataset
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        elif "TaskLocalization" == task:
            multi_task_config["TaskLocalization"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"], "train.csv")
                elif mode == "test":
                    anno_path = os.path.join(mode_info["data_path"], "test.csv")
                elif mode == "validation":
                    anno_path = os.path.join(mode_info["data_path"], "val.csv")
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                task_localization_dataset = TaskLocalizationDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    mode=mode,
                    clip_len=mode_info["num_frames"],
                    num_segment=1,
                    test_num_segment=mode_info["test_num_segment"],
                    test_num_crop=mode_info["test_num_crop"],
                    num_crop=1 if not mode == "test" else 3,
                    keep_aspect_ratio=True,
                    crop_size=mode_info["input_size"],
                    short_side_size=mode_info["short_side_size"],
                    new_height=256,
                    new_width=320,
                    label2id_path=mode_info["label2id_path"],
                    sample_type=mode_info["sample_type"],
                    data_dict=mode_info["data_dict"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = task_localization_dataset
                    multi_task_config[task][
                        "label2id"
                    ] = task_localization_dataset.label2id
                elif mode == "test" or mode == "validation":
                    test_dataset_dict[task] = task_localization_dataset
                    multi_task_config[task][
                        "label2id"
                    ] = task_localization_dataset.label2id
                else:
                    raise ValueError(f"Unknown mode: {mode}")

        elif "TaskRetrieval" == task:
            multi_task_config["TaskRetrieval"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"], "train.csv")
                elif mode == "test":
                    anno_path = os.path.join(mode_info["data_path"], "test.csv")
                elif mode == "validation":
                    anno_path = os.path.join(mode_info["data_path"], "val.csv")
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                task_retrieval_dataset = TaskRetrievalDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    mode=mode,
                    clip_len=mode_info["num_frames"],
                    num_segment=1,
                    num_crop=1 if not mode == "test" else 3,
                    keep_aspect_ratio=True,
                    crop_size=mode_info["input_size"],
                    short_side_size=mode_info["short_side_size"],
                    new_height=256,
                    new_width=320,
                    data_dict=mode_info["data_dict"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = task_retrieval_dataset
                elif mode == "test" or mode == "validation":
                    test_dataset_dict[task] = task_retrieval_dataset
                else:
                    raise ValueError(f"Unknown mode: {mode}")
        elif "TaskVIS" == task:
            multi_task_config[task] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = mode_info["data_path"]
                elif mode == "test":
                    raise NotImplementedError
                    anno_path = os.path.join(mode_info["data_path"], "test.json")
                elif mode == "validation":
                    anno_path = mode_info["data_path"]
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                task_vis_dataset = TaskVISDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    split=",",
                    mode=mode,
                    num_segment=mode_info["num_frames"],
                    # test_num_segment=mode_info["test_num_segment"],
                    num_crop=1 if not mode == "test" else 3,
                    # test_num_crop=mode_info["test_num_crop"],
                    new_height=256,
                    new_width=320,
                    label2id_path=mode_info["label2id_path"],
                    data_dict=mode_info["data_dict"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = task_vis_dataset
                    multi_task_config[task]["label2id"] = task_vis_dataset.label2id
                elif mode == "test" or mode == "validation":
                    test_dataset_dict[task] = task_vis_dataset
        elif "TaskReferVOS" == task:
            multi_task_config["TaskReferVOS"] = {}
            for mode, mode_info in dataset.items():
                anno_path = None
                if mode == "train":
                    anno_path = os.path.join(mode_info["data_path"])
                elif mode == "test":
                    raise NotImplementedError
                    anno_path = os.path.join(mode_info["data_path"])
                elif mode == "validation":
                    anno_path = os.path.join(mode_info["data_path"])
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                task_refer_vos_dataset = TaskReferVOSDataset(
                    anno_path=anno_path,
                    prefix=mode_info["prefix"],
                    split=",",
                    mode=mode,
                    num_segment=mode_info["num_frames"],
                    num_crop=1 if not mode == "test" else 3,
                    new_height=256,
                    new_width=320,
                    data_dict=mode_info["data_dict"],
                    args=args,
                )
                if mode == "train":
                    train_dataset_dict[task] = task_refer_vos_dataset
                    multi_task_config[task][
                        "label2id"
                    ] = task_refer_vos_dataset.label2id
                elif mode == "test" or mode == "validation":
                    test_dataset_dict[task] = task_refer_vos_dataset
    if train_dataset_dict:
        # balance_sample_num = True
        # if 'TaskRetrieval' in train_dataset_dict and 'Kinetics' in train_dataset_dict and 'SSV2' in train_dataset_dict:
        #     balance_sample_num = False
        multitask_train_dataset = MultiTaskDataset(
            train_dataset_dict,
            balance_sample_num=True,
            scale=args.balance_sample_num_scale,
        )
    else:
        multitask_train_dataset = None
    if test_dataset_dict:
        multitask_test_dataset = MultiTaskDataset(
            test_dataset_dict, balance_sample_num=False
        )
    else:
        multitask_test_dataset = None
    return multitask_train_dataset, multitask_test_dataset, multi_task_config


if __name__ == "__main__":
    import argparse

    import yaml

    with open("scripts/dataset_metadata/ssv2_msrvtt.yaml", "r") as f:
        dataset_metadata = yaml.load(f, Loader=yaml.FullLoader)
    args = argparse.Namespace()
    args.init_vit = "siglip"
    print(dataset_metadata)
    multitask_train_dataset, multitask_test_dataset = build_multi_task_dataset(
        dataset_metadata, args
    )
    print(f"Train dataset length: {len(multitask_train_dataset)}")
    print(f"Test dataset length: {len(multitask_test_dataset)}")

import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
from pathlib import Path

import cv2
import mmcv
import numpy as np

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config  # noqa
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from ctvis import add_ctvis_config
from predictor import VisualizationDemo

# constants
VERSION2ANNOPATH = {
    '2019_val': 'datasets/ytvis_2019/valid.json',
    '2021_val': 'datasets/ytvis_2021/valid.json',
    '2022_val': 'datasets/ytvis_2022/valid.json',
    '2019_train': 'datasets/ytvis_2019/train.json',
    'ovis_val': 'datasets/ovis/annotations/valid.json'
}

VERSION2VIDEOPATH = {
    '2019_val': 'datasets/ytvis_2019/valid/JPEGImages',
    '2021_val': 'datasets/ytvis_2021/valid/JPEGImages',
    '2022_val': 'datasets/ytvis_2022/valid/JPEGImages',
    '2019_train': 'datasets/ytvis_2019/train/JPEGImages',
    'ovis_val': 'datasets/ovis/valid',
}


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_ctvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Demo for visualization on all images.")
    parser.add_argument(
        "--config-file",
        default="configs/ytvis_2019/CTVIS_R50.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--version',
        help='The version of YouTube-VIS or OVIS Dataset',
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--save-frames",
        action='store_true',
        help="Save frame level image outputs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    os.makedirs(args.output, exist_ok=True)

    dataset_version = args.version

    logger.info(f"Visualization on dataset {dataset_version}")

    anno_file_path = VERSION2ANNOPATH[dataset_version]
    IMG_PATH = Path(VERSION2VIDEOPATH[dataset_version])

    anno_file = mmcv.load(anno_file_path)

    all_frame_num = 0
    all_elapsed_time = 0.
    for i, video in enumerate(anno_file['videos']):
        frame_names = video['file_names']
        video_name = frame_names[0].split('/')[0]

        all_frame_num += len(frame_names)

        vid_frames = []
        for path in frame_names:
            img = read_image(IMG_PATH / path, format="BGR")
            vid_frames.append(img)

        start_time = time.perf_counter()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames, args.confidence_threshold)

        elapsed_time = time.perf_counter() - start_time
        all_elapsed_time += elapsed_time

        logger.info(
            f"Done Video [{i + 1:>3} / {len(anno_file['videos'])}], "
            f"Instances: {len(predictions['pred_scores'])}, "
            f"Frames: {len(frame_names)}, "
            f"FPS: {len(frame_names) / elapsed_time:.1f}, "
            f"Average_FPS: {all_frame_num / all_elapsed_time:.1f}")

        if args.save_frames:
            mmcv.mkdir_or_exist(Path(args.output) / video_name)
            for path, _vis_output in zip(frame_names, visualized_output):
                out_filename = os.path.join(args.output, path)
                _vis_output.save(out_filename)
        else:
            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()
    logger.info(f"All {len(anno_file['videos'])} videos in finished, "
                f"the AVERAGE_FPS is {all_frame_num / all_elapsed_time:.1f}")

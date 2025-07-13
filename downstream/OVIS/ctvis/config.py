from detectron2.config import CfgNode as CN


def add_ctvis_config(cfg):
    # Adjustable parameters
    cfg.MODEL.CLIP_NUM_FRAMES = 16
    cfg.MODEL.MASK_FORMER.REID_HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_REID_HEAD_LAYERS = 3

    cfg.TEST.NUM_CLIP_FRAMES = 16
    cfg.TEST.TEST_INTERPOLATE_CHUNK_SIZE = 5
    cfg.TEST.TEST_INSTANCE_CHUNK_SIZE = 5
    cfg.FIND_UNUSED_PARAMETERS = False
    cfg.SOLVER.DETECTOR_MULTIPLIER = 1.0
    cfg.TEST.TO_CPU_FRAMES = 40

    # Tracker
    cfg.MODEL.TRACKER = CN()
    cfg.MODEL.TRACKER.TRACKER_NAME = "SimpleTracker"
    cfg.MODEL.TRACKER.REID_WEIGHT = 2.0
    cfg.MODEL.TRACKER.AUX_REID_WEIGHT = 3.0
    cfg.MODEL.TRACKER.MATCH_METRIC = "bisoftmax"
    cfg.MODEL.TRACKER.MATCH_SCORE_THR = 0.2
    cfg.MODEL.TRACKER.TEMPORAL_SCORE_TYPE = "mean"  # mean or max
    cfg.MODEL.TRACKER.MATCH_TYPE = "greedy"
    cfg.MODEL.TRACKER.INFERENCE_SELECT_THR = 0.01
    cfg.MODEL.TRACKER.INIT_SCORE_THR = 0.01
    cfg.MODEL.TRACKER.MASK_NMS_THR = 0.6
    cfg.MODEL.TRACKER.FRAME_WEIGHT = True
    cfg.MODEL.TRACKER.NOISE_FRAME_NUM = 8
    cfg.MODEL.TRACKER.NOISE_FRAME_RATIO = 0.4
    cfg.MODEL.TRACKER.SUPPRESS_FRAME_NUM = 3
    cfg.MODEL.TRACKER.NONE_FRAME_NUM = 2

    cfg.MODEL.TRACKER.MEMORY_BANK = CN()
    cfg.MODEL.TRACKER.MEMORY_BANK.NUM_DEAD_FRAMES = 20
    cfg.MODEL.TRACKER.MEMORY_BANK.EMBED_TYPE = "similarity_guided"
    cfg.MODEL.TRACKER.MEMORY_BANK.maximum_cache = 10

    # dataloader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 10
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = []  # "brightness", "contrast", "saturation", "rotation"
    cfg.INPUT.COCO_PRETRAIN = None
    cfg.INPUT.PRETRAIN_SAME_CROP = False
    cfg.INPUT.CROP.CLIP_FRAME_CNT = 1
    cfg.INPUT.CROP.SAME_CROP = True
    cfg.INPUT.IMAGE_MODE = False

    # coco joint training
    cfg.DATASETS.DATASET_RATIO = [1.0, ]
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (320, 352, 392, 416, 448, 480, 512, 544, 576, 608, 640)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice"  # choice by clip | choice
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ["rotation"]
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)
    cfg.INPUT.PSEUDO.ROTATE_ANGLE = 15

    # contrastive learning plugin
    cfg.MODEL.CL_PLUGIN = CN()
    cfg.MODEL.CL_PLUGIN.CL_PLUGIN_NAME = "SimpleCLPlugin"
    cfg.MODEL.CL_PLUGIN.REID_WEIGHT = 2.
    cfg.MODEL.CL_PLUGIN.AUX_REID_WEIGHT = 3.
    cfg.MODEL.CL_PLUGIN.NUM_NEGATIVES = 99
    cfg.MODEL.CL_PLUGIN.FUSION_LOSS = False
    cfg.MODEL.CL_PLUGIN.BIO_CL = False
    cfg.MODEL.CL_PLUGIN.ONE_DIRECTION = True
    cfg.MODEL.CL_PLUGIN.MOMENTUM_EMBED = True
    # cfg.MODEL.CL_PLUGIN.NOISE_EMBED = True  # seems brings unstable training in small batch size (<16) and frame nums (<10)
    cfg.MODEL.CL_PLUGIN.NOISE_EMBED = False  

    cfg.MODEL.BACKBONE.PRETRAINED = None
    cfg.MODEL.BACKBONE.HIDDEN_SIZE = None
# Streamformer - Action Recognition

This is the repo for finetuning Streamformer on the action recognition task. The code is modified from [UMT](https://github.com/OpenGVLab/unmasked_teacher) and [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2)

## Installation
We recomend to install [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) by simply running ```pip install deepspeed```.

## Datasets

1. Download Kinetics 400 and Something-Something V2. The videos we used are downloaded from [OpenDataLab](https://opendatalab.com/). 

2. Prepare the annotation files. We provide the annotations [HERE](xxx).

## Training

**Notes before training:**
1. Chage `DATA_PATH` And `PREFIX` to your data path before running the scripts.
2. Chage `MODEL_PATH` and `PRETRAINED_CKPT` to your model path.
3. Set `--test_num_segment` and `--test_num_crop` for different evaluation strategies.

For training on K400 on 8GPUs, you can simply run
```
./exp/k400/streamformer_multitask_f16_res224.sh
```

On SSv2, you can simply run
```
./exp/ssv2/streamformer_multitask_lora_f16_res224.sh
```

## Main Results and checkpoints

### K400

|       method      | Top-1 Acc  (%) |  Top-5 Acc(%)  |   checkpoint   |
|  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  Streamformer |  82.4   | 95.5 | [Download](xxx) |

### SSv2

|       method      | Top-1 Acc  (%) |  Top-5 Acc(%)  |   checkpoint   |
|  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  Streamformer |  66.3   | 90.1 | [Download](xxx) |

## Acknowledgements

This codebase is built upon[UMT](https://github.com/OpenGVLab/unmasked_teacher) and [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2). Thanks for their great work.
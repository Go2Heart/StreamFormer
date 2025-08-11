# StreamFormer Online Action Detection

## Introduction

This is a PyTorch implementation for the downstream task Online Action Detection based on MAT "[`Memory-and-Anticipation Transformer for Online Action Understanding`](https://echo0125.github.io/mat/)".

## Environment

The code is developed with CUDA 10.2, ***Python >= 3.7.7***, ***PyTorch >= 1.7.1***

```
pip install -r requirements.txt
```

## Data Preparation 

### 1. Prepare the dataset

Prepare the THUMOS and TVSeries video datasets, including raw videos, metafiles, and ground-truth annotations. Please refer to [`LSTR`](https://github.com/amazon-research/long-short-term-transformer#data-preparation).

### 2. Extract video features using the following script
Change the following paths in **Streamformer/scripts/downstream_extract_oad_feature.sh**. 
```
MODEL_PATH='/PATH/TO/PRETRAINED/timesformer-siglip-16'
PRETRAINED_CKPT='/PATH/TO/CHECKPOINT/checkpoint-epoch.pth'

THUMOS_DATA_PATH='/PATH/TO/thumos-video'
THUMOS_ANNO_PATH='/PATH/TO/metadata/thumos'
THUMOS_SAVE_PATH='/PATH/TO/THUMOS14/streamformer_multitask_feature'

TVSERIES_DATA_PATH='/PATH/TO/tv_series/mkv_videos'
TVSERIES_ANNO_PATH='/PATH/TO/metadata/tvseries' 
TVSERIES_SAVE_PATH='/PATH/TO/TVSeries/streamformer_multitask_feature'
```

Then, run feature extraction in parallel on 8GPUs. 
```
cd Streamformer
./scripts/downstream_extract_oad_feature.sh
```

(Optionally) You can also download our pre-extracted feature from [HERE](xxxx).

Put all the files under data/THUMOS or data/TVSeries, and the file structure should be as:

* THUMOS dataset:
    ```
    data/THUMOS/
    ├── streamformer_multitask_feature/
    |   ├── video_validation_0000051.npy (of size L x 768)
    │   ├── ...
    ├── flow_kinetics_bninception/
    |   ├── video_validation_0000051.npy (of size L x 1024)
    |   ├── ...
    ├── target_perframe/
    |   ├── video_validation_0000051.npy (of size L x 22)
    |   ├── ...
    ```
* TVSeries dataset (optionally, you can generate optical flow for TVSeries.):
```
    data/TVSeries/
    ├── streamformer_multitask_feature/
    |   ├── Breaking_Bad_ep1.npy (of size L x 768)
    │   ├── ...
    ├── target_perframe/
    |   ├── Breaking_Bad_ep1.npy (of size L x 31)
    |   ├── ...
```

## Training and Inference

Train and inference MAT on pre-extracted THUMOS video features with flow on a single GPU.
```
./scripts/train_thumos_with_flow.sh
```
or visual feature only (without flow).
```
./scripts/train_thumos_without_flow.sh
```

Note that we use the first online evaluation mode in MAT for simplicity, where each test video is split into non-overlapping samples, and the model makes prediction on the all the frames in the short-term memory as if they were the latest frame. 

## Main Results and checkpoints

### THUMOS14

|       method      | visual feature   |  mAP  (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MAT (with flow)          |  Streamformer |  73.8    | [yaml](configs/THUMOS/MAT/streamformer_multitask_with_flow.yaml) | [Download](https://huggingface.co/StreamFormer/streamformer-downstream/blob/main/online_action_detection/thumos_best_with_flow.pt) |
|  MAT (without flow)          |    Streamformer   |  68.3    | [yaml](configs/THUMOS/MAT/streamformer_multitask_without_flow.yaml)      | [Download](https://huggingface.co/StreamFormer/streamformer-downstream/blob/main/online_action_detection/thumos_best_without_flow.pt) |

### TVSeries

|       method      | visual feature   |  mAP  (%)  |                             config                                                |   checkpoint   |
|  :--------------: |  :-------------:  |  :-----:  |  :-----------------------------------------------------------------------------:  |  :----------:  |
|  MAT (without flow)          |    Streamformer   |   87.8    | [yaml](configs/TVSeries/MAT/streamformer_multitask_without_flow.yaml)      | [Download](xxx) |
## Acknowledgements

This codebase is built upon [`MAT`](https://github.com/Echo0125/MAT-Memory-and-Anticipation-Transformer/tree/main/).

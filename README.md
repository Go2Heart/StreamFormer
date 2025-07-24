# Learning Streaming Video Representation via Multitask Training
[ICCV 2025 Oral] The official PyTorch implementation of Learning Streaming Video Representation via Multitask Training: https://arxiv.org/abs/2504.20041

<div style="line-height: 1;">
  <a href="https://go2heart.github.io/streamformer/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website-StreamFormer-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2504.20041" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv-StreamFormer-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center">
   <img src="./images/teaser.png">
   <img src="./images/main.png">
</div>

## TODO
- [x] Add instructions for quick start.
- [x] Add downstream evaluation pipelines.
- [ ] Release StreamFormer Checkpoints.
- [ ] Release Datasets Annotations.

## Quick Start
### Installation
```bash
conda create -n streamformer python=3.10
conda activate streamformer
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Pre-training 
Change some necessary paths in [scripts/pretrain_streamformer.sh](scripts/pretrain_streamformer.sh) and [dataset metadata](scripts/dataset_metadata/all.yaml), and run the scripts.
```bash
bash scripts/pretrain_streamformer.sh 
```


### Evaluations
#### 1. Action Recognition

Check the [README](downstream/AR/README.md) of Action Recognition.

#### 2. Online Action Detection

Check the [README](downstream/OAD/README.md) of Online Action Detection.

#### 3. OVIS

Follow the [README](downstream/OVIS/README.md) of CTVIS to install the corresponding environment.

Train StreamFormer for OVIS.
```bash
export DETECTRON2_DATASETS=/PATH/TO/VIS/DATA;
python -m downstream.OVIS.train_ctvis --resume --config-file downstream/OVIS/configs/ytvis_2019/CTVIS_Streamformer.yaml --num-gpus 4
```

#### 4. VideoQA

Follow the [README](downstream/VideoQA/README.md) of LLaVA-NeXT to install the corresponding environment.

Prepare the necessary data:
 - [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
 - [LLaVA-Next-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data)
 - [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)

Train StreamFormer checkpoint in 3 statges.
```bash
cd downstream/VideoQA
## stage 1 for pretraining
bash scripts/train/stage1_pretrain_timesformer_siglip_base.sh

## stage 2 for image-qa instruction tuning
bash scripts/train/stage2_direct_finetune_timesformer_siglip_base.sh 

## stage 3 for video-qa instruction tuning
bash scripts/train/stage3_direct_finetune_timesformer_video_only.sh 
```

## Ackowledgements
Thanks to the codebase of [UMT](https://github.com/OpenGVLab/unmasked_teacher/tree/main), [transformers](https://github.com/huggingface/transformers/tree/main), [MAT](https://github.com/Echo0125/MAT-Memory-and-Anticipation-Transformer), [CTVIS](https://github.com/KainingYing/CTVIS), [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT/tree/main?tab=readme-ov-file).

## Citations
If you find our work useful, please cite:
```bibtex
@misc{yan2025learning,
    title={Learning Streaming Video Representation via Multitask Training},
    author={Yibin Yan and Jilan Xu and Shangzhe Di and Yikun Liu and Yudi Shi and Qirui Chen and Zeqian Li and Yifei Huang and Weidi Xie},
    year={2025},
    eprint={2504.20041},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

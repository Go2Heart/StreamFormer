# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

from rekognition_online_action_detection.utils.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'streamformer_multitask_feature': 768,
    'streamformer_multitask_feature_so400m': 1152,
}


@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('TVSeries')
@FEATURE_HEADS.register('EK100')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in cfg.INPUT.MODALITY
            self.with_motion = 'visual' not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            visual_size = fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            motion_size = fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            if self.with_motion:
                self.motion_linear = nn.Sequential(
                    nn.Linear(motion_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            if self.with_visual:
                self.visual_linear = nn.Sequential(
                    nn.Linear(visual_size, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Sequential(
                    nn.Linear(2 * self.d_model, self.d_model),
                    nn.LayerNorm(self.d_model),
                    nn.ReLU(inplace=True)
                )
        else:
            if self.with_motion:
                self.motion_linear = nn.Identity()
            if self.with_visual:
                self.visual_linear = nn.Identity()
            if self.with_motion and self.with_visual:
                self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input):
        if self.with_visual and self.with_motion:
            visual_input = self.visual_linear(visual_input)
            motion_input  = self.motion_linear(motion_input)
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
            fusion_input = self.input_linear(fusion_input)
        elif self.with_visual:
            fusion_input = self.visual_linear(visual_input)
        elif self.with_motion:
            fusion_input = self.motion_linear(motion_input)

        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)

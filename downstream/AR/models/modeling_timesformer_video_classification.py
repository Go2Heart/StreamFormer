# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimeSformerV2 built upon PyTorch TimeSformer model."""
import collections
import collections.abc
import types
from typing import Dict, List, Optional, Tuple, Union
import torch

from einops import rearrange
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


import sys 
sys.path.append("/PATH/TO/Streamformer")
from Streamformer.models.modeling_timesformer_siglip import (
    TimesformerPreTrainedModel,
    TimeSformerAttention,
    TimesformerEmbeddingsSigLIP,
    TimesformerEncoder,
    TimesformerSiglipMultiheadAttentionPoolingHead
)


class TimesformerModelSigLIPForVideoClassification(TimesformerPreTrainedModel):
    """A TimeSFormer utilizing SigLIP's pretrained weights."""
    def __init__(self, config, num_classes=174, fc_dropout=0.0):
        super().__init__(config)
        self.config = config
        self.num_classes = num_classes

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = TimesformerSiglipMultiheadAttentionPoolingHead(config)

        ### classification head ###
        self.fc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc_dropout = nn.Dropout(config.hidden_dropout_prob) if fc_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'position_embeddings', 'time_embeddings', 'cls_token'}
        
    def get_num_layers(self):
        return len(self.encoder.layer)

    def add_lora_spatial(self):
        assert self.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("encoder.layer." + name)
                    module.attention._add_lora(32)
                    module.output._add_lora(32)
        print("Added LoRA to the following layers: ", name_list) 
    
    def frozen_spatial(self):
        assert self.encoder.layer[0].attention_type == "divided_space_time", "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if 'temporal_attention' not in name and 'attention' in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("encoder.layer." + name)
                    for param in module.attention.qkv.parameters():
                        param.requires_grad = False
                    for param in module.output.dense.parameters():
                        param.requires_grad = False
        print("Freezing spatial attention to the following layers: ", name_list) 
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values.shape[1] == 3:
            pixel_values = rearrange(pixel_values, 'b c t h w -> b t c h w')
        num_frames = pixel_values.shape[1]

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            num_frames=num_frames,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)
        pre_pool_output = sequence_output.view(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1)) # reshape to (batch_size * num_frames, patch_num, hidden_size)

        pooled_output = torch.mean(self.head(pre_pool_output).view(sequence_output.size(0), self.config.num_frames, sequence_output.size(-1)), 1, True).squeeze(1)
        out = self.fc_dropout(self.fc_norm(pooled_output))
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    pass
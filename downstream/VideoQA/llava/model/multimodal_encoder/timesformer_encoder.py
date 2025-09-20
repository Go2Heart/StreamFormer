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
import copy
import math
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    logging,
)

# from .configuration_timesformerV2 import TimesformerV2Config
""" TimeSformer model configuration"""

from functools import partial, reduce

from PIL import Image
from transformers.configuration_utils import PretrainedConfig
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/timesformer": "https://huggingface.co/facebook/timesformer/resolve/main/config.json",
}


class StreamformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TimesformerModel`]. It is used to instantiate a
    TimeSformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TimeSformer
    [facebook/timesformer-base-finetuned-k600](https://huggingface.co/facebook/timesformer-base-finetuned-k600)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_frames (`int`, *optional*, defaults to 8):
            The number of frames in each video.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_type (`str`, *optional*, defaults to `"divided_space_time"`):
            The attention type to use. Must be one of `"divided_space_time"`, `"space_only"`, `"joint_space_time"`.
        drop_path_rate (`float`, *optional*, defaults to 0):
            The dropout ratio for stochastic depth.

    Example:

    ```python
    >>> from transformers import TimesformerConfig, TimesformerModel

    >>> # Initializing a TimeSformer timesformer-base style configuration
    >>> configuration = TimesformerConfig()

    >>> # Initializing a model from the configuration
    >>> model = TimesformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "timesformer"

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_frames=8,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_type="divided_space_time",
        drop_path_rate=0,
        clip_config=None,
        enable_causal_temporal=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias

        self.attention_type = attention_type
        self.drop_path_rate = drop_path_rate
        self.clip_config = clip_config
        self.enable_causal_temporal = enable_causal_temporal


_CONFIG_FOR_DOC = "TimesformerConfig"
logger = logging.get_logger(__name__)

TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/timesformer-base-finetuned-k400",
    # See all TimeSformer models at https://huggingface.co/models?filter=timesformer
]

from torch import distributed as dist

has_distributed = True
import torch.nn.functional as F
from llava.utils import rank0_print


class TimesformerPatchEmbeddings(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, config):
        super().__init__()

        image_size = config.image_size
        patch_size = config.patch_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )

        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )  # H // Ph x W // Pw

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )  # Conv2d to get the patch embeddings

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = (
            pixel_values.shape
        )  # (B, T, 3, H, W)
        pixel_values = pixel_values.reshape(
            batch_size * num_frames, num_channels, height, width
        )  # (B*T, 3, H, W)

        embeddings = self.projection(pixel_values)  # (B*T, D, H', W')
        patch_width = embeddings.size(-1)
        embeddings = embeddings.flatten(2).transpose(
            1, 2
        )  # (B*T, H'*W', D) or (B*T, N, D)

        return embeddings, num_frames, patch_width


class TimesformerEmbeddingsSigLIP(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        embed_dim = config.hidden_size
        num_frames = config.num_frames
        drop_rate = config.hidden_dropout_prob
        attention_type = config.attention_type

        self.attention_type = attention_type
        self.patch_embeddings = TimesformerPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches

        # Positional Embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        if attention_type != "space_only":
            # Time Embeddings
            self.time_embeddings = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.position_embeddings.shape[1]
        if npatch == N and w == h:
            return self.position_embeddings
        pos_embed = self.position_embeddings.float()
        patch_pos_embed = pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embeddings.patch_size[0]
        h0 = h // self.patch_embeddings.patch_size[1]
        M = int(math.sqrt(N))  # Recover the number of patches in each dimension
        assert N == M * M
        kwargs = {}
        # if self.interpolate_offset:
        #     # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
        #     # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
        #     sx = float(w0 + self.interpolate_offset) / M
        #     sy = float(h0 + self.interpolate_offset) / M
        #     kwargs["scale_factor"] = (sx, sy)
        # else:
        # Simply specify an output size instead of a scale factor
        kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=True,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return patch_pos_embed.to(previous_dtype)

    def forward(self, pixel_values, past_key_values=None):
        batch_size, num_frames, _, height, width = pixel_values.shape
        embeddings, _, _ = self.patch_embeddings(pixel_values)  # (B*T, N, D)

        # Resizing the positional embeddings in case they don't match the input at inference
        embeddings = embeddings + self.interpolate_pos_encoding(
            embeddings, width, height
        )  # (B*T, N, D)
        embeddings = self.pos_drop(embeddings)

        # Time Embeddings
        if self.attention_type != "space_only":
            # embeddings = embeddings[:, 0:]
            _, num_patch, num_dim = embeddings.shape
            embeddings = (
                embeddings.reshape(
                    batch_size, num_frames, num_patch, num_dim
                )  # (B, T, N, D)
                .permute(0, 2, 1, 3)  # (B, N, T, D)
                .reshape(batch_size * num_patch, num_frames, num_dim)  # (B*N, T, D)
            )
            if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
                past_frames = past_key_values.get_seq_length()
                start_pos = past_frames
                end_pos = past_frames + num_frames
            else:
                start_pos = 0
                end_pos = num_frames
                
            total_needed_frames = end_pos
            original_time_frames = self.time_embeddings.size(1)
            
            # If we need more time embeddings than originally available, interpolate
            if total_needed_frames > original_time_frames:
                # Interpolate time embeddings to cover the full sequence
                time_embeddings = self.time_embeddings.transpose(1, 2)  # (1, D, original_frames)
                expanded_time_embeddings = nn.functional.interpolate(
                    time_embeddings, 
                    size=(total_needed_frames), 
                    mode="nearest",
                    align_corners=False
                )  # (1, D, total_needed_frames)
                expanded_time_embeddings = expanded_time_embeddings.transpose(1, 2)  # (1, total_needed_frames, D)
                
                # Extract the time embeddings for current frames
                current_time_embeddings = expanded_time_embeddings[:, start_pos:end_pos, :]  # (1, num_frames, D)
            else:
                # We have enough original time embeddings
                if end_pos <= original_time_frames:
                    # Direct indexing
                    current_time_embeddings = self.time_embeddings[:, start_pos:end_pos, :]  # (1, num_frames, D)
                else:
                    # Need to interpolate
                    time_embeddings = self.time_embeddings.transpose(1, 2)  # (1, D, original_frames)
                    expanded_time_embeddings = nn.functional.interpolate(
                        time_embeddings, 
                        size=(total_needed_frames), 
                        mode="nearest",
                        align_corners=False
                    )  # (1, D, total_needed_frames)
                    expanded_time_embeddings = expanded_time_embeddings.transpose(1, 2)  # (1, total_needed_frames, D)
                    current_time_embeddings = expanded_time_embeddings[:, start_pos:end_pos, :]
            embeddings = embeddings + current_time_embeddings
            embeddings = self.time_drop(embeddings)
            embeddings = embeddings.reshape(
                batch_size, num_patch * num_frames, num_dim
            )  # (B, N*T, D)

        return embeddings  # (B, N*T, D)


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(
    input: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (
        input.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device
    )
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->TimeSformer
class TimeSformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class TimesformerCausalSelfAttention(nn.Module):
    def __init__(self, config: StreamformerConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        self.head_dim = config.hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)


    def _add_lora(self, lora_rank):
        # freeze the qkv layer
        for param in self.qkv.parameters():
            param.requires_grad = False

        self.qkv_lora_a = nn.Linear(self.qkv.in_features, lora_rank, bias=False)
        self.qkv_lora_b = nn.Linear(lora_rank, self.qkv.out_features, bias=False)
        self.add_module("qkv_lora_a", self.qkv_lora_a)
        self.add_module("qkv_lora_b", self.qkv_lora_b)
        # move to device
        self.qkv_lora_a.to(self.qkv.weight.device)
        self.qkv_lora_b.to(self.qkv.weight.device)

        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.qkv_lora_a.weight, std=0.02)
        nn.init.zeros_(self.qkv_lora_b.weight)

        def lora_forward(self, hidden_states, output_attentions: bool = False):
            batch_size, hidden_size, num_channels = hidden_states.shape
            qkv = (
                (
                    self.qkv(hidden_states)
                    + self.qkv_lora_b(self.qkv_lora_a(hidden_states))
                )
                .reshape(
                    batch_size,
                    hidden_size,
                    3,
                    self.num_heads,
                    num_channels // self.num_heads,
                )  # B x C x 3 x num_heads x head_dim
                .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
            )
            query, key, value = qkv[0], qkv[1], qkv[2]

            attention_probs = (query @ key.transpose(-2, -1)) * self.scale
            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = self.attn_drop(attention_probs)

            context_layer = (
                (attention_probs @ value)
                .transpose(1, 2)
                .reshape(batch_size, hidden_size, num_channels)
            )

            outputs = (
                (context_layer, attention_probs)
                if output_attentions
                else (context_layer,)
            )

            return outputs

        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)

    def forward(
        self, 
        hidden_states,
        output_attentions: bool = False,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: Optional[int] = None
    ):
        batch_size, hidden_size, num_channels = hidden_states.shape  # (B*H*W) x T x C
        if self.training:
            use_cache =False
            past_key_value = None
        qkv = (
            self.qkv(hidden_states)
            .reshape(
                batch_size,
                hidden_size,
                3,
                self.num_heads,
                num_channels // self.num_heads,
            )  # B x T x 3 x num_heads x head_dim
            .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x T x head_dim
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, layer_idx,  cache_kwargs={"cache_position": cache_position})
            
        attention_scores = (query @ key.transpose(-2, -1)) * self.scale

        if past_key_value is not None:
            # For streaming: only mask future positions relative to current query positions
            total_seq_len = key.shape[-2]
            current_seq_len = query.shape[-2]
            
            # Create causal mask for the current query positions
            causal_mask = torch.tril(
                torch.ones(current_seq_len, total_seq_len, device=attention_scores.device, dtype=torch.bool)
            )
            
            # Adjust mask for current position in the sequence
            if cache_position is not None:
                start_pos = cache_position[0].item() if len(cache_position) > 0 else 0
                causal_mask = torch.zeros(current_seq_len, total_seq_len, device=attention_scores.device, dtype=torch.bool)
                for i in range(current_seq_len):
                    causal_mask[i, :start_pos + i + 1] = True
        else:
            # Standard causal mask for full sequence
            causal_mask = torch.tril(
                torch.ones(hidden_size, hidden_size, device=attention_scores.device, dtype=torch.bool)
            )

        attention_scores = attention_scores.masked_fill(
            ~causal_mask, float("-inf")
        )  # mask out future frames
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (
            (attention_probs @ value)
            .transpose(1, 2)
            .reshape(batch_size, hidden_size, num_channels)
        )

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class TimesformerSelfAttention(nn.Module):
    def __init__(self, config: StreamformerConfig):
        super().__init__()

        num_heads = config.num_attention_heads
        qkv_bias = config.qkv_bias
        attention_dropout_prob = config.attention_probs_dropout_prob

        self.num_heads = num_heads
        head_dim = config.hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout_prob)

    def _add_lora(self, lora_rank):
        # freeze the qkv layer
        for param in self.qkv.parameters():
            param.requires_grad = False

        self.qkv_lora_a = nn.Linear(self.qkv.in_features, lora_rank, bias=False)
        self.qkv_lora_b = nn.Linear(lora_rank, self.qkv.out_features, bias=False)
        self.add_module("qkv_lora_a", self.qkv_lora_a)
        self.add_module("qkv_lora_b", self.qkv_lora_b)
        # move to device
        self.qkv_lora_a.to(self.qkv.weight.device)
        self.qkv_lora_b.to(self.qkv.weight.device)

        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.qkv_lora_a.weight, std=0.02)
        nn.init.zeros_(self.qkv_lora_b.weight)

        def lora_forward(self, hidden_states, output_attentions: bool = False):
            batch_size, hidden_size, num_channels = hidden_states.shape
            qkv = (
                (
                    self.qkv(hidden_states)
                    + self.qkv_lora_b(self.qkv_lora_a(hidden_states))
                )
                .reshape(
                    batch_size,
                    hidden_size,
                    3,
                    self.num_heads,
                    num_channels // self.num_heads,
                )  # B x C x 3 x num_heads x head_dim
                .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
            )
            query, key, value = qkv[0], qkv[1], qkv[2]

            attention_probs = (query @ key.transpose(-2, -1)) * self.scale
            attention_probs = attention_probs.softmax(dim=-1)
            attention_probs = self.attn_drop(attention_probs)

            context_layer = (
                (attention_probs @ value)
                .transpose(1, 2)
                .reshape(batch_size, hidden_size, num_channels)
            )

            outputs = (
                (context_layer, attention_probs)
                if output_attentions
                else (context_layer,)
            )

            return outputs

        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)

    def forward(self, hidden_states, output_attentions: bool = False):
        batch_size, hidden_size, num_channels = hidden_states.shape  # B x N x C
        qkv = (
            self.qkv(hidden_states)
            .reshape(
                batch_size,
                hidden_size,
                3,
                self.num_heads,
                num_channels // self.num_heads,
            )  # B x C x 3 x num_heads x head_dim
            .permute(2, 0, 3, 1, 4)  # 3 x B x num_heads x C x head_dim
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attention_probs = (query @ key.transpose(-2, -1)) * self.scale
        attention_probs = attention_probs.softmax(dim=-1)
        attention_probs = self.attn_drop(attention_probs)

        context_layer = (
            (attention_probs @ value)
            .transpose(1, 2)
            .reshape(batch_size, hidden_size, num_channels)
        )

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _add_lora(self, lora_rank: int = 32):
        # freeze the dense layer
        for param in self.dense.parameters():
            param.requires_grad = False
        self.dense_lora_a = nn.Linear(self.dense.in_features, lora_rank, bias=False)
        self.dense_lora_b = nn.Linear(lora_rank, self.dense.out_features, bias=False)
        self.add_module("dense_lora_a", self.dense_lora_a)
        self.add_module("dense_lora_b", self.dense_lora_b)

        # move to device
        self.dense_lora_a.to(self.dense.weight.device)
        self.dense_lora_b.to(self.dense.weight.device)

        # initialize the lora projection a with gaussian noise and b with zeros
        nn.init.normal_(self.dense_lora_a.weight, std=0.02)
        nn.init.zeros_(self.dense_lora_b.weight)

        def lora_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.dense(hidden_states) + self.dense_lora_b(
                self.dense_lora_a(hidden_states)
            )
            hidden_states = self.dropout(hidden_states)

            return hidden_states

        # replace the forward method with the lora_forward method
        self.forward = types.MethodType(lora_forward, self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimeSformerCausalAttention(nn.Module):
    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.attention = TimesformerCausalSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            layer_idx=layer_idx
        ) 

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TimeSformerAttention(nn.Module):
    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.attention = TimesformerSelfAttention(config)
        self.output = TimesformerSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, output_attentions)

        attention_output = self.output(self_outputs[0])

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TimesformerIntermediate(nn.Module):
    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimesformerOutput(nn.Module):
    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TimesformerLayerSigLIP(nn.Module):
    def __init__(self, config: StreamformerConfig, layer_index: int) -> None:
        super().__init__()

        attention_type = config.attention_type

        drop_path_rates = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]  # stochastic depth decay rule
        drop_path_rate = drop_path_rates[layer_index]

        self.drop_path = (
            TimeSformerDropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
        self.attention = TimeSformerAttention(config)
        self.intermediate = TimesformerIntermediate(config)
        self.output = TimesformerOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.config = config
        self.attention_type = attention_type
        if attention_type not in [
            "divided_space_time",
            "space_only",
            "joint_space_time",
        ]:
            raise ValueError("Unknown attention type: {}".format(attention_type))

        # Temporal Attention Parameters
        if self.attention_type == "divided_space_time":
            self.temporal_layernorm = nn.LayerNorm(
                config.hidden_size, eps=config.layer_norm_eps
            )
            if config.enable_causal_temporal:
                self.temporal_attention = TimeSformerCausalAttention(config)
            elif not config.enable_causal_temporal:
                self.temporal_attention = TimeSformerAttention(config)
            self.temporal_dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.temporal_attention_gating = nn.Parameter(torch.tensor(0.0))
            # self.register_parameter("temporal_attention_gating", nn.Parameter(torch.tensor(0.0)))

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        output_attentions: bool = False,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: Optional[int] = None
    ):
        # hidden_states: (B, N*T, D)

        # num_frames = self.config.num_frames
        # num_patch_width = self.config.image_size // self.config.patch_size
        batch_size = hidden_states.shape[0]
        # num_spatial_tokens = (hidden_states.size(1)) // num_frames # siglip has no cls token
        # num_patch_height = num_spatial_tokens // num_patch_width

        if self.attention_type in ["space_only", "joint_space_time"]:
            self_attention_outputs = self.attention(
                self.layernorm_before(hidden_states),
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights

            hidden_states = hidden_states + self.drop_path(attention_output)

            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs
        elif self.attention_type == "divided_space_time":
            # Temporal Attention

            temporal_embedding = hidden_states  # siglip has no need to remove cls token
            temporal_embedding = temporal_embedding.reshape(
                -1, num_frames, temporal_embedding.shape[2]
            )  # (B*N, T, D)
            # temporal_embedding = temporal_embedding.reshape(
            #     batch_size, num_patch_height, num_patch_width, num_frames, temporal_embedding.shape[2]
            # ).reshape(batch_size * num_patch_height * num_patch_width, num_frames, temporal_embedding.shape[2]) # (B*N, T, D) s.t. temporal attension is applied to the same patch within frames

            # temporal_attention_outputs = self.temporal_attention(
            #     self.temporal_layernorm(temporal_embedding),
            # )
            if self.config.enable_causal_temporal and isinstance(self.temporal_attention, TimeSformerCausalAttention):
                temporal_attention_outputs = self.temporal_attention(
                    self.temporal_layernorm(temporal_embedding),
                    output_attentions=output_attentions,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    layer_idx=layer_idx
                )
            else:
                # Non-causal temporal attention doesn't use caching
                temporal_attention_outputs = self.temporal_attention(
                    self.temporal_layernorm(temporal_embedding),
                    output_attentions=output_attentions
                )
            attention_output = temporal_attention_outputs[
                0
            ]  # (B*N, T, D) only the attention output wanted

            residual_temporal = self.drop_path(attention_output)  # (B*N, T, D)
            # residual_temporal = residual_temporal.reshape(
            #     batch_size, num_patch_height, num_patch_width, num_frames,
            #     residual_temporal.shape[2]
            # ).reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_temporal.shape[2])
            residual_temporal = residual_temporal.reshape(
                batch_size, -1, residual_temporal.shape[2]
            )  # (B, N*T, D)
            residual_temporal = self.temporal_dense(residual_temporal)
            temporal_embedding = (
                hidden_states
                + self.temporal_attention_gating.tanh() * residual_temporal
            )  # (B, N*T, D)

            # Spatial
            # init_cls_token = hidden_states[:, 0, :].unsqueeze(1) # TODO check shape
            # cls_token = init_cls_token.repeat(1, num_frames, 1)
            # cls_token = cls_token.reshape(batch_size * num_frames, 1, cls_token.shape[2])
            spatial_embedding = temporal_embedding  # (B, N*T, D)
            # spatial_embedding = (
            #     spatial_embedding.reshape(
            #         batch_size, num_patch_height, num_patch_width, num_frames,
            #         spatial_embedding.shape[2]
            #     )
            #     .permute(0, 3, 1, 2, 4) # B x T x H x W x C
            #     .reshape(batch_size * num_frames, num_patch_height * num_patch_width, spatial_embedding.shape[2]) # (B x T) x (H x W) x C
            # )
            spatial_embedding = (
                spatial_embedding.reshape(
                    batch_size,
                    -1,
                    num_frames,
                    spatial_embedding.shape[2],  # (B, N, T, D)
                )
                .permute(0, 2, 1, 3)  # (B, T, N, D)
                .reshape(batch_size * num_frames, -1, spatial_embedding.shape[2])
            )  # (B*T, N, D)

            spatial_attention_outputs = self.attention(
                self.layernorm_before(spatial_embedding),
                output_attentions=output_attentions,
            )
            attention_output = spatial_attention_outputs[0]  # (B*T, N, D)
            outputs = spatial_attention_outputs[1:]  # TODO check shape here, null?

            residual_spatial = self.drop_path(attention_output)  # (B*T, N, D)

            # CLS token
            # cls_token = residual_spatial[:, 0, :]
            # cls_token = cls_token.reshape(batch_size, num_frames, cls_token.shape[1])
            # cls_token = torch.mean(cls_token, 1, True) # average over frames
            # residual_spatial = residual_spatial # siglip no need to remove cls token
            # residual_spatial = (
            #     residual_spatial.reshape(
            #         batch_size, num_frames, num_patch_height, num_patch_width, residual_spatial.shape[2]
            #     ) # B x T x H x W x C
            #     .permute(0, 2, 3, 1, 4) # B x H x W x T x C
            #     .reshape(batch_size, num_patch_height * num_patch_width * num_frames, residual_spatial.shape[2]) # B x (H x W x T) x C
            # )
            residual_spatial = (
                residual_spatial.reshape(
                    batch_size,
                    num_frames,
                    -1,
                    residual_spatial.shape[2],  # (B, T, N, D)
                )
                .permute(0, 2, 1, 3)  # (B, N, T, D)
                .reshape(batch_size, -1, residual_spatial.shape[2])
            )  # (B, N*T, D)
            residual = residual_spatial
            hidden_states = temporal_embedding

            # MLP
            hidden_states = hidden_states + residual
            layer_output = self.layernorm_after(hidden_states)
            layer_output = self.intermediate(layer_output)
            layer_output = self.output(layer_output)
            layer_output = hidden_states + self.drop_path(layer_output)

            outputs = (layer_output,) + outputs

            return outputs  # (B, N*T, D)


class TimesformerEncoder(nn.Module):
    def __init__(self, config: StreamformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                TimesformerLayerSigLIP(config, ind)
                for ind in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, N*T, D)
        num_frames: int,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    num_frames,
                    output_attentions,
                    past_key_values,
                    use_cache,
                    cache_position,
                    i
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, 
                    num_frames, 
                    output_attentions,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    layer_idx=i
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        if use_cache:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                past_key_values=past_key_values,
            )
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )


class TimesformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = StreamformerConfig
    base_model_prefix = "timesformer"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddingsSigLIP):
            nn.init.normal_(
                module.position_embeddings, std=self.config.initializer_range
            )
            module.patch_embeddings.apply(self._init_weights)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        else:
            if hasattr(module, "weight") and module.weight is not None:
                if module.weight.dim() > 1:
                    nn.init.kaiming_uniform_(module.weight)
                else:
                    nn.init.normal_(
                        module.weight, mean=0.0, std=self.config.initializer_range
                    )

            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0)


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class TimesformerSiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_state):
        # hidden_state: (B*T, N, D)
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[
            0
        ]  # (B*T, 1, D)

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]  # (B*T, D)


class TimesformerModelSigLIP(TimesformerPreTrainedModel):
    """A TimeSFormer utilizing SigLIP's pretrained weights."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)
        # self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = TimesformerSiglipMultiheadAttentionPoolingHead(config)

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

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)
        pre_pool_output = sequence_output.view(
            sequence_output.size(0) * self.config.num_frames,
            -1,
            sequence_output.size(-1),
        )  # reshape to (batch_size * num_frames, patch_num, hidden_size)
        pooled_output = torch.mean(
            self.head(pre_pool_output).view(
                sequence_output.size(0),
                self.config.num_frames,
                sequence_output.size(-1),
            ),
            1,
            True,
        ).squeeze(1)

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TimesformerMultiTaskingModelSigLIP(TimesformerPreTrainedModel):
    """A TimeSFormer utilizing SigLIP's pretrained weights."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TimesformerEmbeddingsSigLIP(config)
        self.encoder = TimesformerEncoder(config)
        # self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.head = TimesformerSiglipMultiheadAttentionPoolingHead(config)
        self.add_lora_spatial()
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

    def add_lora_spatial(self):
        assert (
            self.encoder.layer[0].attention_type == "divided_space_time"
        ), "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if "temporal_attention" not in name and "attention" in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    module.attention._add_lora(32)
                    module.output._add_lora(32)
        print("Added LoRA to the following layers: ", name_list)

    def frozen_spatial(self):
        assert (
            self.encoder.layer[0].attention_type == "divided_space_time"
        ), "Please use divided_space_time attention type"
        name_list = []
        for name, module in self.encoder.layer.named_modules():
            if "temporal_attention" not in name and "attention" in name:
                if isinstance(module, TimeSformerAttention):
                    name_list.append("timesformer.encoder.layer." + name)
                    for param in module.attention.qkv.parameters():
                        param.requires_grad = False
                    for param in module.attention.dense.parameters():
                        param.requires_grad = False
        print("Freezing spatial attention to the following layers: ", name_list)

    def forward(
        self,
        pixel_values: torch.FloatTensor,  # (B, T, 3, H, W)
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        B, T, _, _, _ = pixel_values.shape
        if use_cache and cache_position is None:
            if past_key_values is None:
                cache_position = torch.arange(0, T, dtype=torch.long, device=pixel_values.device)
            else:
                cache_position = torch.arange(
                    past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else 0,
                    past_key_values.get_seq_length() + T if hasattr(past_key_values, 'get_seq_length') else T,
                    dtype=torch.long,
                    device=pixel_values.device
                )
        
        embedding_output = self.embeddings(pixel_values, past_key_values=past_key_values)  # (B, N*T, D)
        encoder_outputs = self.encoder(
            embedding_output,
            num_frames=T,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.post_layernorm(sequence_output)  # (B, N*T, D)
        # pre_pool_output = sequence_output.view(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1)) # reshape to (batch_size * num_frames, patch_num, hidden_size)
        # FIX BUG HERE with reshape
        # num_patch_width = self.config.image_size // self.config.patch_size
        # num_patch_height = (encoder_outputs[0].size(1)) // self.config.num_frames // num_patch_width
        # pre_pool_output = sequence_output.reshape(sequence_output.size(0), num_patch_height, num_patch_width,self.config.num_frames, sequence_output.size(-1)).permute(0, 3, 1, 2, 4).reshape(sequence_output.size(0) * self.config.num_frames, -1, sequence_output.size(-1))
        pre_pool_output = (
            sequence_output.reshape(B, -1, T, sequence_output.size(-1))  # (B, N, T, D)
            .permute(0, 2, 1, 3)  # (B, T, N, D)
            .reshape(B * T, -1, sequence_output.size(-1))
        )  # (B*T, N, D)
        # the reduce step for frame dimension should be done in each task head
        pooled_output = self.head(pre_pool_output).reshape(  # (B*T, D)
            B, T, sequence_output.size(-1)
        )  # (B, T, D)
        # pooled_output = pooled_output.to(torch.float32) # TODO why float 16 is auto-transformed here
        # pooled_output = torch.mean(self.head(pre_pool_output).view(sequence_output.size(0), self.config.num_frames, sequence_output.size(-1)), 1, True).squeeze(1)
        sequence_output = (
            sequence_output.reshape(B, -1, T, sequence_output.size(-1))
            .permute(0, 2, 1, 3)
            .reshape(B, T, -1, sequence_output.size(-1))
        )
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return BaseModelOutputWithPast(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,  # L x (B, N*T, D)
            attentions=encoder_outputs.attentions,
            past_key_values=encoder_outputs.past_key_values if use_cache else None,
        )


class TimesformerImageProcessor:
    """
    Image processor for Timesformer, modified from SigLipImageProcessor
    """

    def __init__(
        self,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        size=(384, 384),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size if crop_size is not None else {"height": 384, "width": 384}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(
                resize,
                size=self.size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class TimesformerVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        # self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(
                f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True."
            )
            self.load_model()
        elif (
            hasattr(vision_tower_cfg, "mm_tunable_parts")
            and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts
        ):
            rank0_print(
                f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`."
            )
            self.load_model()
        else:
            rank0_print(f"Force loading checkpoint!!!")  # TODO yibin debug
            self.load_model()
            # self.cfg_only = self.config
        
        self.streaming_mode = getattr(vision_tower_cfg, "streaming_mode", False)
        if self.streaming_mode:
            context_length = getattr(vision_tower_cfg, "context_length", 16)
            print(f"Streamformer-Timesformer Using streaming mode with context length: {context_length}")
            self.context_length = context_length
            self.past_key_values = None
            self.hidden_states = None
    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(
                "{} is already loaded, `load_model` called again, skipping.".format(
                    self.vision_tower_name
                )
            )
            return

        # self.vision_tower = SigLipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower = TimesformerMultiTaskingModelSigLIP.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.config = self.vision_tower.config
        self.image_processor = TimesformerImageProcessor(
            size=(self.config.image_size, self.config.image_size),
            crop_size={
                "height": self.config.image_size,
                "width": self.config.image_size,
            },
        )
        # del self.vision_tower.vision_model.encoder.layers[-1:]
        # self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        
    def clear_cache(self):
        self.hidden_states = None
        self.past_key_values = None

    def forward(self, images):
        if self.streaming_mode:
            # self.clear_cache() # for eval donot use cache
            # images should be a streaming video frame of shape (1, T, C, H, W)
            outputs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, use_cache=True, past_key_values=self.past_key_values, cache_position=None)
            # saving the hidden states for streaming
            self.hidden_states = torch.cat([self.hidden_states, outputs.last_hidden_state], dim=1) if self.hidden_states is not None else outputs.last_hidden_state # concating along the time dimension
            self.past_key_values = outputs.past_key_values
            
            image_features = self.hidden_states[:, -self.context_length:, :].to(images.dtype)
            
            # image_features = outputs.last_hidden_state.to(images.dtype)
            return image_features
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            # images should be a tensor of shape (B, T, C, H, W)
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            # image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            image_features = image_forward_outs.last_hidden_state.to(
                images.dtype
            )  # (B, T, N, D)
            # assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size

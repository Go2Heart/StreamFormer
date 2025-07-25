import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch.nn import functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.structures import BitMasks
from detectron2.layers import Conv2d

from ....mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .base_arch import PositionEmbeddingSine, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP


@TRANSFORMER_DECODER_REGISTRY.register()
class CLMultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    if state_dict[k].shape[0] != self.num_queries:
                        del state_dict[k]
                        continue
                    newk = k.replace("static_query", "query_feat")

                # if "query_embed" in k:
                #     if state_dict[k].shape[0] != self.num_queries:
                #         del state_dict[k]
                #         continue

                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            reid_hidden_dim,
            num_reid_head_layers,
            **kwargs
    ):

        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.hidden_dim = hidden_dim
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(
                    Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        if num_reid_head_layers > 0:
            self.reid_embed = MLP(
                self.hidden_dim, reid_hidden_dim, self.hidden_dim, num_reid_head_layers)
            for layer in self.reid_embed.layers:
                weight_init.c2_xavier_fill(layer)
        else:
            self.reid_embed = nn.Identity()  # do nothing

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = dict()
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["reid_hidden_dim"] = cfg.MODEL.MASK_FORMER.REID_HIDDEN_DIM
        ret["num_reid_head_layers"] = cfg.MODEL.MASK_FORMER.NUM_REID_HEAD_LAYERS

        return ret

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(
                2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_query = []
        predictions_embed = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = self.forward_prediction_heads(output,
                                                                                                             mask_features,
                                                                                                             attn_mask_target_size=size_list[
                                                                                                                 0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_query.append(outputs_query)
        predictions_embed.append(outputs_embed)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) ==
                                  attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask, outputs_query, outputs_embed = self.forward_prediction_heads(output,
                                                                                                                 mask_features,
                                                                                                                 attn_mask_target_size=size_list[
                                                                                                                     (
                                                                                                                         i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_query.append(outputs_query)
            predictions_embed.append(outputs_embed)

        assert len(predictions_class) == self.num_layers + 1

        # get pred_bbox here
        h, w = outputs_mask.shape[-2:]
        pred_bboxes = BitMasks(outputs_mask.flatten(0, 1) > 0).get_bounding_boxes().to(attn_mask.device).tensor.reshape(
            *outputs_mask.shape[:2], 4)
        bbox_scale = pred_bboxes.new_tensor([w, h, w, h])
        pred_bboxes = pred_bboxes / bbox_scale

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_queries': predictions_query[-1],
            'pred_embeds': predictions_embed[-1],
            'pred_bboxes': pred_bboxes,
            'query_pos_embeds': self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1),
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_query,
                predictions_embed
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)

        # for track
        reid_embed = self.reid_embed(decoder_output)
        outputs_mask = torch.einsum(
            "bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(
            outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, decoder_output, reid_embed

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_queries, outputs_embeds):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b,
                    "pred_queries": c, "pred_embeds": d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_seg_masks[:-1],
                                      outputs_queries[:-1], outputs_embeds[:-1])
            ]
        else:
            return [{"pred_masks": b, "pred_queries": c, "pred_embeds": d}
                    for b, c, d in zip(outputs_seg_masks[:-1], outputs_queries[:-1], outputs_embeds[:-1])]

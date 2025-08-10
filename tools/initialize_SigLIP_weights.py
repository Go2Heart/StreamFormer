import json
from argparse import ArgumentParser

import torch
from transformers import SiglipModel

from models import (
    StreamformerConfig,
    StreamformerForMultiTaskingSigLIP,
)


def main(args):
    siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")

    config = StreamformerConfig(
        clip_config=siglip_model.config, enable_causal_temporal=True
    )
    model = StreamformerForMultiTaskingSigLIP(config, None)

    loaded_dict = {}
    for name, param in model.named_parameters():
        loaded_dict[name] = False

    clip_vision_model = siglip_model.vision_model
    for name, param in clip_vision_model.named_parameters():
        if "embeddings.class_embedding" in name:
            model.state_dict()["timesformer.embeddings.cls_token"].copy_(
                param.unsqueeze(0).unsqueeze(0)
            )
            loaded_dict["timesformer.embeddings.cls_token"] = True
        elif "embeddings.patch_embedding.weight" in name:
            model.state_dict()[
                "timesformer.embeddings.patch_embeddings.projection.weight"
            ].copy_(param)
            loaded_dict["timesformer.embeddings.patch_embeddings.projection.weight"] = (
                True
            )
        elif "embeddings.patch_embedding.bias" in name:
            model.state_dict()[
                "timesformer.embeddings.patch_embeddings.projection.bias"
            ].copy_(param)
            loaded_dict["timesformer.embeddings.patch_embeddings.projection.bias"] = (
                True
            )
        elif "embeddings.position_embedding" in name:
            model.state_dict()["timesformer.embeddings.position_embeddings"].copy_(
                param
            )
            loaded_dict["timesformer.embeddings.position_embeddings"] = True
        elif "pre_layrnorm.weight" in name:
            model.state_dict()["timesformer.pre_layernorm.weight"].copy_(param)
            loaded_dict["timesformer.pre_layernorm.weight"] = True
        elif "pre_layrnorm.bias" in name:
            model.state_dict()["timesformer.pre_layernorm.bias"].copy_(param)
            loaded_dict["timesformer.pre_layernorm.bias"] = True
        elif "post_layernorm.weight" in name:
            model.state_dict()["timesformer.post_layernorm.weight"].copy_(param)
            loaded_dict["timesformer.post_layernorm.weight"] = True
        elif "post_layernorm.bias" in name:
            model.state_dict()["timesformer.post_layernorm.bias"].copy_(param)
            loaded_dict["timesformer.post_layernorm.bias"] = True
        elif "encoder.layers" in name:
            layer_id = name.split(".")[2]
            if "self_attn" in name:
                x = name.split(".")[4]
                if "k_proj.weight" in name:
                    # merge qkv weights
                    q_name = name.replace(x, "q_proj")
                    k_name = name.replace(x, "k_proj")
                    v_name = name.replace(x, "v_proj")
                    qkv = torch.cat(
                        [
                            clip_vision_model.state_dict()[q_name],
                            clip_vision_model.state_dict()[k_name],
                            clip_vision_model.state_dict()[v_name],
                        ],
                        dim=0,
                    )
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.attention.qkv.weight"
                    ].copy_(qkv)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.attention.qkv.weight"
                    ] = True
                elif "k_proj.bias" in name:
                    # merge qkv bias
                    q_name = name.replace(x, "q_proj")
                    k_name = name.replace(x, "k_proj")
                    v_name = name.replace(x, "v_proj")
                    qkv = torch.cat(
                        [
                            clip_vision_model.state_dict()[q_name],
                            clip_vision_model.state_dict()[k_name],
                            clip_vision_model.state_dict()[v_name],
                        ],
                        dim=0,
                    )
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.attention.qkv.bias"
                    ].copy_(qkv)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.attention.qkv.bias"
                    ] = True
                elif "out_proj.weight" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.output.dense.weight"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.output.dense.weight"
                    ] = True
                elif "out_proj.bias" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.output.dense.bias"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".attention.output.dense.bias"
                    ] = True

            # copy layer_norm1 to layernorm_before and layer_norm2 to layernorm_after
            # TODO check again
            elif "layer_norm1" in name:
                if "weight" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_before.weight"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_before.weight"
                    ] = True
                elif "bias" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_before.bias"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_before.bias"
                    ] = True

            elif "layer_norm2" in name:
                if "weight" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_after.weight"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_after.weight"
                    ] = True
                elif "bias" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_after.bias"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".layernorm_after.bias"
                    ] = True
            elif "mlp" in name:
                if "fc1.weight" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".intermediate.dense.weight"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".intermediate.dense.weight"
                    ] = True
                elif "fc1.bias" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".intermediate.dense.bias"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer."
                        + layer_id
                        + ".intermediate.dense.bias"
                    ] = True
                elif "fc2.weight" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer." + layer_id + ".output.dense.weight"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer." + layer_id + ".output.dense.weight"
                    ] = True
                elif "fc2.bias" in name:
                    model.state_dict()[
                        "timesformer.encoder.layer." + layer_id + ".output.dense.bias"
                    ].copy_(param)
                    loaded_dict[
                        "timesformer.encoder.layer." + layer_id + ".output.dense.bias"
                    ] = True
        elif "head" in name:
            model.state_dict()["timesformer." + name].copy_(param)
            loaded_dict["timesformer." + name] = True
        else:
            print(name)
            print(param.shape)

    for name, param in model.named_parameters():
        if (
            "temporal_dense" in name
            or "temporal_attention" in name
            or "time_embeddings" in name
        ) and "temporal_attention_gating" not in name:
            print(name)
            print(param.shape)
            # random initialization using xavier
            # torch.nn.init.xavier_normal_(param)
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            loaded_dict[name] = True
    # for name, param in model.named_parameters():
    #     # if 'temporal' in name and "layernorm" not in name:
    #     if ('temporal' in name or 'time_embeddings' in name) and "layernorm" not in name:
    #         print(name)
    #         print(param.shape)
    #         model.state_dict()[name].zero_()
    #         loaded_dict[name] = True

    model.state_dict()["logit_scale"].copy_(
        siglip_model.state_dict()["logit_scale"].squeeze(0)
    )
    model.state_dict()["logit_bias"].copy_(
        siglip_model.state_dict()["logit_bias"].squeeze(0)
    )

    loaded_dict["logit_scale"] = True
    loaded_dict["logit_bias"] = True

    # copy text encoder weights
    text_model = siglip_model.text_model
    for name, param in text_model.named_parameters():
        name = "text_encoder.text_model." + name
        model.state_dict()[name].copy_(param)
        loaded_dict[name] = True

    loaded, unloaded = [], []
    print("Not loaded:")
    for k, v in loaded_dict.items():
        if v == False:
            unloaded.append(k)
            print(k)
        else:
            loaded.append(k)

    # model.save_pretrained("timesformer-siglip")
    model.save_pretrained(args.save_name)
    print(f"Done saving all pretrained checkpoints to {args.save_name}")
    json.dump(
        {"Loaded": loaded, "Not loaded": unloaded},
        open(f"{args.save_name}/params.json", "w"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--enable-causal-temporal",
        type=bool,
        default=True,
        help="whether to enable causal temporal attention",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="timesformer-siglip",
        help="name to save the pretrained checkpoint",
    )
    parser.add_argument(
        "--attention_type",
        default="divide_space_time",
        type=str,
        choices=["divided_space_time", "hierachical_space_time"],
    )
    args = parser.parse_args()
    main(args)

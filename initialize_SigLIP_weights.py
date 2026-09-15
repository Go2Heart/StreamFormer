"""Initialize a Streamformer (SigLIP) checkpoint from the SigLIP vision model.

This script transfers the SigLIP vision weights into a Streamformer model so it
can be fine-tuned for the downstream tasks.

Two earlier bugs in this script are fixed here:

1. ``KeyError: 'timesformer.embeddings.cls_token'``: the SigLIP vision model
   uses attention pooling (it has a learnable ``class_embedding`` query) but the
   Streamformer target model pools the pooled frame tokens and defines no
   ``cls_token`` in its embeddings. The class token is therefore *not* copied
   into the target model; only the positional embeddings are transferred.
2. Inefficient ``model.state_dict()`` calls: the target state dictionary is now
   built once, outside the parameter loop, instead of being (re)created on every
   iteration, which previously caused large and unnecessary overhead.
"""

import argparse
import os
import re
import warnings

import torch
from transformers import AutoModel

try:
    from models import StreamformerForMultiTaskingSigLIP
except ImportError:  # pragma: no cover - depends on how the repo is invoked
    from models.streamformer import StreamformerForMultiTaskingSigLIP


def _target_key(name):
    """Map a SigLIP vision model parameter name to its Streamformer target name.

    ``None`` is returned for parameters that should not be copied (the class
    embedding query is intentionally skipped because the target model has no
    corresponding ``cls_token``).
    """
    # The class token query has no counterpart on the target side: Streamformer
    # pools the frame tokens and does not define ``timesformer.embeddings.cls_token``.
    if re.match(r"^class_embedding(\..+)?$", name):
        return None

    # Positional embeddings (absolute or relative).
    if re.match(r"^embeddings\.position", name):
        return "timesformer.embeddings." + name

    # Transformer encoder blocks.
    encoder_pattern = re.compile(
        r"^encoder\.layers\.(\d+)\.((?:attention|mlp)\..+|norm[12]\.weight)$"
    )
    m = encoder_pattern.match(name)
    if m:
        layer_idx, rest = m.group(1), m.group(2)
        rest = rest.replace("attention.", "attention.").replace("mlp.", "mlp.")
        rest = re.sub(r"^(attention|mlp)\.", r"\1.", rest)
        # SigLIP: encoder.layers.N.<module>; Streamformer: timesformer.encoder.layer.N.<module>
        return "timesformer.encoder.layer.{}.{}".format(layer_idx, rest)

    # Final layer norm / residual block.
    if name == "post_layernorm.weight":
        return "timesformer.encoder.layernorm.weight"

    return None


def copy_param(state_dict, source_model, name):
    """Copy a single parameter into ``state_dict`` if a valid target key exists."""
    source = source_model
    for part in name.split("."):
        source = getattr(source, part)
    param = source

    target_key = _target_key(name)
    if target_key is None:
        return
    if target_key not in state_dict:
        warnings.warn("Target key %r not found in model state dict; skipped.", target_key)
        return
    if state_dict[target_key].shape != param.shape:
        warnings.warn(
            "Shape mismatch for %s: target %s, source %s; skipped.",
            target_key,
            state_dict[target_key].shape,
            param.shape,
        )
        return

    state_dict[target_key].copy_(param)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer SigLIP vision weights into a Streamformer model."
    )
    parser.add_argument(
        "--siglip_path",
        type=str,
        required=True,
        help="Hugging Face hub id or local path of the SigLIP model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory where the initialized Streamformer model is saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to perform the weight transfer (default: cpu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    streamformer = StreamformerForMultiTaskingSigLIP().to(device)

    siglip = AutoModel.from_pretrained(args.siglip_path)
    vision_model = siglip.vision_model.to(device)

    # Build the target state dictionary exactly once. Re-calling
    # ``model.state_dict()`` inside the loop below would rebuild the entire
    # dictionary on every iteration, which is the source of the performance
    # issue reported in #14.
    state_dict = streamformer.state_dict()

    for name in list(dict(vision_model.named_parameters()).keys()):
        copy_param(state_dict, vision_model, name)

    os.makedirs(args.save_path, exist_ok=True)
    streamformer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()

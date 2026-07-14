"""Convert a training checkpoint into a Hugging Face-loadable model directory.

The training loop in ``main.py`` saves either:
  * ``teacher_epochN.pt`` — a raw ``state_dict`` of the ``DINOv3ForSelfSupervisedPretraining``
    wrapper (backbone + projection head), or
  * ``last.pt`` — a resume bundle whose ``teacher`` entry contains that same state_dict.

This script strips off the projection head, restores the underlying Hugging Face
backbone weights, and writes a directory that can be loaded with
``AutoModel.from_pretrained(<dir>)`` exactly like the original ``base_model``.
Optionally pushes the result to the Hub.

Usage:
    python export_to_hf.py \
        --checkpoint checkpoints/teacher_epoch99.pt \
        --output exported/dinov3-vitl16-finetuned

    # And optionally push:
    python export_to_hf.py \
        --checkpoint checkpoints/last.pt \
        --output exported/dinov3-vitl16-finetuned \
        --push-to-hub your-username/dinov3-vitl16-finetuned
"""

import argparse
import json
import os

import torch
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel


BACKBONE_PREFIX = "backbone."


def _hf_login() -> None:
    """Login to the Hugging Face Hub if a local tokens file is available."""
    token_path = "./.tokens.json"
    if not os.path.isfile(token_path):
        return
    with open(token_path, "r", encoding="utf-8") as f:
        tokens = json.load(f)
    if "dinov3" in tokens:
        login(token=tokens["dinov3"])


def load_teacher_state_dict(checkpoint_path: str) -> dict:
    """Load either a raw teacher state_dict or the ``teacher`` entry of a resume bundle."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "teacher" in ckpt:
        state_dict = ckpt["teacher"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unexpected checkpoint format at {checkpoint_path}")

    return state_dict


def extract_backbone_state_dict(state_dict: dict) -> dict:
    """Keep only ``backbone.*`` keys and strip the prefix."""
    backbone_sd = {
        k[len(BACKBONE_PREFIX):]: v
        for k, v in state_dict.items()
        if k.startswith(BACKBONE_PREFIX)
    }
    if not backbone_sd:
        raise ValueError(
            "No 'backbone.*' keys found in checkpoint. "
            "Was this trained with DINOv3ForSelfSupervisedPretraining?"
        )
    return backbone_sd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to teacher_epochN.pt or last.pt saved by main.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the Hugging Face model to",
    )
    parser.add_argument(
        "--base-model",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Original model ID used to initialize training (for config/architecture)",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional Hub repo id (e.g. 'user/my-dinov3'). If provided, pushes the model.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If pushing to the Hub, create the repo as private.",
    )
    args = parser.parse_args()

    _hf_login()

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = load_teacher_state_dict(args.checkpoint)
    backbone_sd = extract_backbone_state_dict(state_dict)
    print(f"Extracted {len(backbone_sd)} backbone tensors")

    print(f"Instantiating base model: {args.base_model}")
    model = AutoModel.from_pretrained(args.base_model)

    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("All backbone weights loaded cleanly.")

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving model to: {args.output}")
    model.save_pretrained(args.output)

    # Save the matching image processor so downstream users get identical preprocessing.
    try:
        processor = AutoImageProcessor.from_pretrained(args.base_model)
        processor.save_pretrained(args.output)
        print("Saved matching image processor.")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Could not save image processor: {e}")

    if args.push_to_hub:
        print(f"Pushing to Hub repo: {args.push_to_hub} (private={args.private})")
        model.push_to_hub(args.push_to_hub, private=args.private)
        try:
            processor.push_to_hub(args.push_to_hub, private=args.private)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Could not push processor: {e}")
        print("Push complete.")

    print("Done. Reload with:")
    print(f"    AutoModel.from_pretrained('{args.push_to_hub or args.output}')")


if __name__ == "__main__":
    main()

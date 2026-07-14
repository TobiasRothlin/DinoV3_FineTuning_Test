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
    Edit the configuration constants below, then run:
        python export_to_hf.py
"""

import json
import os

import torch
from huggingface_hub import login
from transformers import AutoImageProcessor, AutoModel


BACKBONE_PREFIX = "backbone."

# --- Configuration ---------------------------------------------------------
# Path to teacher_epochN.pt or last.pt saved by main.py
CHECKPOINT = "checkpoints/teacher_epoch99.pt"
# Directory to save the Hugging Face model to
OUTPUT = "exported/dinov3-vitl16-finetuned"
# Original model ID used to initialize training (for config/architecture)
BASE_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
# Optional Hub repo id (e.g. 'user/my-dinov3'). If set, pushes the model.
PUSH_TO_HUB = None
# If pushing to the Hub, create the repo as private.
PRIVATE = False
# ---------------------------------------------------------------------------


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
    _hf_login()

    print(f"Loading checkpoint: {CHECKPOINT}")
    state_dict = load_teacher_state_dict(CHECKPOINT)
    backbone_sd = extract_backbone_state_dict(state_dict)
    print(f"Extracted {len(backbone_sd)} backbone tensors")

    print(f"Instantiating base model: {BASE_MODEL}")
    model = AutoModel.from_pretrained(BASE_MODEL)

    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("All backbone weights loaded cleanly.")

    os.makedirs(OUTPUT, exist_ok=True)
    print(f"Saving model to: {OUTPUT}")
    model.save_pretrained(OUTPUT)

    # Save the matching image processor so downstream users get identical preprocessing.
    try:
        processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
        processor.save_pretrained(OUTPUT)
        print("Saved matching image processor.")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Could not save image processor: {e}")

    if PUSH_TO_HUB:
        print(f"Pushing to Hub repo: {PUSH_TO_HUB} (private={PRIVATE})")
        model.push_to_hub(PUSH_TO_HUB, private=PRIVATE)
        try:
            processor.push_to_hub(PUSH_TO_HUB, private=PRIVATE)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Could not push processor: {e}")
        print("Push complete.")

    print("Done. Reload with:")
    print(f"    AutoModel.from_pretrained('{PUSH_TO_HUB or OUTPUT}')")


if __name__ == "__main__":
    main()

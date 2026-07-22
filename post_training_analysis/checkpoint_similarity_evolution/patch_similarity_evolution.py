"""Visualize how patch similarity evolves across training checkpoints.

Given a checkpoints directory (populated by ``main.py`` / ``trainer.py``), an
image, and a single query patch coordinate on the feature grid, this script
computes the cosine-similarity map from the query patch to every other patch
under every checkpoint and renders one figure whose subplots are those maps,
titled by the checkpoint filename.

The checkpoint parsing logic mirrors ``post_training_analysis/export_to_hf.py``:
each ``teacher_epochN.pt`` is a raw state_dict of
``DINOv3ForSelfSupervisedPretraining`` (backbone + projection heads), and the
``backbone.*`` prefix is stripped before loading into a Hugging Face
``AutoModel`` backbone. ``last.pt`` bundles are transparently unwrapped.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # Headless-safe backend for containers.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel


# Normalization statistics — must match the training pipeline in ``Config.py``.
FINETUNED_MEAN = [0.4114, 0.4183, 0.4359]
FINETUNED_STD = [0.3005, 0.2981, 0.2966]
# Standard ImageNet stats used by the original DINOv3 base checkpoint.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONE_PREFIX = "backbone."
PATCH_SIZE = 16

_EPOCH_RE = re.compile(r"teacher_epoch(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors post_training_analysis/export_to_hf.py)
# ---------------------------------------------------------------------------

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


def discover_checkpoints(
    checkpoints_dir: Path,
    stride: int,
    include_last: bool,
) -> list[Path]:
    """Return a sorted list of checkpoint paths to visualize."""
    epoch_ckpts: list[tuple[int, Path]] = []
    for p in checkpoints_dir.iterdir():
        m = _EPOCH_RE.match(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
    epoch_ckpts.sort(key=lambda x: x[0])

    selected = [p for _, p in epoch_ckpts[:: max(stride, 1)]]

    if include_last:
        last = checkpoints_dir / "last.pt"
        if last.is_file() and last not in selected:
            selected.append(last)

    if not selected:
        raise FileNotFoundError(
            f"No 'teacher_epoch*.pt' files found in {checkpoints_dir}"
        )
    return selected


# ---------------------------------------------------------------------------
# Feature extraction & similarity (mirrors image_patch_similarity.py)
# ---------------------------------------------------------------------------

def build_transform(target_size: tuple[int, int], mean: list[float], std: list[float]):
    return T.Compose([
        T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


@torch.inference_mode()
def extract_patch_embeddings(
    model: torch.nn.Module,
    image: Image.Image,
    target_size: tuple[int, int],
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Return L2-normalized patch embeddings shaped (h_feat, w_feat, hidden_dim)."""
    transform = build_transform(target_size, mean, std)
    pixel_values = transform(image).unsqueeze(0).to(device)

    _, _, H, W = pixel_values.shape
    if H % PATCH_SIZE != 0 or W % PATCH_SIZE != 0:
        raise ValueError(
            f"Target size ({H}x{W}) must be a multiple of patch size {PATCH_SIZE}."
        )
    h_feat, w_feat = H // PATCH_SIZE, W // PATCH_SIZE
    num_patches = h_feat * w_feat

    outputs = model(pixel_values=pixel_values)
    last_hidden_state = outputs.last_hidden_state
    patch_embs = last_hidden_state[0, -num_patches:]
    patch_embs = F.normalize(patch_embs, p=2, dim=-1)
    return patch_embs.view(h_feat, w_feat, -1), h_feat, w_feat


def similarity_map(
    patch_embs: torch.Tensor,
    feat_x: int,
    feat_y: int,
    orig_H: int,
    orig_W: int,
) -> np.ndarray:
    """Cosine-similarity map upsampled to the original image resolution."""
    query_emb = patch_embs[feat_y, feat_x].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    sim = (patch_embs * query_emb).sum(dim=-1)  # (h_feat, w_feat)
    sim_tensor = sim.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    sim_up = F.interpolate(
        sim_tensor,
        size=(orig_H, orig_W),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return sim_up.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def render_grid(
    raw_image: Image.Image,
    panels: list[tuple[str, np.ndarray]],
    feat_x: int,
    feat_y: int,
    h_feat: int,
    w_feat: int,
    output_path: Path,
    overlay_alpha: float,
    vmin: float,
    vmax: float,
    cmap: str,
    suptitle: str,
) -> None:
    n = len(panels)
    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    orig_W, orig_H = raw_image.size
    aspect = orig_W / max(orig_H, 1)
    subplot_w = 4.0
    subplot_h = max(2.0, subplot_w / max(aspect, 1e-6))
    fig_w = ncols * subplot_w
    fig_h = nrows * subplot_h + 1.5  # leave space for suptitle + colorbar

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    patch_pixel_w = orig_W / w_feat
    patch_pixel_h = orig_H / h_feat

    last_im = None
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= n:
            ax.axis("off")
            continue
        title, sim = panels[idx]
        ax.imshow(raw_image)
        last_im = ax.imshow(
            sim,
            cmap=cmap,
            alpha=overlay_alpha,
            vmin=vmin,
            vmax=vmax,
        )
        rect = plt.Rectangle(
            (feat_x * patch_pixel_w, feat_y * patch_pixel_h),
            patch_pixel_w,
            patch_pixel_h,
            edgecolor="red",
            facecolor="none",
            lw=1.5,
        )
        ax.add_patch(rect)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))

    if last_im is not None:
        cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a grid of patch-similarity heatmaps for one query patch, "
            "one subplot per training checkpoint."
        )
    )
    parser.add_argument("--checkpoints", required=True, type=Path,
                        help="Folder containing teacher_epoch*.pt files.")
    parser.add_argument("--image", required=True, type=Path,
                        help="Path to the input image.")
    parser.add_argument("--patch-x", required=True, type=int,
                        help="Query patch column index on the feature grid (0..w_feat-1).")
    parser.add_argument("--patch-y", required=True, type=int,
                        help="Query patch row index on the feature grid (0..h_feat-1).")
    parser.add_argument("--base-model", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="Hugging Face model id used to initialize training.")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("H", "W"),
                        default=[2304, 832],
                        help="Model input size (must be multiples of 16).")
    parser.add_argument("--output", type=Path, default=Path("similarity_evolution.png"),
                        help="Output PNG path.")
    parser.add_argument("--include-base", action="store_true",
                        help="Also render the un-finetuned base pretrained model as the first panel.")
    parser.add_argument("--include-last", action="store_true",
                        help="Also include last.pt (skipped by default as it duplicates the final epoch).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Only visualize every Nth checkpoint (sorted by epoch).")
    parser.add_argument("--overlay-alpha", type=float, default=0.6,
                        help="Alpha blend for the heatmap over the original image.")
    parser.add_argument("--sim-vmin", type=float, default=-1.0,
                        help="Colorbar lower bound.")
    parser.add_argument("--sim-vmax", type=float, default=1.0,
                        help="Colorbar upper bound.")
    parser.add_argument("--cmap", default="jet", help="matplotlib colormap name.")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.checkpoints.is_dir():
        print(f"[error] --checkpoints is not a directory: {args.checkpoints}", file=sys.stderr)
        return 2
    if not args.image.is_file():
        print(f"[error] --image is not a file: {args.image}", file=sys.stderr)
        return 2

    target_size = (int(args.target_size[0]), int(args.target_size[1]))
    if target_size[0] % PATCH_SIZE != 0 or target_size[1] % PATCH_SIZE != 0:
        print(
            f"[error] --target-size {target_size} must be multiples of {PATCH_SIZE}.",
            file=sys.stderr,
        )
        return 2

    h_feat_expected = target_size[0] // PATCH_SIZE
    w_feat_expected = target_size[1] // PATCH_SIZE
    if not (0 <= args.patch_x < w_feat_expected):
        print(
            f"[error] --patch-x {args.patch_x} out of range [0, {w_feat_expected - 1}] "
            f"for target size {target_size}.",
            file=sys.stderr,
        )
        return 2
    if not (0 <= args.patch_y < h_feat_expected):
        print(
            f"[error] --patch-y {args.patch_y} out of range [0, {h_feat_expected - 1}] "
            f"for target size {target_size}.",
            file=sys.stderr,
        )
        return 2

    ckpt_paths = discover_checkpoints(args.checkpoints, args.stride, args.include_last)
    print(f"Found {len(ckpt_paths)} checkpoint(s) to visualize.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_image = Image.open(args.image).convert("RGB")
    orig_W, orig_H = raw_image.size

    print(f"Instantiating base backbone: {args.base_model}")
    model = AutoModel.from_pretrained(args.base_model).to(device)
    model.eval()

    panels: list[tuple[str, np.ndarray]] = []

    if args.include_base:
        print("Extracting features for base pretrained model...")
        embs, h_feat, w_feat = extract_patch_embeddings(
            model, raw_image, target_size, IMAGENET_MEAN, IMAGENET_STD, device
        )
        sim = similarity_map(embs, args.patch_x, args.patch_y, orig_H, orig_W)
        panels.append(("base (pretrained)", sim))
        del embs

    pbar = tqdm(ckpt_paths, desc="Checkpoints", unit="ckpt")
    for ckpt_path in pbar:
        pbar.set_postfix_str(ckpt_path.name)
        state_dict = load_teacher_state_dict(str(ckpt_path))
        backbone_sd = extract_backbone_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        if missing:
            tqdm.write(f"  [warn] {ckpt_path.name}: missing keys: {len(missing)} (first: {missing[:3]})")
        if unexpected:
            tqdm.write(f"  [warn] {ckpt_path.name}: unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
        model.eval()

        embs, h_feat, w_feat = extract_patch_embeddings(
            model, raw_image, target_size, FINETUNED_MEAN, FINETUNED_STD, device
        )
        sim = similarity_map(embs, args.patch_x, args.patch_y, orig_H, orig_W)
        panels.append((ckpt_path.stem, sim))
        del embs, state_dict, backbone_sd
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if h_feat != h_feat_expected or w_feat != w_feat_expected:
        print(
            f"[warn] feature grid {h_feat}x{w_feat} differs from expected "
            f"{h_feat_expected}x{w_feat_expected}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suptitle = (
        f"Similarity to patch (x={args.patch_x}, y={args.patch_y}) — "
        f"{os.path.basename(args.image)}"
    )
    print(f"Saving figure with {len(panels)} panel(s) to {args.output}")
    render_grid(
        raw_image=raw_image,
        panels=panels,
        feat_x=args.patch_x,
        feat_y=args.patch_y,
        h_feat=h_feat,
        w_feat=w_feat,
        output_path=args.output,
        overlay_alpha=args.overlay_alpha,
        vmin=args.sim_vmin,
        vmax=args.sim_vmax,
        cmap=args.cmap,
        suptitle=suptitle,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

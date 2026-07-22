# Checkpoint Similarity Evolution

Visualize how a single query patch's cosine-similarity map evolves across
training checkpoints. For every `teacher_epoch*.pt` file in a checkpoints
folder, the script computes the similarity between one query patch and every
other patch of the input image, then renders one figure whose subplots are
those heatmaps — titled by the checkpoint filename.

This is useful for eyeballing:

- when a fine-tune starts producing semantically clustered features,
- whether later epochs continue to sharpen (or degrade) the representation,
- how the fine-tuned features compare against the base pretrained model
  (`--include-base`).

## Prerequisites

The shared base image `dgx-train-base:latest` must already exist on the host.
If you have not built it yet, do so from the repository root:

```bash
docker build -f Dockerfile.base -t dgx-train-base:latest .
```

## Build

From this folder:

```bash
docker build -t dinov3-sim-evolution:latest .
```

## Run

The container's entrypoint is the script, so CLI flags go directly on the
`docker run` command line. Mount:

- the repository `checkpoints/` folder as `/checkpoints` (read-only),
- the folder containing your image(s) as `/data` (read-only),
- an output folder to receive the PNG as `/out`,
- the shared HF cache volume so the base model download is deduplicated with
  the training project.

```bash
docker run --rm --gpus all   -v /home/spark/Projects/DinoV3_FineTuning_Test/checkpoints:/checkpoints:ro   -v /home/spark/Datasets/AGI:/data:ro   -v /home/spark/Projects/DinoV3_FineTuning_Test/post_training_analysis/checkpoint_similarity_evolution/output:/out   -v dgx_hf_cache:/hf_cache   dinov3-sim-evolution:latest     --checkpoints /checkpoints     --image /data/Alpha2/frame_2420_1771945610547.jpg     --patch-x 27 --patch-y 27     --target-size 2304 832     --stride 1     --output /out/similarity_evolution.png --sim-vmin 0.0 --sim-vmax 1.0
```

Show all flags:

```bash
docker run --rm dinov3-sim-evolution:latest --help
```

## Choosing `--patch-x` / `--patch-y`

The patch coordinates are **indices on the feature grid**, not pixel
coordinates. The grid dimensions are derived from `--target-size` and the
backbone patch size (16):

- `w_feat = target_W // 16`
- `h_feat = target_H // 16`

With the default `--target-size 2304 832` (H W), the grid is `52 × 144`
(`h_feat × w_feat`), so valid ranges are `--patch-x` in `[0, 143]` and
`--patch-y` in `[0, 51]`.

To convert a pixel `(px, py)` on your original image (of size
`orig_W × orig_H`) to grid indices:

```
patch_x = int((px / orig_W) * w_feat)
patch_y = int((py / orig_H) * h_feat)
```

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--checkpoints` | *(required)* | Folder containing `teacher_epoch*.pt`. |
| `--image` | *(required)* | Input image path. |
| `--patch-x`, `--patch-y` | *(required)* | Query patch on the feature grid. |
| `--base-model` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | HF model id used to init the backbone architecture. |
| `--target-size H W` | `2304 832` | Model input size; must be multiples of 16. |
| `--output` | `similarity_evolution.png` | Output PNG path. |
| `--include-base` | *(off)* | Prepend a panel for the un-finetuned base model (uses ImageNet normalization). |
| `--include-last` | *(off)* | Also include `last.pt` (skipped by default because it duplicates the last epoch). |
| `--stride` | `1` | Only visualize every Nth checkpoint (sorted by epoch). |
| `--overlay-alpha` | `0.6` | Heatmap alpha blend over the original image. |
| `--sim-vmin` / `--sim-vmax` | `-1.0` / `1.0` | Colorbar range. Cosine similarity ∈ `[-1, 1]`. |
| `--cmap` | `jet` | matplotlib colormap. |

## Notes

- The script is headless (uses the `Agg` matplotlib backend) and writes a PNG;
  no X server is required on the DGX Spark.
- Checkpoint parsing matches the format written by `trainer.py`
  (`save_checkpoint`) and the extractor in
  `../export_to_hf.py`: `teacher_epochN.pt` is a raw `state_dict` of the
  full `DINOv3ForSelfSupervisedPretraining` wrapper, and `last.pt` is a resume
  bundle whose `teacher` entry contains that same state_dict.
- Only `backbone.*` weights are loaded into the HF `AutoModel`; the projection
  heads (`dino_head`, `ibot_head`) are ignored (as in `export_to_hf.py`).

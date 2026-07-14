import random
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


from dataset import DinoDataset
from dino_v3_transforms import CustomDomainMultiCropTransform
from model import DINOv3ForSelfSupervisedPretraining
from loss import DINOLoss, iBOTPatchLoss, GramLoss, update_teacher_ema
from masking import iBOTMaskGenerator
from cuda_utility import check_cuda
from huggingface_hub import login


# ImageNet stats used in CustomDomainMultiCropTransform
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Training configuration constants ---
DEBUG = False                       # Set True to visualize crops before training (blocks on plt.show)
ACCUMULATION_STEPS = 16              # Gradient accumulation steps (lower batch size + raise this to save VRAM)
BATCH_SIZE = 16
CHECKPOINT_DIR = "checkpoints"      # Where teacher checkpoints and the resume bundle are saved
RESUME_PATH = ""                   # Path to a last.pt bundle to resume from; empty = start fresh

# --- DINOv3 dense (iBOT) + Gram anchoring config (tune here for DGX sweeps) ---
IBOT_OUT_DIM = 65536               # iBOT head output dim (drop to 16384/8192 on OOM)
DINO_OUT_DIM = 65536               # DINO (global CLS) head output dim
HIDDEN_DIM = 2048                  # Projection-head hidden width
W_GRAM = 2.0                       # Gram anchoring loss weight
MASK_RATIO_MIN = 0.1               # Min proportion of patch tokens masked per global crop
MASK_RATIO_MAX = 0.5               # Max proportion of patch tokens masked per global crop
MASK_PROB = 0.5                    # Per-sample probability of applying masking at all
GRAM_REFRESH_STEPS = 10000         # Refresh the Gram teacher from the teacher every N optimizer steps

# --- LP-FT (Linear-Probe then Fine-Tune) learning-rate config ---
HEAD_LR = 1e-4                     # Peak LR for the randomly-initialized projection heads
BACKBONE_LR = 1e-6                 # Peak LR for the pre-trained backbone (kept tiny)
HEAD_WARMUP_EPOCHS = 3             # Epochs to train heads only (backbone LR held at 0)
BACKBONE_RAMP_EPOCHS = 1           # Epochs to linearly ramp the backbone LR to BACKBONE_LR

def _hf_login() -> None:
    """Login to the Hugging Face Hub using a local tokens JSON file."""
    with open("./.tokens.json", "r", encoding="utf-8") as f:
        tokens = json.load(f)
    login(token=tokens["dinov3"])


def cosine_schedule(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0.0):
    """Build a per-step schedule: linear warmup followed by cosine decay/growth."""
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    iters = np.arange(total_steps - warmup_steps)
    cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup, cosine))
    assert len(schedule) == total_steps
    return schedule


def _backbone_lr_schedule(total_steps, steps_per_epoch, warmup_epochs, ramp_epochs, peak_lr):
    """LP-FT backbone schedule (step-indexed, epoch-gated phase transitions).

    Phase 1 (0 .. warmup_epochs):        LR = 0            (train heads only)
    Phase 2 (warmup .. warmup+ramp):     LR ramps 0 -> peak (linear, avoids gradient shock)
    Phase 3 (rest):                      LR = peak         (held flat)
    """
    schedule = np.zeros(total_steps, dtype=np.float64)
    warmup_end = warmup_epochs * steps_per_epoch
    ramp_end = warmup_end + ramp_epochs * steps_per_epoch
    ramp_end = min(ramp_end, total_steps)

    if ramp_end > warmup_end:
        schedule[warmup_end:ramp_end] = np.linspace(0.0, peak_lr, ramp_end - warmup_end)
    schedule[ramp_end:] = peak_lr
    return schedule


def denormalize(tensor):
    """Reverse the ImageNet normalization and return an HWC image in [0, 1]."""
    img = tensor * _STD + _MEAN
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_crops(crops, num_local_crops):
    """Plot the two global crops and the N local crops in a single figure."""
    num_crops = len(crops)
    ncols = 4
    nrows = (num_crops + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < num_crops:
            ax.imshow(denormalize(crops[i]))
            if i < 2:
                ax.set_title(f"Global {i + 1}")
            else:
                ax.set_title(f"Local {i - 1}")
        ax.axis("off")

    fig.suptitle(f"2 Global Crops + {num_local_crops} Local Crops")
    plt.tight_layout()
    plt.show()


def train():
    _hf_login()

    base_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    device = check_cuda()

    # Example usage of DinoDataset
    dino_transform = CustomDomainMultiCropTransform(global_size=(448, 256), local_size=(224, 128), num_local_crops=6)

    data_root = os.environ.get("DATA_ROOT", "./data")
    dataset = DinoDataset(folder_path=os.path.join(data_root, "Alpha1"), transform=dino_transform)
    print(f"Number of images in dataset: {len(dataset)}")

    crops = dataset[random.randint(0,len(dataset)-1)]  # Get the first image and its crops

    if DEBUG:
        visualize_crops(crops, num_local_crops=6)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    student = DINOv3ForSelfSupervisedPretraining(
        base_model, dino_out_dim=DINO_OUT_DIM, ibot_out_dim=IBOT_OUT_DIM, hidden_dim=HIDDEN_DIM
    )
    teacher = copy.deepcopy(student)

    for param in teacher.parameters():
        param.requires_grad = False

    student = student.to(device)
    teacher = teacher.to(device)

    # The Gram teacher is a slow geometric anchor: a frozen clone of the teacher that
    # is refreshed only every GRAM_REFRESH_STEPS. It lives on the CPU between refreshes
    # to save VRAM (a ViT-L in fp16 is ~600MB) and is moved to the GPU only when used.
    gram_teacher = copy.deepcopy(teacher)
    for param in gram_teacher.parameters():
        param.requires_grad = False
    gram_teacher.eval()
    gram_teacher = gram_teacher.to("cpu")

    # --- LP-FT parameter groups: heads (high LR) vs. backbone (tiny LR) ---
    head_params, backbone_params = [], []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:  # dino_head / ibot_head / mask_token
            head_params.append(param)

    # DINOv3 uses AdamW; here each group carries its own peak LR (set per-step below).
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": HEAD_LR, "name": "head"},
            {"params": backbone_params, "lr": 0.0, "name": "backbone"},
        ],
        weight_decay=0.04,
    )

    # Loss modules (each with its own center buffer) + on-device mask generator.
    dino_loss_fn = DINOLoss(out_dim=DINO_OUT_DIM).to(device)
    ibot_loss_fn = iBOTPatchLoss(out_dim=IBOT_OUT_DIM).to(device)
    gram_loss_fn = GramLoss().to(device)
    mask_generator = iBOTMaskGenerator(ratio_min=MASK_RATIO_MIN, ratio_max=MASK_RATIO_MAX, mask_prob=MASK_PROB)

    # Mixed-precision scaler (AMP) to reduce VRAM and speed up training
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 100

    # --- Build per-iteration warmup/cosine schedules ---
    steps_per_epoch = len(dataloader) // ACCUMULATION_STEPS
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * 10  # 10-epoch LR warmup for the heads

    # Heads follow a step-indexed warmup + cosine curve from step 0.
    head_lr_schedule = cosine_schedule(HEAD_LR, 1e-6, total_steps, warmup_steps=warmup_steps)
    # Backbone LR is gated by epoch (LP-FT): 0 during head warmup, then a 1-epoch
    # linear ramp to BACKBONE_LR, then held flat. Built per-step for a smooth ramp.
    backbone_lr_schedule = _backbone_lr_schedule(
        total_steps, steps_per_epoch, HEAD_WARMUP_EPOCHS, BACKBONE_RAMP_EPOCHS, BACKBONE_LR
    )
    teacher_temp_schedule = cosine_schedule(0.07, 0.07, total_steps, warmup_steps=warmup_steps, start_warmup_value=0.04)
    momentum_schedule = cosine_schedule(0.996, 1.0, total_steps)

    # --- Optional resume from a saved bundle ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start_epoch = 0
    global_step = 0
    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        print(f"Resuming from checkpoint: {RESUME_PATH}")
        ckpt = torch.load(RESUME_PATH, map_location=device)
        student.load_state_dict(ckpt['student'])
        teacher.load_state_dict(ckpt['teacher'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        dino_loss_fn.center.copy_(ckpt['center'].to(device))
        if 'ibot_center' in ckpt:
            ibot_loss_fn.center.copy_(ckpt['ibot_center'].to(device))
        if 'gram_teacher' in ckpt:
            gram_teacher.load_state_dict(ckpt['gram_teacher'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    print(f"Starting training on {device}...")

    # 2. The Core Training Loop
    for epoch in range(start_epoch, num_epochs):
        student.train()
        epoch_loss = 0.0

        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch}/{num_epochs - 1} | Epoch Loss: {epoch_loss:.4f} | Head LR: {head_lr_schedule[min(global_step, total_steps - 1)]:.2e}",
            dynamic_ncols=True,
        )
        for batch_idx, crops in pbar:
            # Index the schedules by the current optimizer step
            sched_idx = min(global_step, total_steps - 1)
            head_lr = head_lr_schedule[sched_idx]
            backbone_lr = backbone_lr_schedule[sched_idx]
            for group in optimizer.param_groups:
                group['lr'] = head_lr if group.get('name') == 'head' else backbone_lr
            teacher_temp = teacher_temp_schedule[sched_idx]
            momentum = momentum_schedule[sched_idx]

            # Move all 8 crops (2 global, 6 local) to the GPU
            crops = [crop.to(device, non_blocking=True) for crop in crops]
            global_crops = crops[:2]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # --- MASK GENERATION (global crops only) ---
                # Derive the patch-grid size from the backbone's patch embedding, then
                # build a boolean mask per global crop. bool_masked_pos indexes patch
                # tokens only (CLS + register tokens are excluded).
                with torch.no_grad():
                    patch_embeds = student.backbone.embeddings.patch_embeddings(global_crops[0])
                    num_patches = patch_embeds.flatten(2).shape[-1]
                masks = [
                    mask_generator(g.shape[0], num_patches, device) for g in global_crops
                ]

                # --- TEACHER FORWARD PASS ---
                # The Teacher ONLY sees the global crops (unmasked) and computes NO gradients.
                with torch.no_grad():
                    teacher_outputs = [teacher(crop) for crop in global_crops]  # (cls, patch, raw)

                # --- STUDENT FORWARD PASS ---
                # Student sees EVERYTHING. Global crops are masked for the iBOT objective;
                # local crops are always unmasked (too small to reconstruct).
                student_global = [student(g, mask=m) for g, m in zip(global_crops, masks)]
                student_local = [student(crop) for crop in crops[2:]]
                student_outputs = student_global + student_local

                # --- GLOBAL DINO LOSS ---
                # Compare the Student's CLS predictions (all crops) against the Teacher's
                # CLS predictions (global crops), skipping same-view comparisons.
                dino_loss = 0.0
                n_dino_terms = 0
                for t_idx, t_out in enumerate(teacher_outputs):
                    t_cls = t_out[0]
                    for s_idx, s_out in enumerate(student_outputs):
                        if t_idx == s_idx:
                            continue  # Do not compare a global crop to itself
                        dino_loss += dino_loss_fn(s_out[0], t_cls, teacher_temp=teacher_temp)
                        n_dino_terms += 1
                dino_loss = dino_loss / n_dino_terms

                # --- LOCAL iBOT LOSS (same-view masked patch reconstruction) ---
                ibot_loss = 0.0
                for (s_cls, s_patch, _s_raw), t_out, m in zip(student_global, teacher_outputs, masks):
                    ibot_loss += ibot_loss_fn(s_patch, t_out[1], m, teacher_temp=teacher_temp)
                ibot_loss = ibot_loss / len(student_global)

            # --- GRAM ANCHORING LOSS (fp32, outside autocast for numerical stability) ---
            with torch.no_grad():
                gram_teacher_gpu = gram_teacher.to(device)
                gram_outputs = [gram_teacher_gpu(crop) for crop in global_crops]  # (cls, patch, raw)
                gram_teacher.to("cpu")

            gram_loss = 0.0
            for (s_cls, s_patch, s_raw), g_out in zip(student_global, gram_outputs):
                gram_loss = gram_loss + gram_loss_fn(s_raw, g_out[2])
            gram_loss = gram_loss / len(student_global)

            # --- AGGREGATE THE TRIPARTITE LOSS ---
            loss = dino_loss + ibot_loss + W_GRAM * gram_loss

            # --- BACKPROPAGATION (with gradient accumulation) ---
            scaler.scale(loss / ACCUMULATION_STEPS).backward()

            is_step = ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader))
            if is_step:
                # Unscale before clipping, then step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=3.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # --- EMA & CENTER UPDATES ---
                # Update the Teacher's weights and the loss centers using moving averages
                update_teacher_ema(student, teacher, momentum=momentum)

                # Update the global (CLS) center and the dense (patch) center.
                all_teacher_cls = torch.cat([t[0] for t in teacher_outputs], dim=0)
                dino_loss_fn.update_center(all_teacher_cls)
                all_teacher_patch = torch.cat([t[1] for t in teacher_outputs], dim=0)
                ibot_loss_fn.update_center(all_teacher_patch)

                global_step += 1

                # --- GRAM TEACHER REFRESH ---
                # Periodically hard-copy the (slow) teacher into the Gram teacher so it
                # tracks the improving geometry while staying a stable anchor.
                if global_step % GRAM_REFRESH_STEPS == 0:
                    gram_teacher.load_state_dict(teacher.state_dict())
                    gram_teacher.eval()
                    print(f"[step {global_step}] Refreshed Gram teacher from teacher.")

            epoch_loss += loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dino': f"{float(dino_loss):.3f}",
                'ibot': f"{float(ibot_loss):.3f}",
                'gram': f"{float(gram_loss):.3f}",
                'h_lr': f"{head_lr:.2e}",
                'b_lr': f"{backbone_lr:.2e}",
                'step': global_step,
            })

        pbar.close()
        print(f"--- Epoch {epoch} completed. Average Loss: {epoch_loss / len(dataloader):.4f} ---")

        # --- CHECKPOINTING ---
        # The teacher (EMA of the student) is the model kept for downstream use.
        teacher_path = os.path.join(CHECKPOINT_DIR, f"teacher_epoch{epoch}.pt")
        torch.save(teacher.state_dict(), teacher_path)

        # Full resume bundle (overwritten each epoch)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'gram_teacher': gram_teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'center': dino_loss_fn.center.detach().cpu(),
            'ibot_center': ibot_loss_fn.center.detach().cpu(),
        }, os.path.join(CHECKPOINT_DIR, "last.pt"))
        print(f"Saved teacher checkpoint -> {teacher_path}")

if __name__ == '__main__':
    train()
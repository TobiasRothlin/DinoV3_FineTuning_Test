import random
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy


from dataset import DinoDataset
from dino_v3_transforms import CustomDomainMultiCropTransform
from model import DINOv3ForSelfSupervisedPretraining
from loss import DINOLoss, update_teacher_ema
from cuda_utility import check_cuda


# ImageNet stats used in CustomDomainMultiCropTransform
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Training configuration constants ---
DEBUG = False                       # Set True to visualize crops before training (blocks on plt.show)
ACCUMULATION_STEPS = 1              # Gradient accumulation steps (lower batch size + raise this to save VRAM)
BATCH_SIZE = 256
CHECKPOINT_DIR = "checkpoints"      # Where teacher checkpoints and the resume bundle are saved
RESUME_PATH = ""                   # Path to a last.pt bundle to resume from; empty = start fresh


def cosine_schedule(base_value, final_value, total_steps, warmup_steps=0, start_warmup_value=0.0):
    """Build a per-step schedule: linear warmup followed by cosine decay/growth."""
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    iters = np.arange(total_steps - warmup_steps)
    cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup, cosine))
    assert len(schedule) == total_steps
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

    base_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    device = check_cuda()

    # Example usage of DinoDataset
    dino_transform = CustomDomainMultiCropTransform(global_size=(448, 256), local_size=(224, 128), num_local_crops=6)

    dataset = DinoDataset(folder_path=r'E:\Alpha1', transform=dino_transform)
    print(f"Number of images in dataset: {len(dataset)}")

    crops = dataset[random.randint(0,len(dataset)-1)]  # Get the first image and its crops

    if DEBUG:
        visualize_crops(crops, num_local_crops=6)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    student = DINOv3ForSelfSupervisedPretraining(base_model)
    teacher = copy.deepcopy(student)

    for param in teacher.parameters():
        param.requires_grad = False

    student = student.to(device)
    teacher = teacher.to(device)

    # DINOv3 uses AdamW with a constant learning rate for the main training phase[cite: 1]
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)
    dino_loss_fn = DINOLoss().to(device)

    # Mixed-precision scaler (AMP) to reduce VRAM and speed up training
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    num_epochs = 100

    # --- Build per-iteration warmup/cosine schedules ---
    steps_per_epoch = len(dataloader) // ACCUMULATION_STEPS
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = steps_per_epoch * 10  # 10-epoch LR warmup

    lr_schedule = cosine_schedule(1e-4, 1e-6, total_steps, warmup_steps=warmup_steps)
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
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    print(f"Starting training on {device}...")

    # 2. The Core Training Loop
    for epoch in range(start_epoch, num_epochs):
        student.train()
        epoch_loss = 0.0

        optimizer.zero_grad()

        for batch_idx, crops in enumerate(dataloader):
            # Index the schedules by the current optimizer step
            sched_idx = min(global_step, total_steps - 1)
            for group in optimizer.param_groups:
                group['lr'] = lr_schedule[sched_idx]
            teacher_temp = teacher_temp_schedule[sched_idx]
            momentum = momentum_schedule[sched_idx]

            # Move all 8 crops (2 global, 6 local) to the GPU
            crops = [crop.to(device, non_blocking=True) for crop in crops]
            global_crops = crops[:2]

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # --- TEACHER FORWARD PASS ---
                # The Teacher ONLY sees the global crops and computes NO gradients[cite: 1].
                with torch.no_grad():
                    teacher_outputs = [teacher(crop) for crop in global_crops]

                # --- STUDENT FORWARD PASS ---
                # The Student sees EVERYTHING (global and local crops)[cite: 1].
                student_outputs = [student(crop) for crop in crops]

                # --- LOSS CALCULATION ---
                loss = 0
                n_loss_terms = 0

                # We compare the Student's predictions for ALL crops against the
                # Teacher's predictions for the GLOBAL crops.
                for t_idx, t_out in enumerate(teacher_outputs):
                    for s_idx, s_out in enumerate(student_outputs):
                        # Do not compare a global crop to itself
                        if t_idx == s_idx:
                            continue

                        loss += dino_loss_fn(s_out, t_out, teacher_temp=teacher_temp)
                        n_loss_terms += 1

                loss = loss / n_loss_terms

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
                # Update the Teacher's weights and the loss center using moving averages
                update_teacher_ema(student, teacher, momentum=momentum)

                # Concatenate teacher outputs to update the center
                all_teacher_outputs = torch.cat(teacher_outputs, dim=0)
                dino_loss_fn.update_center(all_teacher_outputs)

                global_step += 1

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} LR: {lr_schedule[sched_idx]:.2e}")

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
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'center': dino_loss_fn.center.detach().cpu(),
        }, os.path.join(CHECKPOINT_DIR, "last.pt"))
        print(f"Saved teacher checkpoint -> {teacher_path}")

if __name__ == '__main__':
    train()
import random

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

    visualize_crops(crops, num_local_crops=6)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    student = DINOv3ForSelfSupervisedPretraining(base_model)
    teacher = copy.deepcopy(student)

    for param in teacher.parameters():
        param.requires_grad = False

    student = student.to(device)
    teacher = teacher.to(device)

    # DINOv3 uses AdamW with a constant learning rate for the main training phase[cite: 1]
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)
    dino_loss_fn = DINOLoss().to(device)

    num_epochs = 100
    print(f"Starting training on {device}...")

    # 2. The Core Training Loop
    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0.0

        for batch_idx, crops in enumerate(dataloader):
            # Move all 8 crops (2 global, 6 local) to the GPU
            crops = [crop.to(device) for crop in crops]
            global_crops = crops[:2]

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

                    loss += dino_loss_fn(s_out, t_out)
                    n_loss_terms += 1

            loss = loss / n_loss_terms

            # --- BACKPROPAGATION ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- EMA & CENTER UPDATES ---
            # Update the Teacher's weights and the loss center using moving averages
            update_teacher_ema(student, teacher)

            # Concatenate teacher outputs to update the center
            all_teacher_outputs = torch.cat(teacher_outputs, dim=0)
            dino_loss_fn.update_center(all_teacher_outputs)

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        print(f"--- Epoch {epoch} completed. Average Loss: {epoch_loss / len(dataloader):.4f} ---")

if __name__ == '__main__':
    train()
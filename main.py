import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os

# Import your custom modules
from Config import Config
from cuda_utility import check_cuda
from dataset import DinoDataset
from model import DinoV3Pretrainer
from trainer import DinoV3Trainer


def main():
    # 1. Hardware Setup
    # The check_cuda function handles the detection and returns the device
    device = check_cuda()

    # If a list of devices is returned (multiple GPUs), default to cuda:0 for this single-GPU script
    # For distributed training (DDP), this logic would be expanded.
    if isinstance(device, list):
        device = device[0]
        print(f"Selecting primary device: {device} for single-node training.")

    # 2. Dataset and DataLoader Initialization
    # Update this path to where your reactor images are stored on the DGX
    dataset_path = r"/data"
    print(f"Loading dataset from: {dataset_path}")

    dataset = DinoDataset(folder_path=dataset_path, high_res=False)

    # The standard PyTorch collate_fn perfectly handles the multi-crop list output.
    # It will automatically stack the crops into a list of batched tensors:
    # batch[0] -> Global 1, batch[1] -> Global 2, batch[2:] -> Locals
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=8,  # Adjust based on DGX CPU core availability
        pin_memory=True,  # Speeds up host-to-device transfers
        drop_last=True  # Ensures consistent batch sizes for the Koleo loss
    )

    # 3. Model Initialization
    print("Initializing Student and Teacher networks...")
    student = DinoV3Pretrainer(model_name=Config.base_model, out_dim=Config.output_dim)
    teacher = DinoV3Pretrainer(model_name=Config.base_model, out_dim=Config.output_dim)

    # Optional: Initialize a Gram Teacher for the late-stage refinement phase.
    # Set to None for the standard primary training phase.
    gram_teacher = None

    # 4. Optimizer Setup
    # AdamW is the standard optimizer for Vision Transformers to handle weight decay correctly
    optimizer = optim.AdamW(
        student.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )

    # 5. Trainer Initialization
    print("Setting up the DINOv3 Trainer...")
    trainer = DinoV3Trainer(
        student=student,
        teacher=teacher,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        gram_teacher=gram_teacher
    )

    # 6. Launch Training
    trainer.train()


if __name__ == "__main__":
    # Ensure the checkpoints directory exists before starting
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    main()
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random

from Config import Config
from loss import DINOLoss, iBOTLoss, KoleoLoss
from teacher_updater import TeacherUpdater


class DinoV3Trainer:
    def __init__(
            self,
            student,
            teacher,
            dataloader,
            optimizer,
            device,
            gram_teacher=None
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.gram_teacher = gram_teacher.to(device) if gram_teacher else None

        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device

        # Initialize Mixed Precision Scaler for DGX memory efficiency
        self.scaler = GradScaler()

        # Initialize the EMA Updater
        self.updater = TeacherUpdater(self.student, self.teacher, self.gram_teacher)

        # Initialize Loss Functions
        self.dino_loss_fn = DINOLoss().to(device)
        self.ibot_loss_fn = iBOTLoss().to(device)
        self.koleo_loss_fn = KoleoLoss().to(device)

        # Ensure checkpoint directory exists
        os.makedirs(Config.checkpoint_dir, exist_ok=True)
        self.global_step = 0

    def generate_ibot_mask(self, batch_size, seq_len):
        """
        Implements block-wise masking for iBOT.
        seq_len corresponds to the number of patches (e.g., 28 * 16 for a global crop).
        """
        # Assuming your grid is square for simplicity, but adjust for (28, 16)
        grid_h, grid_w = 28, 16

        mask = torch.zeros(batch_size, grid_h, grid_w, device=self.device, dtype=torch.bool)

        for i in range(batch_size):
            # Randomly choose block dimensions
            block_h = random.randint(grid_h // 4, grid_h // 2)
            block_w = random.randint(grid_w // 4, grid_w // 2)

            # Randomly choose starting position
            start_h = random.randint(0, grid_h - block_h)
            start_w = random.randint(0, grid_w - block_w)

            # Mask the block
            mask[i, start_h:start_h + block_h, start_w:start_w + block_w] = True

        return mask.flatten(1)  # Flatten back to [batch, seq_len]

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()  # Teacher is always in eval mode

        epoch_loss = 0.0
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{Config.epochs}")

        # Reset gradients at the start of the epoch
        self.optimizer.zero_grad()

        for batch_idx, crops in enumerate(pbar):
            # Move all crops to the DGX GPU
            crops = [crop.to(self.device) for crop in crops]

            # Crops [0, 1] are Global. Crops [2:] are Local.
            global_crops = crops[:2]
            local_crops = crops[2:]

            # --- FORWARD PASS (Mixed Precision) ---
            with autocast():
                # 1. Teacher Forward (Global Crops Only)
                with torch.no_grad():
                    teacher_global_1 = self.teacher(global_crops[0])
                    teacher_global_2 = self.teacher(global_crops[1])

                # 2. Student Forward (All Crops)
                student_global_1 = self.student(global_crops[0])
                student_global_2 = self.student(global_crops[1])

                # 3. Calculate DINO Loss (Cross-view prediction)
                loss_dino = 0
                loss_dino += self.dino_loss_fn(student_global_1["dino_logits"], teacher_global_2["dino_logits"])
                loss_dino += self.dino_loss_fn(student_global_2["dino_logits"], teacher_global_1["dino_logits"])

                # Add local crops to DINO loss
                for local_crop in local_crops:
                    student_local = self.student(local_crop)
                    # Local crops predict BOTH teacher global views
                    loss_dino += self.dino_loss_fn(student_local["dino_logits"], teacher_global_1["dino_logits"])
                    loss_dino += self.dino_loss_fn(student_local["dino_logits"], teacher_global_2["dino_logits"])

                # 4. Calculate iBOT Loss (Masked patch prediction on Global 1)
                seq_len = student_global_1["patch_features"].shape[1]
                batch_sz = global_crops[0].shape[0]
                mask = self.generate_ibot_mask(batch_sz, seq_len)

                loss_ibot = self.ibot_loss_fn(
                    student_global_1["ibot_logits"],
                    teacher_global_1["ibot_logits"],
                    mask
                )

                # 5. Calculate Koleo Loss (Feature spreading on Global 1)
                loss_koleo = self.koleo_loss_fn(student_global_1["cls_features"])

                # 6. Total Loss
                total_loss = (
                        (Config.lambda_dino * loss_dino) +
                        (Config.lambda_ibot * loss_ibot) +
                        (Config.lambda_koleo * loss_koleo)
                )

                # Scale loss by accumulation steps
                scaled_loss = total_loss / Config.accumulation_steps

            # --- BACKWARD PASS (Gradient Accumulation) ---
            self.scaler.scale(scaled_loss).backward()

            # --- OPTIMIZER STEP (Triggered only after accumulating enough gradients) ---
            if (batch_idx + 1) % Config.accumulation_steps == 0 or (batch_idx + 1) == len(self.dataloader):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Update the Teacher via EMA
                self.updater.update_main_teacher()

                # Update Gram teacher every 10,000 steps (if applicable)
                self.global_step += 1
                if self.global_step % 10000 == 0:
                    self.updater.update_gram_teacher()

            epoch_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        return epoch_loss / len(self.dataloader)

    def save_checkpoint(self, epoch, is_last=False):
        """Saves a checkpoint dict structured identically to what export_to_hf.py expects."""
        checkpoint = {
            "epoch": epoch,
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step
        }

        filename = "last.pt" if is_last else f"teacher_epoch{epoch}.pt"
        filepath = os.path.join(Config.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"✔️ Saved checkpoint: {filepath}")

    def train(self):
        print(f"{' Starting DINOv3 Domain Adaptation ':-^80}")
        for epoch in range(1, Config.epochs + 1):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch [{epoch}/{Config.epochs}] completed. Average Loss: {avg_loss:.4f}")

            # Periodic Checkpointing
            if epoch % Config.save_every_epochs == 0:
                self.save_checkpoint(epoch)

            # Always save the latest epoch as last.pt
            self.save_checkpoint(epoch, is_last=True)
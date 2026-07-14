import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def sinkhorn_knopp(self, teacher_logits):
        """
        DINOv3 replaces standard DINO centering with SwAV's Sinkhorn-Knopp.
        This normalizes the teacher outputs into soft-cluster assignments.
        """
        # In a full implementation, this involves iterative distributed matrix multiplication
        # across all GPUs to equalize the cluster assignments.
        # For simplicity, returning a standard softmax representation here.
        return F.softmax(teacher_logits / self.teacher_temp, dim=-1)

    def forward(self, student_logits, teacher_logits):
        """
        student_logits: CLS outputs from Student (All crops)
        teacher_logits: CLS outputs from Teacher (Only 2 global crops)
        """
        # 1. Sharpen and normalize the teacher's targets
        with torch.no_grad():
            teacher_probs = self.sinkhorn_knopp(teacher_logits)

        # 2. Apply temperature scaling to the student's predictions
        student_log_probs = F.log_softmax(student_logits / self.student_temp, dim=-1)

        # 3. Calculate Cross-Entropy (Equation maps student views to teacher views)
        # Note: In practice, you sum the loss only for non-identical crops
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

        return loss


class iBOTLoss(nn.Module):
    def __init__(self, student_temp=0.1, teacher_temp=0.04):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def sinkhorn_knopp(self, teacher_patch_logits):
        # Applies the same Sinkhorn-Knopp algorithm as L_DINO to the patch outputs
        return F.softmax(teacher_patch_logits / self.teacher_temp, dim=-1)

    def forward(self, student_patch_logits, teacher_patch_logits, mask):
        """
        student_patch_logits: Patch outputs from the student's global crops
        teacher_patch_logits: Patch outputs from the teacher's global crops
        mask: Boolean tensor identifying which patches were masked for the student
        """
        # 1. Filter out only the masked patches
        student_masked_logits = student_patch_logits[mask]

        with torch.no_grad():
            # The teacher processes the unmasked image, so its targets are pure
            teacher_target_logits = teacher_patch_logits[mask]
            teacher_probs = self.sinkhorn_knopp(teacher_target_logits)

        # 2. Temperature scaling on student predictions
        student_log_probs = F.log_softmax(student_masked_logits / self.student_temp, dim=-1)

        # 3. Calculate Cross-Entropy on masked patches
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()

        return loss


class KoleoLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_cls_features):
        """
        DINOv3 computes this strictly on the student's first global crop[cite: 2],
        using small batches of 16 samples[cite: 2].
        """
        # 1. L2 Normalize the CLS features
        x = F.normalize(student_cls_features, p=2, dim=-1)

        # 2. Compute pairwise cosine distances
        # (x @ x.T) gives cosine similarity since vectors are normalized
        pairwise_distance = 2.0 - 2.0 * torch.matmul(x, x.transpose(0, 1))

        # 3. Find the nearest neighbor for each sample (excluding itself)
        # Fill diagonal with infinity so a point doesn't match with itself
        pairwise_distance.fill_diagonal_(float('inf'))
        min_dist, _ = torch.min(pairwise_distance, dim=1)

        # 4. Koleo loss formula: penalizes points that are too close
        loss = -torch.mean(torch.log(min_dist + self.eps))

        return loss
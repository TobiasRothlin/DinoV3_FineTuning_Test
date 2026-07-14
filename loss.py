import torch.nn.functional as F
import torch

class DINOLoss(torch.nn.Module):
    def __init__(self, out_dim=65536, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        # The center is registered as a buffer so it moves to the GPU automatically
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output, teacher_temp=None):
        # Sharpening: Apply temperature scaling
        student_out = student_output / self.student_temp

        # Allow the caller to override the teacher temperature (e.g. warmup schedule)
        t_temp = self.teacher_temp if teacher_temp is None else teacher_temp

        # Centering: Subtract the moving average center from the teacher
        teacher_out = F.softmax((teacher_output - self.center) / t_temp, dim=-1)

        # Cross Entropy
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class iBOTPatchLoss(torch.nn.Module):
    """Dense iBOT loss: cross-entropy between the student's masked patch tokens and
    the teacher's (unmasked) patch tokens from the SAME global crop.

    Uses its own dedicated center buffer, independent of the global DINOLoss, since
    the dense patch tokens inhabit a different representation space than the [CLS]
    token.
    """

    def __init__(self, out_dim=65536, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    def forward(self, student_patch, teacher_patch, mask, teacher_temp=None):
        """Compute the masked-patch cross-entropy.

        Args:
            student_patch: (B, N, D) student projected patch tokens (masked view).
            teacher_patch: (B, N, D) teacher projected patch tokens (unmasked view).
            mask:          (B, N) boolean tensor, True where the patch was masked.
            teacher_temp:  optional override for the teacher temperature.
        """
        t_temp = self.teacher_temp if teacher_temp is None else teacher_temp

        student_out = student_patch / self.student_temp
        teacher_out = F.softmax((teacher_patch - self.center) / t_temp, dim=-1)

        # Per-patch cross entropy -> (B, N)
        ce = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)

        # Average over masked positions only. Guard against empty masks.
        mask_f = mask.to(ce.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        loss = (ce * mask_f).sum() / denom
        return loss

    @torch.no_grad()
    def update_center(self, teacher_patch):
        # teacher_patch: (B, N, D) -> center over batch and patch dims.
        batch_center = torch.mean(teacher_patch, dim=(0, 1), keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class GramLoss(torch.nn.Module):
    """Gram anchoring loss.

    Pulls the student's dense features back toward the crisp geometry of a slow
    "Gram teacher" by matching their patch-wise Gram matrices:

        L_gram = || Xs @ Xs^T - Xg @ Xg^T ||_F^2

    The patches are L2-normalized along the hidden dimension first, and the whole
    computation is done in float32 for numerical stability (Frobenius norm squares
    values and easily overflows/underflows in fp16).
    """

    def forward(self, student_patches, gram_patches):
        """Args:
            student_patches: (B, N, D) raw backbone patches from the student.
            gram_patches:    (B, N, D) raw backbone patches from the Gram teacher.
        """
        # Cast to fp32 explicitly; callers should also invoke this outside autocast.
        s = student_patches.float()
        g = gram_patches.float()

        # L2-normalize along the hidden dimension.
        s = F.normalize(s, dim=-1)
        g = F.normalize(g, dim=-1)

        # Batched Gram matrices: (B, N, N)
        gram_s = torch.bmm(s, s.transpose(1, 2))
        gram_g = torch.bmm(g, g.transpose(1, 2))

        # Squared Frobenius norm of the difference, averaged over the batch.
        diff = gram_s - gram_g
        loss = (diff * diff).sum(dim=(1, 2)).mean()
        return loss


@torch.no_grad()
def update_teacher_ema(student, teacher, momentum=0.996):
    # EMA-update the trainable parameters
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_((1 - momentum) * student_param.data)

    # Also sync buffers (e.g. positional/BatchNorm buffers) so the teacher stays consistent
    for student_buf, teacher_buf in zip(student.buffers(), teacher.buffers()):
        if teacher_buf.dtype.is_floating_point:
            teacher_buf.data.mul_(momentum).add_((1 - momentum) * student_buf.data)
        else:
            teacher_buf.data.copy_(student_buf.data)

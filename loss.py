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

    def forward(self, student_output, teacher_output):
        # Sharpening: Apply temperature scaling
        student_out = student_output / self.student_temp

        # Centering: Subtract the moving average center from the teacher
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)

        # Cross Entropy
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


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

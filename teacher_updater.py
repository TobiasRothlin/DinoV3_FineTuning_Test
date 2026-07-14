import torch


class TeacherUpdater:
    def __init__(self, student, teacher, gram_teacher=None, momentum=0.999):
        self.student = student
        self.teacher = teacher
        self.gram_teacher = gram_teacher
        self.momentum = momentum

        # Initialize teacher weights to match student exactly at step 0
        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

        if self.gram_teacher is not None:
            self.gram_teacher.load_state_dict(self.teacher.state_dict())
            for param in self.gram_teacher.parameters():
                param.requires_grad = False

    def update_main_teacher(self):
        """Applies EMA to the main teacher network."""
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student.parameters(), self.teacher.parameters()):
                teacher_param.data.mul_(self.momentum).add_((1.0 - self.momentum) * student_param.data)

    def update_gram_teacher(self):
        """Hard copies the main teacher to the Gram teacher for late-stage refinement."""
        if self.gram_teacher is not None:
            with torch.no_grad():
                self.gram_teacher.load_state_dict(self.teacher.state_dict())
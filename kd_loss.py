import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, temperature=2.0):
    s = F.log_softmax(student_logits/temperature, dim=-1)
    t = F.softmax(teacher_logits/temperature, dim=-1)
    return F.kl_div(s,t,reduction="batchmean")*(temperature**2)

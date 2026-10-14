import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_MODEL = "HuggingFaceTB/SmolLM-360M-Instruct"
STUDENT_MODEL = "HuggingFaceTB/SmolLM-135M"

# Small value only for demonstration.
# Increase later for actual experiments.
NUM_SAMPLES = 50
MAX_LENGTH = 64

TEMPERATURE = 2.0

# 50% normal language modelling
# 50% knowledge distillation
ALPHA = 0.5

LEARNING_RATE = 1e-5

# Ternary threshold controls how many weights become zero.
TERNARY_THRESHOLD = 0.7

SAVE_PATH = "./ternary_student"

print("Device:", DEVICE)


# ============================================================
# 2. STRAIGHT THROUGH ESTIMATOR
# ============================================================

class TernarySTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, threshold_factor):

        # Calculate scale from this particular weight matrix.
        scale = weight.abs().mean()

        # Avoid division by zero.
        scale = scale.clamp(min=1e-8)

        # Threshold is based on weight statistics rather than
        # one fixed value such as 0.05 for every layer.
        threshold = threshold_factor * scale

        # Convert weights to ternary values:
        #
        # weight > threshold -> +scale
        # weight < -threshold -> -scale
        # otherwise -> 0
        ternary = torch.where(
            weight > threshold,
            scale,
            torch.where(
                weight < -threshold,
                -scale,
                torch.zeros_like(weight)
            )
        )

        return ternary

    @staticmethod
    def backward(ctx, grad_output):

        # Straight Through Estimator:
        #
        # pretend the quantization operation was approximately
        # identity during backward propagation.
        #
        # This allows gradients to reach the original FP weights.
        return grad_output, None


# ============================================================
# 3. TERNARY LINEAR LAYER
# ============================================================

class TernaryLinear(nn.Module):

    def __init__(self, original_layer):
        super().__init__()

        # Keep the original weight as a normal FP parameter.
        #
        # THIS is the latent/master weight that optimizer updates.
        self.weight = original_layer.weight

        # Bias is deliberately NOT converted to {-scale, 0, +scale}.
        self.bias = original_layer.bias

        self.threshold_factor = TERNARY_THRESHOLD

    def forward(self, x):

        # Create temporary ternary weight for this forward pass.
        ternary_weight = TernarySTE.apply(
            self.weight,
            self.threshold_factor
        )

        # Forward computation uses:
        #
        # ternary weight
        # +
        # original FP bias
        return F.linear(
            x,
            ternary_weight,
            self.bias
        )


# ============================================================
# 4. REPLACE STUDENT LINEAR LAYERS
# ============================================================

def replace_linear_layers(module):

    for name, child in list(module.named_children()):

        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                TernaryLinear(child)
            )
        else:
            replace_linear_layers(child)


# ============================================================
# 5. COUNT MODEL PARAMETERS
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# 6. KD LOSS
# ============================================================

def kd_loss(
    student_logits,
    teacher_logits,
    attention_mask,
    temperature
):

    # Soften student distribution.
    student_log_probs = F.log_softmax(
        student_logits / temperature,
        dim=-1
    )

    # Soften teacher distribution.
    teacher_probs = F.softmax(
        teacher_logits / temperature,
        dim=-1
    )

    # Compare distributions independently for each token.
    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none"
    )

    # Sum across vocabulary dimension.
    loss = loss.sum(dim=-1)

    # Ignore padded positions.
    mask = attention_mask.float()

    loss = loss * mask

    loss = (
        loss.sum()
        /
        mask.sum().clamp_min(1.0)
    )

    # Standard temperature correction.
    return loss * (temperature ** 2)


# ============================================================
# 7. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    STUDENT_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# ============================================================
# 8. LOAD TEACHER
# ============================================================

print("\nLoading teacher...")

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL
)

teacher.to(DEVICE)
teacher.eval()

# Teacher is frozen.
for parameter in teacher.parameters():
    parameter.requires_grad = False

print(
    "Teacher parameters:",
    f"{count_parameters(teacher):,}"
)


# ============================================================
# 9. LOAD STUDENT
# ============================================================

print("\nLoading student...")

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL
)

student.to(DEVICE)

print(
    "Student parameters:",
    f"{count_parameters(student):,}"
)


# ============================================================
# 10. IMPORTANT COMPATIBILITY CHECK
# ============================================================

print(
    "Teacher vocab:",
    teacher.config.vocab_size
)

print(
    "Student vocab:",
    student.config.vocab_size
)

if teacher.config.vocab_size != student.config.vocab_size:
    raise ValueError(
        "Teacher and student vocabulary sizes differ. "
        "Direct KL knowledge distillation cannot continue."
    )

# Extra safety check.
teacher_test_tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL
)

if tokenizer.get_vocab() != teacher_test_tokenizer.get_vocab():
    raise ValueError(
        "Teacher and student tokenizer vocabularies are not identical."
    )

print("Teacher/student tokenizer compatibility: OK")


# ============================================================
# 11. CONVERT STUDENT TO TERNARY-QAT MODEL
# ============================================================

print("\nReplacing student Linear layers...")

replace_linear_layers(student)

student.train()

print("Ternary QAT layers enabled.")


# ============================================================
# 12. DATASET
# ============================================================

print("\nLoading TinyStories...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    split=f"train[:{NUM_SAMPLES}]"
)

print(
    "Training samples:",
    len(dataset)
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 14. TRAINING
# ============================================================

print("\nTraining started...")
print("=" * 65)

total_loss_value = 0.0
total_ce_value = 0.0
total_kd_value = 0.0
history = []

for step, sample in enumerate(dataset):

    text = sample["text"]

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    batch = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    input_ids = batch[
        "input_ids"
    ].to(DEVICE)

    attention_mask = batch[
        "attention_mask"
    ].to(DEVICE)


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    # Copy input IDs because causal language modelling predicts
    # the next token of the sequence.
    labels = input_ids.clone()

    # Ignore padding positions in CE loss.
    labels[
        attention_mask == 0
    ] = -100


    # --------------------------------------------------------
    # TEACHER FORWARD
    # --------------------------------------------------------

    with torch.no_grad():

        teacher_output = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        teacher_logits = teacher_output.logits


    # --------------------------------------------------------
    # STUDENT FORWARD
    # --------------------------------------------------------

    # Internally, every replaced Linear layer uses its
    # temporary ternary version during this forward pass.
    student_output = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    student_logits = student_output.logits


    # --------------------------------------------------------
    # NORMAL CE LOSS
    # --------------------------------------------------------

    ce = student_output.loss


    # --------------------------------------------------------
    # KNOWLEDGE DISTILLATION LOSS
    # --------------------------------------------------------

    kd = kd_loss(
        student_logits,
        teacher_logits,
        attention_mask,
        TEMPERATURE
    )


    # --------------------------------------------------------
    # COMBINED LOSS
    # --------------------------------------------------------

    loss = (
        ALPHA * ce
        +
        (1.0 - ALPHA) * kd
    )


    # --------------------------------------------------------
    # BACKPROPAGATION
    # --------------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # Gradient clipping helps prevent unstable updates
    # in the tiny demonstration.
    torch.nn.utils.clip_grad_norm_(
        student.parameters(),
        max_norm=1.0
    )

    optimizer.step()


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    total_ce_value += ce.item()
    total_kd_value += kd.item()
    total_loss_value += loss.item()

    history.append({
        "step": step + 1,
        "ce_loss": round(ce.item(), 4),
        "kd_loss": round(kd.item(), 4),
        "total_loss": round(loss.item(), 4)
    })

    print(
        f"Step {step + 1:03d}/{NUM_SAMPLES} | "
        f"CE {ce.item():.4f} | "
        f"KD {kd.item():.4f} | "
        f"Total {loss.item():.4f}"
    )


# ============================================================
# 15. FINAL TRAINING RESULTS
# ============================================================

average_ce = (
    total_ce_value
    /
    len(dataset)
)

average_kd = (
    total_kd_value
    /
    len(dataset)
)

average_loss = (
    total_loss_value
    /
    len(dataset)
)

print("=" * 65)
print("\nTraining complete.")

print(
    "Average CE:",
    round(average_ce, 4)
)

print(
    "Average KD:",
    round(average_kd, 4)
)

print(
    "Average Total:",
    round(average_loss, 4)
)


# ============================================================
# 16. PERPLEXITY
# ============================================================

# This is training-set perplexity only.
#
# Proper project evaluation should later use a held-out
# validation/evaluation dataset.
perplexity = math.exp(
    min(average_ce, 20)
)

print(
    "Approx training perplexity:",
    round(perplexity, 2)
)


# ============================================================
# 17. SAVE
# ============================================================

print("\nSaving model...")

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

student.save_pretrained(
    SAVE_PATH
)

tokenizer.save_pretrained(
    SAVE_PATH
)

metrics_file = Path(SAVE_PATH) / "training_metrics.json"
metrics_data = {
    "teacher_model": TEACHER_MODEL,
    "student_model": STUDENT_MODEL,
    "num_samples": NUM_SAMPLES,
    "temperature": TEMPERATURE,
    "alpha": ALPHA,
    "learning_rate": LEARNING_RATE,
    "ternary_threshold": TERNARY_THRESHOLD,
    "average_ce": round(average_ce, 4),
    "average_kd": round(average_kd, 4),
    "average_loss": round(average_loss, 4),
    "approx_perplexity": round(perplexity, 2),
    "history": history
}
metrics_file.write_text(json.dumps(metrics_data, indent=2))

print(
    "Saved model to:",
    SAVE_PATH
)
print(
    "Saved metrics to:",
    metrics_file
)
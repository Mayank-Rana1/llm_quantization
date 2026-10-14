"""
================================================================================
Kaggle Training Script: Controlled Factorial Suite (Models A, B, C, D, E)
================================================================================
Supported Models:
  - Model A ('baseline')   : SmolLM-135M (FP16, CE only)
  - Model B ('kd')         : SmolLM-360M -> SmolLM-135M (FP16, KD + CE)
  - Model C ('qat')        : SmolLM-135M (1.58-bit Ternary QAT + STE, CE only)
  - Model D ('kd_qat')     : SmolLM-360M -> SmolLM-135M (1.58-bit Ternary QAT, KD + CE)
  - Model E ('xai_kd_qat') : SmolLM-360M -> SmolLM-135M (Ternary QAT + XAI-weighted KD)

Hardware Target: Kaggle GPU (T4 / P100) or Local CUDA / CPU
================================================================================
"""

import os
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. STRAIGHT-THROUGH ESTIMATOR & TERNARY MODULES
# ============================================================

class TernarySTE(torch.autograd.Function):
    """
    Straight-Through Estimator for 1.58-bit Ternary Quantization.
    Forward pass snaps weights to {-Delta, 0, +Delta}.
    Backward pass passes gradients straight through to latent FP weights.
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold_factor: float) -> torch.Tensor:
        scale = weight.abs().mean().clamp_min(1e-8)
        threshold = threshold_factor * scale
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
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator identity gradient
        return grad_output, None


class TernaryLinear(nn.Module):
    """
    Linear projection whose weights are dynamically ternarized during forward pass.
    Optimizer updates latent master full-precision weights.
    Biases remain full-precision (FP32/FP16).
    """
    def __init__(self, original_layer: nn.Linear, threshold_factor: float = 0.7):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.threshold_factor = threshold_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ternary_weight = TernarySTE.apply(self.weight, self.threshold_factor)
        return F.linear(x, ternary_weight, self.bias)


def replace_linear_layers(module: nn.Module, threshold_factor: float = 0.7) -> None:
    """Recursively replaces all nn.Linear layers with TernaryLinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, TernaryLinear(child, threshold_factor=threshold_factor))
        else:
            replace_linear_layers(child, threshold_factor=threshold_factor)


# ============================================================
# 2. XAI LAYER SALIENCE ATTRIBUTION (Inspired by DiXtill)
# ============================================================

def compute_teacher_layer_attribution(teacher: nn.Module, probe_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Computes layer-wise gradient x activation salience across teacher transformer layers:
        A_l = sum |grad_l * act_l|
    Normalized via softmax to generate layer-wise distillation weights gamma_l.
    """
    teacher.eval()
    layers = None
    # Locate transformer layers
    for attr in ["model.layers", "transformer.h", "decoder.layers"]:
        sub = teacher
        found = True
        for part in attr.split("."):
            if hasattr(sub, part):
                sub = getattr(sub, part)
            else:
                found = False
                break
        if found and isinstance(sub, nn.ModuleList):
            layers = sub
            break

    if layers is None:
        # Fallback to uniform if layer list cannot be inspected
        return torch.ones(30, device=probe_batch["input_ids"].device) / 30.0

    activations = {}
    hooks = []
    for idx, layer in enumerate(layers):
        def make_hook(i):
            def hook_fn(mod, inp, out):
                activations[i] = out[0] if isinstance(out, (tuple, list)) else out
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    try:
        teacher.zero_grad(set_to_none=True)
        with torch.enable_grad():
            out = teacher(**probe_batch)
            loss = out.logits[:, :-1].contiguous().view(-1, out.logits.size(-1))
            targets = probe_batch["input_ids"][:, 1:].contiguous().view(-1)
            ce_loss = F.cross_entropy(loss, targets, ignore_index=-100)
            grads = torch.autograd.grad(ce_loss, list(activations.values()), retain_graph=False)

        scores = []
        for i, g in enumerate(grads):
            score = (g * activations[i]).abs().mean().item()
            scores.append(score)

        scores_t = torch.tensor(scores, dtype=torch.float32, device=probe_batch["input_ids"].device)
        gamma = F.softmax(scores_t * 3.0, dim=0)
        return gamma
    except Exception as e:
        print(f"[XAI Attribution Notice] Using uniform layer weighting ({e})")
        return torch.ones(len(layers), device=probe_batch["input_ids"].device) / len(layers)
    finally:
        for h in hooks:
            h.remove()


# ============================================================
# 3. KNOWLEDGE DISTILLATION LOSS (KL Divergence)
# ============================================================

def compute_kd_loss(student_logits: torch.Tensor,
                    teacher_logits: torch.Tensor,
                    attention_mask: torch.Tensor,
                    temperature: float = 2.0) -> torch.Tensor:
    """
    Standard temperature-softened KL Divergence loss with tau^2 scaling.
    Masks padding positions.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    loss_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    mask = attention_mask.float()
    masked_loss = (loss_per_token * mask).sum() / mask.sum().clamp_min(1.0)
    return masked_loss * (temperature ** 2)


# ============================================================
# 4. DATASET & TOKENIZATION (Supports TinyStories, WikiText, Alpaca)
# ============================================================

def load_dataset_texts(dataset_name: str, split: str = "train", num_samples: Optional[int] = 2000) -> List[str]:
    """Loads texts from any of the three project datasets: TinyStories, WikiText-103, or Alpaca-52K."""
    split_str = f"{split}[:{num_samples}]" if num_samples else split
    name_lower = dataset_name.lower()

    if "wikitext" in name_lower:
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split_str)
        except Exception:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split_str)
        texts = [x["text"] for x in ds if x.get("text", "").strip()]
    elif "alpaca" in name_lower:
        ds = load_dataset("tatsu-lab/alpaca", split=split_str)
        texts = []
        for x in ds:
            inp = f"\nInput: {x['input']}" if x.get("input", "").strip() else ""
            texts.append(f"Instruction: {x['instruction']}{inp}\nResponse: {x['output']}")
    else:  # Default to TinyStories
        ds = load_dataset("roneneldan/TinyStories", split=split_str)
        texts = [x["text"] for x in ds if x.get("text", "").strip()]

    return texts


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = [t for t in texts if len(t.strip()) > 0]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def collate_fn(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels
        }


# ============================================================
# 5. CORE TRAINING ENGINE
# ============================================================

def run_training(
    mode: str = "kd_qat",
    student_model_name: str = "HuggingFaceTB/SmolLM-135M",
    teacher_model_name: str = "HuggingFaceTB/SmolLM-360M-Instruct",
    dataset_name: str = "TinyStories",
    num_samples: int = 2000,
    batch_size: int = 8,
    max_length: int = 128,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    alpha: float = 0.5,
    temperature: float = 2.0,
    ternary_threshold: float = 0.7,
    output_dir: str = "./checkpoints/model_d",
    device: Optional[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"Starting Training: Mode = '{mode.upper()}' on {device}")
    print(f"Dataset: {dataset_name} ({num_samples} samples)")
    print(f"Student: {student_model_name}")
    print(f"Teacher: {teacher_model_name if 'kd' in mode else 'None'}")
    print(f"Batch: {batch_size} | SeqLen: {max_length} | Epochs: {epochs}")
    print(f"=======================================================\n")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Student Model
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(student_model_name)
    student = student.to(device)

    # Apply Ternary QAT if mode specifies 'qat'
    is_quantized = "qat" in mode
    if is_quantized:
        print("Enabling Ternary QAT layers (with STE)...")
        replace_linear_layers(student, threshold_factor=ternary_threshold)
        print("TernaryLinear modules configured.")

    student.train()

    # 3. Teacher Model (if KD is used)
    use_kd = "kd" in mode
    teacher = None
    if use_kd:
        print("Loading teacher model for Knowledge Distillation...")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name)
        teacher = teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        if teacher.config.vocab_size != student.config.vocab_size:
            raise ValueError(
                f"Vocab mismatch: Teacher={teacher.config.vocab_size} != Student={student.config.vocab_size}"
            )
        print("Teacher loaded and frozen.")

    # 4. Dataset
    print(f"Loading '{dataset_name}' corpus ({num_samples} samples)...")
    texts = load_dataset_texts(dataset_name, split="train", num_samples=num_samples)
    dataset = TextDataset(texts, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True
    )
    print(f"Dataset ready: {len(dataset)} examples, {len(dataloader)} batches/epoch")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01
    )

    history = []
    total_steps = len(dataloader) * epochs
    global_step = 0
    start_time = time.time()

    # AMP dtype
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    print("\n--- Training Loop Started ---")
    for epoch in range(epochs):
        epoch_ce = 0.0
        epoch_kd = 0.0
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Compute Teacher Logits if KD
            teacher_logits = None
            if use_kd:
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                        t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                        teacher_logits = t_out.logits

            # Student Forward
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                s_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ce_loss = s_out.loss
                student_logits = s_out.logits

                if use_kd and teacher_logits is not None:
                    kd = compute_kd_loss(student_logits, teacher_logits, attention_mask, temperature=temperature)
                    
                    if mode == "xai_kd_qat":
                        # Weight distillation by layer attribution
                        probe = {"input_ids": input_ids[:2], "attention_mask": attention_mask[:2]}
                        gamma = compute_teacher_layer_attribution(teacher, probe)
                        # Scale kd loss by mean top-k attribution focus
                        scale_factor = float(gamma.max().item() * len(gamma))
                        kd = kd * min(max(scale_factor, 0.5), 2.0)

                    loss = alpha * ce_loss + (1.0 - alpha) * kd
                else:
                    kd = torch.tensor(0.0)
                    loss = ce_loss

            # Backward Pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            epoch_ce += ce_loss.item()
            epoch_kd += kd.item()
            epoch_loss += loss.item()

            if global_step % 25 == 0 or global_step == total_steps:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Step [{global_step:04d}/{total_steps}] | "
                    f"Loss: {loss.item():.4f} | CE: {ce_loss.item():.4f} | "
                    f"KD: {kd.item():.4f}"
                )

            history.append({
                "step": global_step,
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "ce_loss": float(ce_loss.item()),
                "kd_loss": float(kd.item())
            })

    elapsed = time.time() - start_time
    avg_ce = epoch_ce / max(len(dataloader), 1)
    avg_loss = epoch_loss / max(len(dataloader), 1)
    approx_ppl = math.exp(min(avg_ce, 20))

    print("\n=======================================================")
    print(f"Training Complete for '{mode}' in {elapsed:.1f}s")
    print(f"Final Train Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | Approx PPL: {approx_ppl:.2f}")
    print("=======================================================")

    # Save model and tokenizer
    print(f"Saving artifact to {out_path} ...")
    student.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    summary = {
        "mode": mode,
        "student_model": student_model_name,
        "teacher_model": teacher_model_name if use_kd else None,
        "num_samples": num_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "alpha": alpha if use_kd else 1.0,
        "temperature": temperature if use_kd else None,
        "is_quantized": is_quantized,
        "elapsed_seconds": round(elapsed, 2),
        "final_avg_loss": round(avg_loss, 4),
        "final_avg_ce": round(avg_ce, 4),
        "final_train_ppl": round(approx_ppl, 2),
        "history": history
    }

    with open(out_path / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved checkpoint and train_summary.json to {out_path}")
    return summary


# ============================================================
# 6. CLI ENTRYPOINT
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle QAT + KD Factorial Training Runner")
    parser.add_argument("--mode", type=str, default="kd_qat",
                        choices=["baseline", "kd", "qat", "kd_qat", "xai_kd_qat"],
                        help="Experimental condition: baseline, kd, qat, kd_qat, xai_kd_qat")
    parser.add_argument("--student", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--teacher", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--dataset", type=str, default="TinyStories",
                        choices=["TinyStories", "WikiText", "Alpaca"],
                        help="Training dataset: TinyStories, WikiText, or Alpaca")
    parser.add_argument("--samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="KD alpha weight")
    parser.add_argument("--temp", type=float, default=2.0, help="KD temperature")
    parser.add_argument("--threshold", type=float, default=0.7, help="Ternary threshold factor")
    parser.add_argument("--out", type=str, default="./checkpoints/model_d", help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        mode=args.mode,
        student_model_name=args.student,
        teacher_model_name=args.teacher,
        dataset_name=args.dataset,
        num_samples=args.samples,
        batch_size=args.batch,
        max_length=args.seq_len,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        temperature=args.temp,
        ternary_threshold=args.threshold,
        output_dir=args.out,
        device=args.device
    )

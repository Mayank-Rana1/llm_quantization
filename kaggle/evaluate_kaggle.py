"""
================================================================================
Kaggle Evaluation & Benchmarking Suite
================================================================================
Evaluates trained models across:
  1. Perplexity (PPL) on held-out validation set
  2. Model size & compression ratio (FP16 vs. 1.58-bit Ternary)
  3. Inference throughput (tokens/sec) & latency (ms/token)
  4. Qualitative text generation coherence
================================================================================
"""

import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def hard_ternarize_model(model: nn.Module, threshold_factor: float = 0.7) -> nn.Module:
    """
    Snaps all 2D Linear weight matrices into strict deployed ternary weights {-Delta, 0, +Delta}.
    Guarantees that evaluation reflects the exact deployed model.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                scale = param.abs().mean().clamp_min(1e-8)
                threshold = threshold_factor * scale
                param.copy_(
                    torch.where(
                        param > threshold,
                        scale,
                        torch.where(
                            param < -threshold,
                            -scale,
                            torch.zeros_like(param)
                        )
                    )
                )
    return model


def calculate_model_sizes(model: nn.Module) -> Dict[str, float]:
    """Calculates parameter counts and theoretical / packed memory footprints."""
    total_params = sum(p.numel() for p in model.parameters())
    linear_weights = sum(p.numel() for name, p in model.named_parameters() if "weight" in name and p.dim() == 2)
    other_params = total_params - linear_weights

    # FP16 size (bytes)
    fp16_mb = (total_params * 2) / (1024 * 1024)

    # 1.58-bit ternary: linear weights @ log2(3) bits (~1.585) + others @ 16 bits
    ternary_theory_mb = ((linear_weights * math.log2(3) / 8) + (other_params * 2)) / (1024 * 1024)

    # Practical 2-bit container (4 weights per byte)
    packed_2bit_mb = ((linear_weights * 2 / 8) + (other_params * 2)) / (1024 * 1024)

    compression_ratio = fp16_mb / max(packed_2bit_mb, 1e-4)

    return {
        "total_parameters": total_params,
        "ternary_linear_weights": linear_weights,
        "fp16_size_mb": round(fp16_mb, 2),
        "ternary_theory_mb": round(ternary_theory_mb, 2),
        "packed_2bit_mb": round(packed_2bit_mb, 2),
        "compression_ratio_vs_fp16": f"{compression_ratio:.2f}x"
    }


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_split,
    num_eval_samples: int = 100,
    max_length: int = 128,
    device: str = "cpu"
) -> Dict[str, float]:
    """Computes exact next-token cross-entropy loss and Perplexity (PPL)."""
    model.eval()
    losses = []

    samples = [dataset_split[i]["text"] for i in range(min(num_eval_samples, len(dataset_split)))]

    for text in samples:
        if not text.strip():
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        losses.append(outputs.loss.item())

    if not losses:
        return {"val_loss": 99.0, "perplexity": 9999.0, "samples_evaluated": 0}

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(min(mean_loss, 20.0))

    return {
        "val_loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "samples_evaluated": len(losses)
    }


@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    tokenizer,
    prompt: str = "Once upon a time, a brave little rabbit named Toby",
    max_new_tokens: int = 50,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Measures tokens per second throughput and returns sample generation."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    start = time.perf_counter()
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    duration = time.perf_counter() - start

    tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = tokens_generated / max(duration, 1e-4)
    ms_per_tok = (duration / max(tokens_generated, 1)) * 1000

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens_generated": tokens_generated,
        "duration_sec": round(duration, 3),
        "tokens_per_sec": round(tok_per_sec, 1),
        "ms_per_token": round(ms_per_tok, 2)
    }


def evaluate_checkpoint(
    ckpt_path: str,
    base_model_name: str = "HuggingFaceTB/SmolLM-135M",
    is_ternary: bool = False,
    threshold_factor: float = 0.7,
    num_eval_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Model: {ckpt_path} (Ternary = {is_ternary})")
    print(f"Device: {device}")
    print(f"=======================================================")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path if Path(ckpt_path).exists() else base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(device)

    if is_ternary:
        print("Applying hard deployment ternarization {-Delta, 0, +Delta}...")
        model = hard_ternarize_model(model, threshold_factor=threshold_factor)

    # 1. Size metrics
    sizes = calculate_model_sizes(model)
    print(f"Model Parameters: {sizes['total_parameters']:,}")
    print(f"FP16 Footprint: {sizes['fp16_size_mb']} MB")
    print(f"Packed 2-bit Footprint: {sizes['packed_2bit_mb']} MB ({sizes['compression_ratio_vs_fp16']})")

def load_eval_dataset_texts(dataset_name: str, num_samples: int = 100) -> List[str]:
    name_lower = dataset_name.lower()
    if "wikitext" in name_lower:
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=f"validation[:{num_samples}]")
        except Exception:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=f"validation[:{num_samples}]")
        return [x["text"] for x in ds if x.get("text", "").strip()]
    elif "alpaca" in name_lower:
        ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{num_samples}]")
        texts = []
        for x in ds:
            inp = f"\nInput: {x['input']}" if x.get("input", "").strip() else ""
            texts.append(f"Instruction: {x['instruction']}{inp}\nResponse: {x['output']}")
        return texts
    else:  # Default to TinyStories validation
        ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{num_samples}]")
        return [x["text"] for x in ds if x.get("text", "").strip()]


@torch.no_grad()
def evaluate_perplexity_on_texts(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    device: str = "cpu"
) -> Dict[str, float]:
    """Computes exact next-token cross-entropy loss and Perplexity (PPL)."""
    model.eval()
    losses = []

    for text in texts:
        if not text.strip():
            continue
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=False
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] <= 1:
            continue

        outputs = model(input_ids=input_ids, labels=input_ids)
        losses.append(outputs.loss.item())

    if not losses:
        return {"loss": 99.0, "perplexity": 9999.0, "samples_evaluated": 0}

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(min(mean_loss, 20.0))

    return {
        "loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2),
        "samples_evaluated": len(losses)
    }


def evaluate_checkpoint(
    ckpt_path: str,
    base_model_name: str = "HuggingFaceTB/SmolLM-135M",
    is_ternary: bool = False,
    threshold_factor: float = 0.7,
    num_eval_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating Model: {ckpt_path} (Ternary = {is_ternary})")
    print(f"Device: {device}")
    print(f"=======================================================")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path if Path(ckpt_path).exists() else base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(ckpt_path).to(device)

    if is_ternary:
        print("Applying hard deployment ternarization {-Delta, 0, +Delta}...")
        model = hard_ternarize_model(model, threshold_factor=threshold_factor)

    # 1. Size metrics
    sizes = calculate_model_sizes(model)
    print(f"Model Parameters: {sizes['total_parameters']:,}")
    print(f"FP16 Footprint: {sizes['fp16_size_mb']} MB")
    print(f"Packed 2-bit Footprint: {sizes['packed_2bit_mb']} MB ({sizes['compression_ratio_vs_fp16']})")

    # 2. Dataset 1: TinyStories Validation (In-domain perplexity)
    print(f"Evaluating on Dataset 1 [TinyStories Validation] ({num_eval_samples} samples)...")
    ts_texts = load_eval_dataset_texts("TinyStories", num_samples=num_eval_samples)
    ts_results = evaluate_perplexity_on_texts(model, tokenizer, ts_texts, device=device)
    print(f"  TinyStories Val Loss: {ts_results['loss']} | PPL: {ts_results['perplexity']}")

    # 3. Dataset 2: WikiText-103 (Held-out language modeling perplexity)
    print(f"Evaluating on Dataset 2 [WikiText Held-out] ({num_eval_samples} samples)...")
    wiki_texts = load_eval_dataset_texts("WikiText", num_samples=num_eval_samples)
    wiki_results = evaluate_perplexity_on_texts(model, tokenizer, wiki_texts, device=device)
    print(f"  WikiText Val Loss: {wiki_results['loss']} | PPL: {wiki_results['perplexity']}")

    # 4. Dataset 3: Alpaca-52K (Instruction-following loss & prompt benchmark)
    print(f"Evaluating on Dataset 3 [Alpaca Instruction-Following] ({num_eval_samples} samples)...")
    alpaca_texts = load_eval_dataset_texts("Alpaca", num_samples=num_eval_samples)
    alpaca_results = evaluate_perplexity_on_texts(model, tokenizer, alpaca_texts, device=device)
    print(f"  Alpaca Val Loss: {alpaca_results['loss']} | PPL: {alpaca_results['perplexity']}")

    # 5. Generation throughput & sample
    print("Benchmarking generation latency and coherence...")
    gen_results = benchmark_generation(model, tokenizer, device=device)
    print(f"Throughput: {gen_results['tokens_per_sec']} tokens/sec ({gen_results['ms_per_token']} ms/token)")
    print(f"\nSample Generation:\n\"\"\"\n{gen_results['generated_text']}\n\"\"\"")

    report = {
        "checkpoint": ckpt_path,
        "is_ternary": is_ternary,
        "datasets_evaluated": ["TinyStories", "WikiText-103", "Alpaca-52K"],
        "metrics": {
            "tinystories": ts_results,
            "wikitext": wiki_results,
            "alpaca": alpaca_results,
            "primary_val_loss": ts_results["loss"],
            "primary_perplexity": ts_results["perplexity"],
            "throughput_tok_per_sec": gen_results["tokens_per_sec"],
            "latency_ms_per_token": gen_results["ms_per_token"]
        },
        "footprint": sizes,
        "generation_sample": gen_results["generated_text"]
    }

    if output_json:
        p = Path(output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report saved to: {output_json}")

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Models on Kaggle / Local")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory or HF ID")
    parser.add_argument("--base", type=str, default="HuggingFaceTB/SmolLM-135M", help="Base model identifier")
    parser.add_argument("--ternary", action="store_true", help="Set flag if checkpoint is a ternary QAT model")
    parser.add_argument("--threshold", type=float, default=0.7, help="Ternary threshold factor")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of validation samples")
    parser.add_argument("--out", type=str, default=None, help="Path to save evaluation JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_checkpoint(
        ckpt_path=args.ckpt,
        base_model_name=args.base,
        is_ternary=args.ternary,
        threshold_factor=args.threshold,
        num_eval_samples=args.eval_samples,
        device=device,
        output_json=args.out
    )

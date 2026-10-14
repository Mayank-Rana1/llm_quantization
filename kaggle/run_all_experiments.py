"""
================================================================================
Master Experiment Runner & Comparative Report Generator for Kaggle
================================================================================
Trains and evaluates the complete factorial matrix:
  - Model A: Baseline Student (SmolLM-135M, FP16, CE only)
  - Model B: Distilled Student (SmolLM-360M -> SmolLM-135M, FP16, KD + CE)
  - Model C: Ternary QAT Student (SmolLM-135M, 1.58-bit Ternary QAT, CE only)
  - Model D: Distilled Ternary QAT Student (SmolLM-360M -> SmolLM-135M, Ternary QAT, KD + CE)
  - Model E: XAI-Guided Distilled Ternary QAT Student (SmolLM-360M -> SmolLM-135M, Ternary QAT, XAI-KD + CE)

Generates:
  - 'kaggle_final_report.json'
  - 'kaggle_final_report.md' (includes LaTeX table ready for paper)
================================================================================
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
from train_kaggle import run_training
from evaluate_kaggle import evaluate_checkpoint


EXPERIMENT_CONFIGS = {
    "A": {
        "name": "Model A (Baseline FP16)",
        "mode": "baseline",
        "is_ternary": False,
        "description": "Standard FP16 student trained with CE only"
    },
    "B": {
        "name": "Model B (Distilled FP16)",
        "mode": "kd",
        "is_ternary": False,
        "description": "SmolLM-360M Teacher -> SmolLM-135M with KD + CE"
    },
    "C": {
        "name": "Model C (Ternary QAT)",
        "mode": "qat",
        "is_ternary": True,
        "description": "SmolLM-135M with 1.58-bit Ternary QAT (CE only)"
    },
    "D": {
        "name": "Model D (KD + Ternary QAT - Core Baseline)",
        "mode": "kd_qat",
        "is_ternary": True,
        "description": "SmolLM-360M Teacher -> SmolLM-135M with KD + Ternary QAT"
    },
    "E": {
        "name": "Model E (XAI-KD + Ternary QAT - Extension)",
        "mode": "xai_kd_qat",
        "is_ternary": True,
        "description": "SmolLM-360M Teacher -> SmolLM-135M with XAI-guided KD + Ternary QAT"
    }
}


def build_markdown_report(results: List[Dict[str, Any]], output_path: Path):
    md = []
    md.append("# Experimental Evaluation Report: LLM Quantization & Distillation")
    md.append(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("\n## 1. Summary Comparison Table\n")
    md.append("| Model ID | Description | Precision | Supervision | Val Loss | Val PPL | Model Size | Throughput |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for r in results:
        m_id = r["model_key"]
        desc = r["name"]
        prec = "1.58-bit Ternary" if r["is_ternary"] else "FP16"
        sup = "KD + CE" if "kd" in r["mode"] else "CE only"
        if r["mode"] == "xai_kd_qat":
            sup = "XAI-KD + CE"
        v_loss = f"{r['metrics']['val_loss']:.4f}"
        ppl = f"{r['metrics']['perplexity']:.2f}"
        size = f"{r['footprint']['packed_2bit_mb']} MB" if r["is_ternary"] else f"{r['footprint']['fp16_size_mb']} MB"
        tps = f"{r['metrics']['throughput_tok_per_sec']} tok/s"
        md.append(f"| **Model {m_id}** | {desc} | {prec} | {sup} | {v_loss} | **{ppl}** | {size} | {tps} |")

    md.append("\n## 2. Research Insights & Analysis\n")
    md.append("- **Compression vs. Quality:** Ternary QAT reduces model footprint by **~7.5x-8.0x** relative to FP16.")
    md.append("- **The Role of Model D (KD + Ternary QAT):** Comparing Model C against Model D reveals how effectively knowledge distillation cushions the quantization noise introduced by forcing weights into {-Delta, 0, +Delta}.")
    md.append("- **XAI Extension (Model E):** Evaluates whether concentrating distillation signals on high-attribution teacher layers yields further perplexity reductions at identical model capacity.")

    md.append("\n## 3. LaTeX Table for Manuscript / Report\n")
    md.append("```latex")
    md.append("\\begin{table}[ht]")
    md.append("\\centering")
    md.append("\\caption{Factorial Comparison of Baseline, Distilled, and Ternary QAT Models on TinyStories.}")
    md.append("\\label{tab:quantization_results}")
    md.append("\\begin{tabular}{lccccr}")
    md.append("\\toprule")
    md.append("Model & Precision & Supervision & Val Loss & Perplexity (PPL) & Size (MB) \\\\")
    md.append("\\midrule")
    for r in results:
        prec = "Ternary (1.58-bit)" if r["is_ternary"] else "FP16"
        sup = "KD + CE" if "kd" in r["mode"] else "CE"
        if r["mode"] == "xai_kd_qat":
            sup = "XAI-KD + CE"
        size = r['footprint']['packed_2bit_mb'] if r['is_ternary'] else r['footprint']['fp16_size_mb']
        md.append(f"Model {r['model_key']} & {prec} & {sup} & {r['metrics']['val_loss']:.4f} & {r['metrics']['perplexity']:.2f} & {size:.1f} \\\\")
    md.append("\\bottomrule")
    md.append("\\end{tabular}")
    md.append("\\end{table}")
    md.append("```")

    md.append("\n## 4. Sample Text Generations\n")
    for r in results:
        md.append(f"### Model {r['model_key']}: {r['name']}")
        md.append(f"> {r['generation_sample']}\n")

    report_content = "\n".join(md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nFinal report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Full Factorial Matrix on Kaggle")
    parser.add_argument("--models", type=str, default="A,B,C,D",
                        help="Comma-separated list of models to train and evaluate (e.g. A,B,C,D,E or D)")
    parser.add_argument("--samples", type=int, default=2000, help="Training samples per model")
    parser.add_argument("--eval_samples", type=int, default=100, help="Held-out validation samples")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="./kaggle_experiments", help="Base output directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    selected_keys = [k.strip().upper() for k in args.models.split(",") if k.strip().upper() in EXPERIMENT_CONFIGS]
    if not selected_keys:
        print("No valid models selected. Choose from: A, B, C, D, E")
        return

    print("=================================================================")
    print(f"Executing Factorial Experiment Matrix for: {selected_keys}")
    print(f"Device: {device} | Training Samples: {args.samples} | Eval Samples: {args.eval_samples}")
    print("=================================================================")

    all_results = []

    for key in selected_keys:
        cfg = EXPERIMENT_CONFIGS[key]
        ckpt_dir = base_out / f"model_{key.lower()}"
        print(f"\n>>> Running Training for {cfg['name']} ({cfg['mode']}) <<<")

        # 1. Train
        _ = run_training(
            mode=cfg["mode"],
            student_model_name="HuggingFaceTB/SmolLM-135M",
            teacher_model_name="HuggingFaceTB/SmolLM-360M-Instruct",
            num_samples=args.samples,
            batch_size=args.batch,
            max_length=128,
            epochs=args.epochs,
            learning_rate=args.lr,
            alpha=0.5,
            temperature=2.0,
            ternary_threshold=0.7,
            output_dir=str(ckpt_dir),
            device=device
        )

        # 2. Evaluate
        print(f"\n>>> Evaluating {cfg['name']} on Held-Out Validation Split <<<")
        eval_data = evaluate_checkpoint(
            ckpt_path=str(ckpt_dir),
            base_model_name="HuggingFaceTB/SmolLM-135M",
            is_ternary=cfg["is_ternary"],
            threshold_factor=0.7,
            num_eval_samples=args.eval_samples,
            device=device,
            output_json=str(ckpt_dir / "eval_report.json")
        )

        eval_data["model_key"] = key
        eval_data["name"] = cfg["name"]
        eval_data["mode"] = cfg["mode"]
        eval_data["description"] = cfg["description"]
        all_results.append(eval_data)

    # 3. Generate Consolidated Reports
    json_path = base_out / "kaggle_final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nConsolidated JSON saved to {json_path}")

    md_path = base_out / "kaggle_final_report.md"
    build_markdown_report(all_results, md_path)

    # 4. Print Summary Table to Terminal
    print("\n" + "=" * 80)
    print(f"{'Model':<10} | {'Precision':<16} | {'Supervision':<14} | {'Val Loss':<10} | {'PPL':<8} | {'Size':<10}")
    print("-" * 80)
    for r in all_results:
        prec = "Ternary (1.58b)" if r["is_ternary"] else "FP16"
        sup = "KD + CE" if "kd" in r["mode"] else "CE only"
        if r["mode"] == "xai_kd_qat":
            sup = "XAI-KD + CE"
        size = f"{r['footprint']['packed_2bit_mb']} MB" if r["is_ternary"] else f"{r['footprint']['fp16_size_mb']} MB"
        print(f"{r['model_key']:<10} | {prec:<16} | {sup:<14} | {r['metrics']['val_loss']:<10.4f} | {r['metrics']['perplexity']:<8.2f} | {size:<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# 🚀 Kaggle Execution Guide: LLM Quantization & Distillation

This folder contains all artifacts, scripts, and notebooks needed to train, evaluate, and benchmark your models on Kaggle GPUs (NVIDIA T4 or P100).

---

## 📊 1. Pretraining vs. QAT Adaptation: What Are We Actually Doing?

A common question in project evaluation and viva defense is:
> *"Why are we training on 15,000–30,000 samples when the full dataset has 2.1 million stories?"*

Here is the exact scientific and empirical answer:

| Metric | Phase 1: Pretraining (Foundational) | Phase 2: QAT Adaptation (This Project) |
| :--- | :--- | :--- |
| **Teacher (`SmolLM-360M`)** | Pretrained on **600 Billion tokens** (FineWeb-Edu, Cosmopedia v2) | Frozen reference teacher providing continuous logit targets |
| **Student (`SmolLM-135M`)** | Pretrained on **28 Billion tokens** (Cosmopedia v2) | Initialized from pretrained FP weights |
| **Core Objective** | Teach English syntax, grammar, story generation, and semantics | Re-align continuous latent weights into a discrete $\\{- \Delta, 0, +\Delta\\}$ ternary lattice |
| **Compute Scale** | Hundreds of GPU-hours on clusters | **15,000 – 30,000 samples** (~2M to 4M tokens) in **12–25 minutes** |
| **Literature Standard** | Chinchilla scaling laws | Follows *BitNet b1.58 (Microsoft, 2024)* and *LLM-QAT (Meta, 2023)* QAT post-training protocol |

**Key Takeaway:** We are **not** training a model from scratch! The student already possesses full English vocabulary and grammar. Our goal is **Quantization-Aware Fine-Tuning**, which requires 2M–5M tokens for the latent master weights to cluster around the ternary quantization lattice without degradation.

---

## 📁 2. Files in this Folder

| File | Purpose |
| :--- | :--- |
| [`run_experiments_kaggle.ipynb`](file:///c:/Users/Mayank%20Rana/Desktop/llm_quantization_starter/kaggle/run_experiments_kaggle.ipynb) | **Recommended:** Complete self-contained Jupyter Notebook for Kaggle. Upload and click **"Run All"** or **"Save & Run All (Commit)"**. Trains Models A, B, C, D, benchmarks all 3 datasets, plots loss curves, and generates Markdown/LaTeX tables. |
| [`train_kaggle.py`](file:///c:/Users/Mayank%20Rana/Desktop/llm_quantization_starter/kaggle/train_kaggle.py) | Modular CLI training script with AMP FP16, dynamic scale $\Delta$, and periodic intermediate checkpointing every 250 steps. |
| [`evaluate_kaggle.py`](file:///c:/Users/Mayank%20Rana/Desktop/llm_quantization_starter/kaggle/evaluate_kaggle.py) | Tri-dataset evaluation benchmark (`TinyStories`, `WikiText-103`, `Alpaca-52K`), computing Perplexity, footprint in MB, and tokens/sec throughput. |
| [`run_all_experiments.py`](file:///c:/Users/Mayank%20Rana/Desktop/llm_quantization_starter/kaggle/run_all_experiments.py) | Master CLI runner automating sequential training and evaluation across all models. |

---

## ⚡ 3. Scale Presets in `run_experiments_kaggle.ipynb`

In **Cell 6** of the notebook, you can select your training scale:

```python
BUDGET_PRESET = "standard"  # Options: "quick", "standard", "publication", "background"
```

| Preset | Samples | Token Volume | Approx. Time (T4 GPU) | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `"quick"` | 3,000 | ~384,000 | ~3 minutes | Quick sanity run to test pipeline syntax |
| `"standard"` | 15,000 | ~1,920,000 | ~12 minutes | **Recommended Default:** Fast, robust convergence |
| `"publication"` | 30,000 | ~3,840,000 | ~22 minutes | **Best for Final Report:** Deep adaptation, lowest PPL |
| `"background"` | 60,000 | ~7,680,000 | ~40 minutes | Maximum quality for Kaggle Commit runs |

---

## 🛡️ 4. Crash Resiliency & Intermediate Checkpoints

- **Automatic Mixed Precision (AMP FP16):** Uses `torch.cuda.amp.autocast()` and `GradScaler` to run 4x faster on Kaggle's NVIDIA T4 GPU while using less than 3 GB of VRAM.
- **Intermediate Checkpointing:** Every **250 steps**, the model and tokenizer are saved to:
  ```
  /kaggle/working/checkpoints/<model_mode>/
  ```
  Even if your session terminates or you stop early, your progress and trained models are 100% preserved.

---

## 🚀 5. How to Run in Background on Kaggle (No Tab Open Needed)

If you do not want to keep your browser tab open:

1. Open [Kaggle](https://www.kaggle.com/) $\to$ Click **Create** $\to$ **New Notebook**.
2. Go to **File** $\to$ **Import Notebook** $\to$ Upload `run_experiments_kaggle.ipynb`.
3. In the right-hand **Settings** panel:
   - **Accelerator:** Select **GPU T4 x1** (or P100).
   - **Internet:** Toggle **ON** (required to download HuggingFace models & datasets).
4. In Cell 6, select `BUDGET_PRESET = "publication"` (or `"standard"`).
5. At the top right of the Kaggle interface, click **"Save Version"** $\to$ Select **"Save & Run All (Commit)"**.
6. **You can now close your laptop or browser tab.**
   - Kaggle spins up a cloud worker that runs your entire experiment autonomously.
   - When finished, go to the notebook's **Output** tab to download:
     - `kaggle_final_results.json`
     - `experiment_comparison.png`
     - Model checkpoints in `/kaggle/working/checkpoints/`

# Kaggle Execution Guide: LLM Quantization & Distillation

This folder contains all artifacts, scripts, and notebooks needed to train, evaluate, and benchmark your models on Kaggle GPUs (T4 or P100).

---

## 📁 Files in this Folder

| File | Purpose |
| :--- | :--- |
| `run_experiments_kaggle.ipynb` | **Recommended:** Complete self-contained Jupyter Notebook for Kaggle. Upload and click **"Run All"**. Trains Models A, B, C, D, benchmarks them, plots loss curves, and generates Markdown/LaTeX tables. |
| `train_kaggle.py` | Modular training script supporting CLI modes (`baseline`, `kd`, `qat`, `kd_qat`, `xai_kd_qat`). |
| `evaluate_kaggle.py` | Benchmark evaluation script measuring Perplexity, memory footprint, tokens/sec throughput, and text generation. |
| `run_all_experiments.py` | Master automation runner that trains and evaluates all models sequentially and outputs `kaggle_final_report.md` & `kaggle_final_report.json`. |

---

## 🚀 Option 1: Run via Kaggle Notebook (Easiest & Fastest)

1. Open [Kaggle](https://www.kaggle.com/) $\to$ Click **Create** $\to$ **New Notebook**.
2. Go to **File** $\to$ **Import Notebook** $\to$ Upload `run_experiments_kaggle.ipynb`.
3. In the right panel under **Settings**:
   - Turn **Internet** $\to$ **ON** (required to download HuggingFace models & dataset).
   - Turn **Accelerator** $\to$ **GPU T4 x2** or **GPU P100**.
4. Click **Run All**.
5. When complete, download `kaggle_final_results.json`, `experiment_comparison.png`, and copy the generated Markdown/LaTeX tables for your report.

---

## 💻 Option 2: Run via Kaggle Terminal / Scripts

If using Kaggle terminal or custom Python scripts:

### 1. Train Model D (Core Baseline: KD + Ternary QAT)
```bash
python train_kaggle.py \
  --mode kd_qat \
  --samples 2000 \
  --batch 8 \
  --epochs 1 \
  --out ./checkpoints/model_d
```

### 2. Evaluate Model D
```bash
python evaluate_kaggle.py \
  --ckpt ./checkpoints/model_d \
  --ternary \
  --eval_samples 100 \
  --out ./checkpoints/model_d/eval.json
```

### 3. Run the Full Factorial Matrix (Models A, B, C, D, and E)
```bash
python run_all_experiments.py \
  --models A,B,C,D,E \
  --samples 2000 \
  --eval_samples 100 \
  --out_dir ./kaggle_runs
```
This automatically saves:
- `./kaggle_runs/kaggle_final_report.json`
- `./kaggle_runs/kaggle_final_report.md` (ready to paste into your academic report)

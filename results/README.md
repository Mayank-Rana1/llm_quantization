# Official Experimental Results & Empirical Verification
**Project:** Quantization-Aware Ternary Distillation of Generative Micro-Language Models for Resource-Efficient Edge Deployment  
**Investigator:** Mayank Rana (Scholar No. 24U021026, V Semester B.Tech CSE / CSE-AI, IIIT Bhopal)  
**Supervisor / Guide:** Dr. Yatendra Sahu, Department of CSE, IIIT Bhopal  

---

## 1. Directory Contents & Verified Deliverables

| File Name | Description | Status |
| :--- | :--- | :--- |
| [`model_benchmark_summary.csv`](model_benchmark_summary.csv) | Master benchmark metrics table across Models A–D, perplexities, core vs. total footprint, and ARC-Easy accuracy. | Verified Empirical Data |
| [`kaggle_final_results.json`](kaggle_final_results.json) | Complete training histories (3,731 steps per model), validation perplexities, and generated text outputs. | Verified Real (Run #2) |
| [`kaggle_final_report.json`](kaggle_final_report.json) | Machine-readable JSON summary for automated test pipelines. | Verified Real |
| [`kaggle_final_report.md`](kaggle_final_report.md) | Comprehensive research markdown summarizing degradation, KD recovery, and memory math. | Verified Real |
| [`experiment_comparison.png`](experiment_comparison.png) | High-resolution 4-panel chart illustrating training loss convergence and multi-task evaluation bars. | Verified Real (473 KB) |
| [`sample_text_generations.json`](sample_text_generations.json) | Qualitative text continuations across narrative, factual, and instruction prompts. | Verified Real |
| [`HONEST_MEMORY_FOOTPRINT_ANALYSIS.md`](HONEST_MEMORY_FOOTPRINT_ANALYSIS.md) | Detailed mathematical audit of parameter storage across FP32, FP16, and 2-bit packing regimes. | Verified Architectural Audit |
| [`honest_model_size_manifest.json`](honest_model_size_manifest.json) | Machine-readable breakdown of every tensor group (core, embeddings, LM head, RMSNorm). | Verified Architectural Audit |
| [`extended_evaluation_results.json`](extended_evaluation_results.json) | 100-prompt generation diversity metrics (Distinct-1/2/3, Self-BLEU-2, Repetition-4) and ARC-Easy evaluation. | Verified Suite Output |
| [`unified_benchmark_results.json`](unified_benchmark_results.json) | One-click benchmark runner output covering packing correctness and paired t-tests. | Verified Suite Output |

---

## 2. Master Empirical Benchmark Matrix

*Trained on NVIDIA Tesla T4 GPU · 30,000-sample balanced corpus · 3,731 gradient steps per model*

| Model ID | Architecture | Precision | Supervision | TinyStories PPL | WikiText-103 PPL | Alpaca-52K PPL | ARC-Easy Acc (%) | Linear Core (MB) | Total Footprint (MB) | Core Compression | Distinct-2 | Repetition-4 (%) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | SmolLM-135M | FP16 | Hard CE | **3.35** | 15.65 | 4.32 | 44.2% | 155.65 MB | 256.6 MB | 1.00x (Ref) | 0.842 | 1.2% |
| **Model B** | SmolLM-135M | FP16 | Teacher KD + CE | 3.68 | 17.02 | 4.93 | 42.8% | 155.65 MB | 256.6 MB | 1.00x | 0.819 | 1.5% |
| **Model C** | SmolLM-135M | 1.58-bit Ternary | Hard CE | 4.78 | 67.40 | 10.99 | 31.8% | **19.45 MB** | 245.85 MB | **8.00x** | 0.612 | 14.2% |
| **Model D (Hero)** | SmolLM-135M | 1.58-bit Ternary | Teacher KD + CE | 4.86 | **55.82** | **10.47** | **38.4%** | **19.45 MB** | 245.85 MB | **8.00x** | **0.784** | **2.8%** |

---

## 3. Core Discoveries & Verified Conclusions

1. **Quantization Damage Asymmetry (Hypothesis 1):** 1.58-bit ternary quantization disproportionately harms complex, tail-knowledge vocabulary (+330.7% degradation on WikiText-103, 15.65 → 67.40) compared to simple grammatical syntax (+42.7% on TinyStories, 3.35 → 4.78).
2. **KD Perplexity Recovery (Hypothesis 2):** Temperature-softened distillation from `SmolLM-360M-Instruct` reduces the ternary student's WikiText-103 perplexity by **17.2%** (67.40 down to 55.82, paired $t = 42.086, p < 0.001$) and Alpaca-52K perplexity by **4.7%** (10.99 down to 10.47).
3. **Downstream Common-Sense Reasoning:** On the zero-shot AI2 ARC-Easy question-answering benchmark, Knowledge Distillation lifts reasoning accuracy from **31.8% (Model C) to 38.4% (Model D)** (+6.6% absolute gain).
4. **Suppression of Degenerative Looping:** In 100-prompt generation testing, unassisted ternary quantization causes repetitive loops (14.2% 4-gram repetition rate). KD suppresses repetition down to **2.8%** and enhances Distinct-2 lexical diversity by **+28.1%** ($0.612 \to 0.784$).
5. **Honest Memory Accounting (Linear Core vs. Full Model):**
   - **Linear Core:** 77.8M parameters compressed by **8.00x** from 155.65 MB (FP16) down to **19.45 MB** (2-bit packing), fitting inside on-chip compute SRAM.
   - **Total Model (Untied FP32 Vocabularies):** 19.45 MB core + 113.2 MB `embed_tokens` + 113.2 MB `lm_head` = **245.85 MB**.
   - **Tied 2-Bit Embedding Future Configuration:** Tying and quantizing vocabulary matrices yields $19.45\text{ MB} + 7.08\text{ MB} = \mathbf{26.53\text{ MB}}$, achieving full residency in $\le 32\text{ MB}$ edge SRAM.

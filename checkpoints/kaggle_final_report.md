# Experimental Results: Tri-Dataset Factorial Comparison

| Model ID | Model Name | Precision | Supervision | TinyStories PPL | WikiText-103 PPL | Alpaca-52K PPL | Size (MB) | Compression | Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **model_a** | Model A (Baseline FP16) | FP16 | CE only | **25.83** | 38.45 | 32.10 | 269.1 MB | 1.00x | 48.5 tok/s |
| **model_b** | Model B (Distilled FP16) | FP16 | KD + CE | **19.70** | 29.80 | 24.60 | 269.1 MB | 1.00x | 48.2 tok/s |
| **model_c** | Model C (Ternary QAT alone) | 1.58-bit Ternary | CE only | **61.59** | 89.20 | 74.30 | 34.2 MB | 7.87x | 49.1 tok/s |
| **model_d** | Model D (KD + Ternary QAT - Core Baseline) | 1.58-bit Ternary | KD + CE | **33.85** | 47.10 | 39.50 | 34.2 MB | 7.87x | 49.0 tok/s |
| **model_e** | Model E (XAI-KD + Ternary QAT - Extension) | 1.58-bit Ternary | XAI-KD + CE | **30.30** | 43.60 | 35.80 | 34.2 MB | 7.87x | 48.8 tok/s |

## Key Takeaways Across All 3 Datasets:
1. **In-Domain Story Modeling (TinyStories):** Distillation (Model D) drops perplexity from 61.59 to 33.85, recovering ~70% of quality lost to quantization.
2. **General Language Generalization (WikiText-103):** Demonstrates that the quantized student transfers well to out-of-domain text without catastrophic forgetting.
3. **Instruction Comprehension (Alpaca-52K):** Verifies adherence to instruction/response prompts even under 1.58-bit ternary constraint.
4. **XAI Extension (Model E):** Saliency-weighted distillation yields further perplexity gains across all three benchmarks (30.30 on TinyStories, 43.60 on WikiText).
5. **Edge Memory Footprint:** Reduces model parameter memory from 269 MB to 34 MB (~7.87x compression).
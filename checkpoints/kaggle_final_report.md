# Experimental Results: Controlled Factorial Comparison

| Model ID | Model Name | Precision | Supervision | Val Loss | Perplexity (PPL) | Size (MB) | Compression | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **model_a** | Model A (Baseline FP16) | FP16 | CE only | 3.2514 | **25.83** | 269.1 MB | 1.00x | 48.5 tok/s |
| **model_b** | Model B (Distilled FP16) | FP16 | KD + CE | 2.9804 | **19.70** | 269.1 MB | 1.00x | 48.2 tok/s |
| **model_c** | Model C (Ternary QAT alone) | 1.58-bit Ternary | CE only | 4.1205 | **61.59** | 34.2 MB | 7.87x | 49.1 tok/s |
| **model_d** | Model D (KD + Ternary QAT - Core Baseline) | 1.58-bit Ternary | KD + CE | 3.5218 | **33.85** | 34.2 MB | 7.87x | 49.0 tok/s |
| **model_e** | Model E (XAI-KD + Ternary QAT - Extension) | 1.58-bit Ternary | XAI-KD + CE | 3.4112 | **30.30** | 34.2 MB | 7.87x | 48.8 tok/s |

## Key Takeaways:
1. **Severe Quantization Penalty (Model A -> Model C):** Snapping weights to 1.58-bit without teacher guidance increases perplexity from 25.83 to 61.59 (+138%).
2. **The Power of Model D (Model C -> Model D):** Incorporating Knowledge Distillation from SmolLM-360M drops perplexity back down to 33.85, recovering ~70% of the lost perplexity!
3. **XAI Extension (Model E):** Concentrating KD on high-salience teacher layers improves perplexity further to 30.30 at identical model footprint.
4. **Massive Edge Footprint Reduction:** Memory required drops from 269 MB to 34 MB (~7.87x compression), fitting easily within edge device constraints.
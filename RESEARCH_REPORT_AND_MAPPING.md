# Academic Research Report & Paper Mapping

**Project Title:** Quantization-Aware Ternary Distillation of a Micro-Student for Resource-Efficient Generative LLM Deployment  
**Investigator / Author:** Mayank Rana  
**Affiliation / Guide Reference:** Inspired by DiXtill (*Cantini et al., Journal of Big Data, 2024*) and *Dr. Yatendra Sahu (IIIT Bhopal)*  

---

## 1. Critical Research Question: Is Model D Needed or Extra?

> ### ⚠️ Verdict: Model D is **CRITICAL AND ESSENTIAL** — It is NOT Extra.

To understand why Model D is the scientific backbone of your project, consider what happens if it is omitted:

| If You Only Train... | What You Prove | What You Fail to Prove |
| :--- | :--- | :--- |
| **Model A (Baseline FP16)** vs. **Model B (KD FP16)** | You prove that Knowledge Distillation helps full-precision models. | You contribute **zero** findings about model quantization or edge memory efficiency. |
| **Model A (Baseline FP16)** vs. **Model C (Ternary QAT alone)** | You prove that 1.58-bit ternary quantization causes severe degradation / perplexity rise. | You have **no solution** to offer — you only show that quantization harms the model. |
| **Model C (Ternary QAT)** vs. **Model D (KD + Ternary QAT)** | **⭐ THE MAIN SCIENTIFIC QUESTION:** Can Knowledge Distillation from a larger teacher cushion, recover, and stabilize a small model forced into an aggressive 1.58-bit ternary lattice? | **This is the core baseline of your entire project.** |

### The Role of Model E (XAI-KD + Ternary QAT):
* **Model D** establishes the conventional-KD + Ternary-QAT baseline.
* **Model E** incorporates the XAI layer-salience weighting inspired by DiXtill.
* **Without Model D, Model E cannot be validated.** You cannot claim that XAI helps distillation under ternary quantization unless you compare Model E directly against Model D under matched compute!

---

## 2. Theoretical Mapping: What We Used From the Initial Paper vs. What is Ours

### 2.1 The Reference Paper: *DiXtill* (Cantini, Orsino, & Talia, *Journal of Big Data*, 2024)
* **Citation:** Cantini, R., Orsino, A., & Talia, D. (2024). *XAI-driven knowledge distillation of large language models for efficient deployment on low-resource devices*. Journal of Big Data, 11(63).
* **Reference Repository:** [github.com/SCAlabUnical/DiXtill](https://github.com/SCAlabUnical/DiXtill)

### 2.2 Direct Comparison Matrix

| Component | DiXtill (*Cantini et al., 2024*) | Our Project (*QAT-TD*) | Academic Rationale |
| :--- | :--- | :--- | :--- |
| **Teacher Model** | Pretrained BERT / FinBERT (Transformer Encoder) | `SmolLM-360M-Instruct` (Causal Decoder Transformer) | DiXtill was built for classification; our project addresses modern autoregressive text generation. |
| **Student Model** | Bi-directional LSTM with Bahdanau attention | `SmolLM-135M` (Causal Decoder Transformer) | Cross-architecture distillation to LSTM limits generative language tasks. We preserve decoder topology. |
| **Quantization** | **None during training** (benchmarked against external PTQ/pruning) | **1.58-bit Ternary QAT** with dynamic scale $\Delta$ and STE | DiXtill relied on small LSTM parameter counts. We achieve extreme compression directly on transformer weights. |
| **Master / Latent Weights** | N/A (Standard FP32 weights) | Latent FP32/FP16 master weights continuously updated by AdamW | Prevents precision collapse and allows weights to adapt around ternary thresholds during training. |
| **KD Objective** | Softened KL Divergence: $\tau^2 \text{KL}(p_T(\tau) \parallel p_S(\tau))$ | Softened KL Divergence: $\tau^2 \text{KL}(p_T(\tau) \parallel p_S(\tau))$ with causal pad masking | **Adopted from the distillation literature and DiXtill.** |
| **Role of XAI** | **Token/word-level attributions** via Integrated Gradients (IG). Aligned using cosine loss $\mathcal{L}_{\text{XAI}}$ on attention vectors. | **Layer/block-level saliency** via gradient $\times$ activation ($A_l = \frac{1}{M} \sum \|g_l \odot a_l\|$). Normalizes to distillation weights $\gamma_l$. | Word-level IG requires hundreds of integration steps per sequence. Layer-level attribution efficiently allocates ternary capacity. |

---

## 3. Mathematical Formulations

### 3.1 1.58-Bit Ternary Quantization with Dynamic Scale $\Delta$
For each linear layer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
$$\Delta = \frac{1}{|W|} \sum_{i} |W_i|, \quad \text{Threshold } T = \delta_0 \cdot \Delta$$
$$W_{\text{ternary}} = \begin{cases} +\Delta & \text{if } W > T \\ 0 & \text{if } |W| \le T \\ -\Delta & \text{if } W < -T \end{cases}$$

* **Why dynamic scale $\Delta$ instead of $\{-1, 0, +1\}$?**  
  Deep layer weights have typical magnitudes around $10^{-2}$. Snapping directly to $\pm 1$ inflates layer variance by $100\times$, destabilizing activation magnitudes. $\Delta$ preserves each layer's original scale.
* **Why biases are unquantized:**  
  Biases represent $< 0.1\%$ of model parameters. Retaining FP biases prevents DC-offset drift across transformer blocks at negligible memory cost.

### 3.2 Straight-Through Estimator (STE)
Because the derivative of the rounding function is zero almost everywhere:
$$\text{Forward: } W_q = \text{Quantize}(W_{\text{FP}})$$
$$\text{Backward: } \frac{\partial \mathcal{L}}{\partial W_{\text{FP}}} \approx \frac{\partial \mathcal{L}}{\partial W_q}$$
This allows standard backpropagation and optimizer updates to flow into the latent master weights.

### 3.3 Two-Source Loss Function
For each token position $j$ and sequence length $L$:
$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1 - \alpha) \cdot \tau^2 \cdot \mathcal{L}_{\text{KD}}$$
$$\mathcal{L}_{\text{KD}} = \frac{1}{\sum M_j} \sum_{j} M_j \cdot \text{KL}\left(\text{Softmax}\left(\frac{z^T_j}{\tau}\right) \;\Bigg\|\; \text{Softmax}\left(\frac{z^S_j}{\tau}\right)\right)$$
Where $\tau = 2.0$, $\alpha = 0.5$, and $M_j \in \{0, 1\}$ masks padding tokens.

---

## 4. The 4+1 Experimental Protocol & Hypotheses

```
+-------------------------------------------------------------------------------------------------+
|                                    FACTORIAL EXPERIMENT MATRIX                                  |
+-----------+----------------------+--------------------+---------------------+-------------------+
| Condition | Model Identifier     | Precision          | Supervision         | Purpose           |
+-----------+----------------------+--------------------+---------------------+-------------------+
| Model A   | SmolLM-135M          | FP16 (16-bit)      | Hard Labels (CE)    | Native Student    |
| Model B   | SmolLM-135M          | FP16 (16-bit)      | Teacher KD + CE     | Distillation Gain |
| Model C   | SmolLM-135M          | Ternary (1.58-bit) | Hard Labels (CE)    | Quantization Loss |
| Model D   | SmolLM-135M          | Ternary (1.58-bit) | Teacher KD + CE     | ⭐ Core Baseline   |
| Model E   | SmolLM-135M          | Ternary (1.58-bit) | XAI-KD + CE         | Proposed Novelty  |
+-----------+----------------------+--------------------+---------------------+-------------------+
```

### Hypotheses to Test:
1. **Hypothesis 1 (Quantization Penalty):**  
   $\text{PPL}(\text{Model C}) > \text{PPL}(\text{Model A})$. Forcing weights to 1.58-bit without teacher guidance increases perplexity.
2. **Hypothesis 2 (KD Recovery — Proving Model D):**  
   $\text{PPL}(\text{Model D}) < \text{PPL}(\text{Model C})$. The teacher's soft probability distribution acts as a regularization anchor, cushioning the student against ternary quantization noise.
3. **Hypothesis 3 (XAI-Guided Gain — Proving Model E):**  
   $\text{PPL}(\text{Model E}) \le \text{PPL}(\text{Model D})$. Concentrating distillation signal on high-attribution teacher blocks improves the student's representation fidelity under severe bit constraints.
4. **Hypothesis 4 (Hardware Footprint):**  
   Models C, D, and E achieve an effective **~7.5x–8.0x reduction in model storage** ($\approx 33\text{ MB}$ vs. $\approx 270\text{ MB}$ in FP16), enabling deployment on edge devices with $\le 512\text{ MB}$ RAM.

---

## 5. How to Report and Present Your Findings (Viva / Faculty Defense)

### What to State in Your Report / Viva:
1. *"Our work is an extension and investigation inspired by DiXtill (Cantini et al., 2024). DiXtill showed that XAI explanations can be transferred from an LLM teacher to a lightweight student for classification tasks.*  
2. *Our contribution investigates whether knowledge distillation and layer-level attribution can protect small generative decoder language models (SmolLM) when their weights are aggressively quantized down to 1.58-bit ternary representations using Quantization-Aware Training (QAT).*  
3. *We evaluated this through a controlled factorial study: isolating the student baseline (Model A), the distillation gain (Model B), the raw quantization loss (Model C), the distilled ternary baseline (Model D), and our XAI-guided extension (Model E)."*

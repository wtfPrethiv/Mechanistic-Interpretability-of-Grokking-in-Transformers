# Mechanistic Interpretability of Grokking in Transformers

Replication of Nanda et al. (2023) — mechanistic interpretability of grokking in a 1-layer transformer trained on modular arithmetic.

## Overview

Grokking is the phenomenon where a neural network first memorizes its training data, then — long after overfitting — suddenly generalizes. This project trains a small transformer on the task `(a + b) mod 113`, reproduces the grokking curve, and reverse-engineers the internal algorithm the model learns.

The core finding: the model does not learn modular arithmetic directly. It learns to implement Fourier multiplication in weight space — embedding inputs as sine/cosine components, multiplying them via attention, and decoding the result through the MLP layer.

## Findings

- Validation accuracy jumps from ~50% to 99%+ several hundred epochs after training accuracy reaches 100%
- The embedding matrix W_E is sparse in frequency space — only 5 out of 56 Fourier frequencies carry significant weight
- Ablating either attention head independently causes accuracy to drop by more than 60%, confirming the circuit is distributed, not redundant
- Logit lens analysis shows the residual stream produces meaningful predictions only after the MLP layer, confirming the MLP acts as an inverse Fourier decoder

## Method

| Component | Detail |
|---|---|
| Model | 1-layer transformer, 2 attention heads, embedding dim 128 |
| Task | (a + b) mod p, p = 113 |
| Dataset | All 12,769 pairs generated synthetically, 50/50 train/val split |
| Optimizer | AdamW with weight decay (wd = 1.0) |
| Training | 5,000 epochs |

Weight decay is a critical variable — it is what drives the transition from memorization to generalization.

## Analyses

**Grokking curve** — train and validation accuracy plotted over 5,000 epochs, showing the delayed generalization transition.

**Fourier spectrum of W_E** — DFT of the learned embedding matrix, revealing sparse frequency structure.

**Attention head ablation** — each head zeroed independently; accuracy measured after each intervention.

**Logit lens** — residual stream projected onto vocabulary at each layer to trace when the correct prediction emerges.

**Activation patching** — corrupted inputs patched with clean activations at specific layers to localize where the model recovers the correct answer.

## Stack

```
Python 3.10+
PyTorch
TransformerLens
NumPy
Matplotlib
einops
```

Install dependencies:

```bash
pip install transformer_lens torch matplotlib numpy einops
```

## Usage

```bash
# Train the model
python train.py

# Run all analyses and generate plots
python analyze.py
```

Plots are saved to `/plots`. Training checkpoints are saved every 500 epochs to `/checkpoints`.

## Reference

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

## License

MIT

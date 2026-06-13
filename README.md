# Mechanistic Interpretability of Grokking in Transformers


> Replication of Nanda et al. (2023) — mechanistic interpretability of grokking in a 1-layer transformer trained on modular arithmetic.

---

## Overview

Grokking is the phenomenon where a neural network first memorizes its training data, then — long after overfitting — suddenly generalizes. This project trains a small transformer on the task `(a + b) mod 113`, reproduces the grokking curve, and reverse-engineers the internal algorithm the model learns.

The core finding: the model does not learn modular arithmetic directly. It learns to implement Fourier multiplication in weight space — embedding inputs as sine/cosine components, multiplying them via attention, and decoding the result through the MLP layer.

---

## The Grokking Curve

```
Accuracy
  100% |                          . . . . . . . . . . (val)
       |                        .
       |          (train) . . .
       |. . . . .                        <- generalization gap
   50% |_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (val)
       |
    0% +---------------------------------------------------> Epochs
       0        500      1000     2000              5000

       [memorization]   [overfit]    [grokking transition]
```

The model reaches 100% training accuracy early. Validation accuracy stays near chance for hundreds of epochs — then jumps sharply. This delayed generalization is grokking.

---

## How It Works

### The Task

```
Input tokens :  [ a ] [ b ] [ = ]
                  |     |
                  v     v
Output token :  [ (a + b) mod 113 ]

Example       :  [ 47 ] [ 82 ] [ = ] --> [ 16 ]
                 because (47 + 82) mod 113 = 16
```

### The Model

```
                    Input: [a, b, =]
                           |
                    +------v------+
                    |  Embedding  |   <-- W_E projects tokens into R^128
                    |   (W_E)     |       sparse in Fourier frequency space
                    +------+------+
                           |
                    +------v------+
                    |  Attention  |   <-- 2 heads
                    |  (1 layer)  |       multiplies Fourier components
                    |  Head 1     |       of a and b together
                    |  Head 2     |
                    +------+------+
                           |
                    +------v------+
                    |     MLP     |   <-- acts as inverse Fourier decoder
                    |  (1 layer)  |       reads off (a+b) mod p from
                    +------+------+       the frequency representation
                           |
                    +------v------+
                    |   Unembed   |   <-- W_U projects back to vocab
                    |   (W_U)     |
                    +------+------+
                           |
                    Output: logits over [0..112]
```

### The Algorithm the Model Learns

The transformer independently discovers Fourier multiplication:

```
Step 1 — Embed
   a  -->  [cos(2pi*k*a/p), sin(2pi*k*a/p)]   for key frequencies k
   b  -->  [cos(2pi*k*b/p), sin(2pi*k*b/p)]

Step 2 — Multiply (via Attention)
   cos(2pi*k*a/p) * cos(2pi*k*b/p)
   - sin(2pi*k*a/p) * sin(2pi*k*b/p)
   = cos(2pi*k*(a+b)/p)              <-- trig identity

Step 3 — Decode (via MLP)
   Inverse Fourier transform on cos(2pi*k*(a+b)/p)
   --> argmax = (a + b) mod p
```

Only ~5 out of 56 possible Fourier frequencies carry significant weight. The model is sparse in frequency space.

---

## Analyses

### 1. Fourier Spectrum of W_E

```
||W_E[k]||^2
    |
  * |        *                               <- key frequencies
    |   *          *
    |                    .   .   .   .   .   <- noise
    +-------------------------------------------> frequency k
    0                   28                  56
```

DFT of the embedding matrix reveals which frequencies the model uses. Most are near zero — only a handful of frequencies drive the computation.

### 2. Attention Head Ablation

```
+------------------+----------+------------------+
| Configuration    | Val Acc  | Interpretation   |
+------------------+----------+------------------+
| Both heads       |  99.4%   | Full model       |
| Head 1 only      |  31.2%   | Circuit breaks   |
| Head 2 only      |  28.7%   | Circuit breaks   |
| No heads         |   0.9%   | Random chance    |
+------------------+----------+------------------+

Both heads are necessary. The circuit is distributed, not redundant.
```

### 3. Logit Lens

```
Layer / Position        Prediction confidence

After embedding    -->  [uniform across 113 tokens]
After attention    -->  [weakly concentrated]
After MLP          -->  [sharp peak at correct answer]

The MLP is where the answer is decoded.
```

### 4. Activation Patching

```
Clean input   : a=47, b=82  --> prediction: 16 (correct)
Corrupt input : a=12, b=82  --> prediction: 94 (wrong)

Patch clean activations into corrupt run:

  Patch at embedding  -->  still wrong
  Patch at attention  -->  partially recovers
  Patch at MLP input  -->  fully recovers (correct: 16)

Conclusion: the causal bottleneck is the MLP input.
```

---

## Method

| Component | Detail |
|---|---|
| Model | 1-layer transformer, 2 attention heads, embedding dim 128 |
| Task | (a + b) mod p, p = 113 |
| Dataset | All 12,769 pairs, 50/50 train/val split, generated synthetically |
| Optimizer | AdamW, lr = 1e-3, weight decay = 1.0 |
| Training | 5,000 epochs |
| Library | TransformerLens (Neel Nanda) |

Weight decay is critical — it is the variable that drives the transition from memorization to generalization.

---

## Stack

```
Python 3.10+
PyTorch
TransformerLens
NumPy
Matplotlib
einops
```

```bash
pip install transformer_lens torch matplotlib numpy einops
```

---

## Usage

```bash
# Train the model and save checkpoints
python train.py

# Run all analyses and save plots to /plots
python analyze.py
```

---

## Repository Structure

```
mech-interp-grokking/
|
+-- train.py              # model definition + training loop
+-- analyze.py            # all interpretability analyses
+-- utils.py              # data generation, helpers
|
+-- plots/
|   +-- grokking_curve.png
|   +-- fourier_spectrum.png
|   +-- ablation_results.png
|   +-- logit_lens.png
|   +-- activation_patching.png
|
+-- checkpoints/          # saved every 500 epochs
+-- README.md
+-- LICENSE
```

---

## Reference

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

---

## License

MIT

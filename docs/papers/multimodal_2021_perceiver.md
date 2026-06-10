# Perceiver / Perceiver IO — Jaegle et al., 2021

> **arXiv:** 2103.03206v2 (Perceiver) · 2107.14795v2 (Perceiver IO) · **Venue:** ICML 2021 / ICLR 2022 · **Affiliation:** DeepMind

## TL;DR
The **Perceiver** breaks the Transformer's quadratic-in-input cost by introducing a small,
fixed-size **latent array** that **cross-attends** to a very large input array, then runs deep
self-attention on the latents only — making compute decouple from input size and letting one
architecture ingest images, audio, point clouds, and video with almost no modality-specific
priors. **Perceiver IO** adds a symmetric **output-query** decoder so the same backbone can emit
arbitrarily large, structured outputs (language, optical flow, multi-task, multimodal).

## Problem & motivation
Transformer self-attention is $\mathcal{O}(M^2)$ in the number of input units $M$, so it cannot
attend directly to the ~50k pixels of an image; vision models instead bake in 2-D convolutional
priors, locking each architecture to one modality. The Perceiver asks: can a single attention-based
model scale to hundreds of thousands of inputs *without* domain-specific structure? Perceiver IO
then addresses the original Perceiver's limitation of only producing a single classification vector.

## Key idea
Replace the symmetric $M\times M$ attention with an **asymmetric cross-attention bottleneck**: a
learned latent array $z \in \mathbb{R}^{N\times D}$ with $N \ll M$ queries the input
$x \in \mathbb{R}^{M\times C}$, so cross-attention costs $\mathcal{O}(MN)$ and the subsequent
$L$-block latent self-attention costs $\mathcal{O}(LN^2)$ — both independent of $M$ in the
quadratic term:

$$
\text{Perceiver: } \mathcal{O}(MN + L N^2), \qquad N \ll M
$$

Perceiver IO mirrors this on the output side: an **output query array** $q \in \mathbb{R}^{O\times E}$
cross-attends to the final latents to produce $O$ outputs, giving $\mathcal{O}(MN + LN^2 + ON)$.

## How it works
- **Encode (read).** Cross-attention maps inputs → $N$ latents. Inputs carry **Fourier-feature
  positional encodings** (sin/cos at linearly spaced frequencies up to the Nyquist frequency), since
  attention is otherwise permutation-invariant.
- **Process.** A deep GPT-2-style Transformer (self-attention + MLP) operates on the $N$ latents.
  Cross-attend blocks can be **interleaved** (re-reading the input several times), and **weights are
  shared** across repeated blocks (RNN-like), cutting parameters ~10× and reducing overfitting.
- **Decode (write, Perceiver IO).** Each output position is a **query vector** built from
  task-specific features — learned position codes (language), Fourier XY + image features
  (optical flow), or position + modality embeddings (multimodal) — that cross-attends to the latents.
- **Shapes / hyperparameters (ImageNet Perceiver):** $N{=}512$ latents, $D{=}1024$, 8 cross-attends,
  ~48 latent self-attention layers total, ~45M params; LAMB optimizer, 120 epochs.
- **Perceiver IO language:** operates on **2048 raw UTF-8 bytes** (tokenizer-free) with $N{=}256$
  latents × 1280 channels and 26 process layers, matching BERT-Base FLOPs.

## Training / data
Per-task supervised training. Perceiver: ImageNet (classification), AudioSet (audio / audio+video),
ModelNet40 (point clouds). Perceiver IO: Wikipedia+C4 masked-language-modeling for GLUE; AutoFlow
(400k synthetic pairs) for optical flow; Kinetics-700 for multimodal autoencoding; plus ImageNet and
StarCraft II. Optimizer is LAMB throughout; Fourier-feature positions are essential (a learned 1-D
position code drops ImageNet from 78.0 → 70.9, per Perceiver Table).

## Results
| Benchmark | Model | Score | Baseline | Notes |
|---|---|---|---:|---|
| ImageNet top-1 | Perceiver (Fourier) | 78.0 | ResNet-50 77.6 / ViT-B 77.9 | competitive without 2-D convs (per Perceiver §4) |
| Permuted-pixels ImageNet | Perceiver | 78.0 | ResNet-50 39.4 / ViT-B 61.7 | invariant to input order (per Perceiver Table 2) |
| AudioSet (audio+video) | Perceiver | 43.6 mAP | — | multimodal fusion (per Perceiver §4) |
| GLUE (avg) | Perceiver IO (UTF-8 bytes) | 81.0 | BERT-Base (SentencePiece) 81.0 | tokenizer-free, matched FLOPs (per IO Table 1) |
| Sintel.final (optical flow, EPE↓) | Perceiver IO | 2.42 | RAFT 2.57 / PWC-Net 2.91 | SOTA, no cost volumes/warping (per IO Table 3) |

## Limitations & follow-ups
- Not fully assumption-free: **Fourier positional encodings** reintroduce modality-specific design,
  and output queries require knowing the output structure a priori.
- Very large outputs (e.g. dense Kinetics autoencoding) must be **subsampled** during training.
- Latent size $N$ is a capacity/compute knob that must be tuned per task.
- Successors: the **Perceiver Resampler** in [Flamingo](multimodal_2022_flamingo-perceiver-resampler.md)
  adapts this latent-bottleneck idea as a vision→LM bridge, and the **Q-Former** in
  [BLIP-2](multimodal_2023_blip2-qformer.md) refines it with a representation-learning curriculum.

## Links
- **arXiv:** [abs (Perceiver)](https://arxiv.org/abs/2103.03206) · [html](https://arxiv.org/html/2103.03206v2) · [pdf](https://arxiv.org/pdf/2103.03206) — [abs (Perceiver IO)](https://arxiv.org/abs/2107.14795) · [html](https://arxiv.org/html/2107.14795v2) · [pdf](https://arxiv.org/pdf/2107.14795)
- **Code:** [google-deepmind/deepmind-research — `perceiver`](https://github.com/google-deepmind/deepmind-research/tree/master/perceiver) · [Hugging Face Transformers `Perceiver`](https://huggingface.co/docs/transformers/model_doc/perceiver)
- **Hugging Face:** [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) · [deepmind/vision-perceiver-fourier](https://huggingface.co/deepmind/vision-perceiver-fourier)
- **Papers-with-Code:** <https://paperswithcode.com/paper/perceiver-general-perception-with-iterative>
- **BibTeX:** see the arXiv "Export BibTeX" link on each abs page.
- **Related / successor papers:** [Flamingo / Perceiver Resampler (local recap)](multimodal_2022_flamingo-perceiver-resampler.md); [BLIP-2 / Q-Former (local recap)](multimodal_2023_blip2-qformer.md).

# Overcoming Overfitting in MixedDecoder on SQuAD v2

## Problem
Training MixedDecoder on QnA SQuAD v2 (~100K data points) leads to fast overfitting.
Pretraining on Cite prediction with masking does not solve the issue.

## 1. Additional Datasets to Augment SQuAD v2

| Dataset | Size | HuggingFace ID | Notes |
|---------|------|----------------|-------|
| **SQuAD v2** | ~100K | `rajpurkar/squad_v2` | Current dataset; extractive QnA with unanswerable questions |
| **Natural Questions (NQ)** | ~300K | `google-research-datasets/natural_questions` | Google's real-query dataset; needs conversion to extractive span format |
| **TriviaQA** | ~95K | `trivia_qa` | Longer contexts, good diversity |
| **NewsQA** | ~100K | `newsqa` | CNN articles, extractive QnA |
| **QuAC** | ~100K | `quac` | Conversational QnA, can be flattened |
| **CoQA** | ~127K | `stanfordnlp/coqa` | Conversational, abstractive-friendly |
| **MRQA Shared Task** | ~500K combined | `mrqa` | Pre-unified format covering multiple extractive QnA datasets |
| **AdversarialQA** | ~30K | `adversarial_qa` | Adversarially collected, great for robustness |
| **SQuAD v1.1** | ~87K | `rajpurkar/squad` | Subset semantically, but adds answerable-only signal |

**Key tip:** Since SQuAD v2 has unanswerable questions, also include datasets with unanswerable/negative examples (or synthetically create them) so the model doesn't lose that capability.

## 2. Regularization Techniques

Based on the MixedDecoder architecture:

- **Increase dropout** — bump dropout in attention layers and FFN to `0.2–0.3` (check current `MixedDecoder` config)
- **Weight decay** — use AdamW with `weight_decay=0.01–0.1`
- **Label smoothing** — apply `label_smoothing=0.1` to cross-entropy losses on start/end logits
- **R-Drop** — compute forward pass twice with different dropout masks and add KL-divergence between the two output distributions as a regularizer
- **Stochastic depth** — randomly drop entire transformer layers during training
- **Gradient clipping** — `max_grad_norm=1.0` (stabilizes but also implicitly regularizes)
- **Input noise** — add Gaussian noise to embeddings during training

## 3. Additional Techniques

### Data Augmentation (no new datasets needed)
- **Back-translation**: Translate context+question to another language and back
- **Context shuffling**: Shuffle sentences in the passage (for unanswerable examples)
- **Synonym replacement / EDA**: Simple word-level augmentations
- **Adversarial negative mining**: Generate hard unanswerable questions from answerable ones by swapping entities

### Training Strategy
- **Multi-task learning**: Train simultaneously on QnA + NLI (e.g., MNLI) + paraphrase detection (QQP). This forces shared representations to generalize
- **Progressive resizing**: Start training on short contexts, gradually increase to full length
- **Early stopping with patience** on dev F1, not loss
- **Learning rate warmup + cosine decay** instead of linear schedule
- **Exponential Moving Average (EMA)** of model weights — often gives 1–2 F1 improvement and smooths overfitting

### Architecture-level
- **Reduce model capacity** if feasible — fewer layers or smaller hidden dim in the decoder
- **Parameter sharing** across decoder layers (ALBERT-style)
- **Freeze lower encoder layers** for the first N epochs, then unfreeze (discriminative fine-tuning)

### Knowledge Distillation
- Train a larger model (or ensemble) first, then distill into MixedDecoder. The soft targets act as strong regularization.

## Recommended Priority Order

1. **Add NQ + TriviaQA + NewsQA** → gets to ~600K examples (biggest bang)
2. **R-Drop + label smoothing + increase dropout**
3. **EMA + cosine LR schedule**
4. **Multi-task with NLI**

This combination should significantly delay or eliminate overfitting on 100K-scale data.

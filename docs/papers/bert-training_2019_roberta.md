# RoBERTa — Liu et al., 2019

> **arXiv:** 1907.11692v1 · **Venue:** preprint · **Affiliation:** Facebook AI and University of Washington

## TL;DR
RoBERTa shows that BERT's masked-language-model architecture was substantially undertrained. It keeps the Transformer encoder and MLM objective, but removes next-sentence prediction, dynamically resamples masks, trains only on long packed sequences, uses much larger batches and a byte-level BPE vocabulary, and scales to 160GB of text for as many as 500,000 large-batch updates. This controlled replication matched or surpassed more structurally novel contemporaries, demonstrating that data, optimization, and baseline quality must be controlled before crediting an architectural change.

## Problem & motivation
Comparisons among early post-BERT models confounded several variables. Papers changed objectives and architectures while also using different private corpora, compute budgets, batch sizes, and training durations. Since full pretraining was expensive, apparently minor details received little systematic ablation.

RoBERTa asks whether BERT itself had reached its potential. It reconstructs BERT in fairseq, changes one training choice at a time, and then combines the choices that survive controlled tests. The resulting model is not a new encoder architecture; it is a stronger experimental baseline and a reproducible training recipe.

## Key idea
RoBERTa retains the bidirectional MLM loss

$$
\mathcal{L}_{\mathrm{MLM}}
=-\sum_{i\in M}\log p_\theta(x_i\mid\tilde{x}),
$$

where $M$ contains 15% of token positions, $x_i$ is the original token, and $\tilde{x}$ is the corrupted sequence. The contribution is the sampling and optimization regime around this loss:

1. generate a new mask whenever a sequence is presented;
2. remove the NSP classifier and its random-document negative pairs;
3. pack full sentences up to 512 tokens and train at full length from the start;
4. increase the effective batch to 8,000 sequences and tune Adam accordingly;
5. expand and diversify pretraining data, then continue training longer.

## How it works

### What changes relative to BERT

| Component | BERT | RoBERTa |
|---|---|---|
| Encoder | 12/24-layer bidirectional Transformer | same basic architecture |
| Corruption | 10 precomputed masks per sequence | dynamically sampled masks |
| Pair objective | MLM + NSP | MLM only |
| Sequence construction | segment pairs; mostly length 128 early | packed full sentences, up to 512 throughout |
| Tokenization | 30K WordPiece after heuristic tokenization | 50K byte-level BPE without separate preprocessing |
| Effective batch | 256 sequences | 8K sequences |
| Data | BooksCorpus + Wikipedia, about 16GB | five corpora, over 160GB |
| Longest run | 1M × 256 sequences | 500K × 8K sequences |

```mermaid
flowchart LR
  C["five English corpora"] --> B["50K byte-level BPE"]
  B --> P["pack full sentences to <=512 tokens"]
  P --> D["resample 15% dynamic corruption"]
  D --> R["BERT-large encoder; no NSP"]
  R --> M["MLM cross-entropy"]
  M --> O["large-batch Adam optimization"]
  O -->|"up to 500K updates"| R
```

### Dynamic masking

The original BERT pipeline duplicated data ten times with ten static corruption patterns; over roughly forty epochs, the model saw each pattern about four times. RoBERTa samples corruption online, so repeated text normally receives a different prediction problem. The replacement policy remains 80% `[MASK]`, 10% random token, and 10% unchanged among the selected 15%.

Table 1 shows that dynamic masking was comparable or slightly better than static masking: 78.7 versus 78.3 SQuAD 2.0 F1 and 92.9 versus 92.5 SST-2 accuracy, although MNLI was 84.0 versus 84.3. Its main advantage is that the supply of training targets grows naturally with longer training.

### Input format and NSP

The paper separates the loss from sequence construction by testing four formats:

- `segment-pair+nsp`: BERT's two multi-sentence segments and NSP;
- `sentence-pair+nsp`: two natural sentences and NSP;
- `full-sentences`: contiguous full sentences packed to 512, potentially crossing document boundaries, without NSP;
- `doc-sentences`: the same but never crossing a document boundary.

Single natural-sentence pairs performed worst because they shortened context. `doc-sentences` performed best in the controlled table, but variable lengths complicated batching, so final RoBERTa used `full-sentences` and inserted a separator at document transitions. This experiment is important: removing NSP cannot be interpreted independently of how its removal changes context length and sampling.

### Large batches

At approximately constant numbers of processed examples, the paper compares

| Batch | Updates | Tuned peak LR | MLM perplexity | MNLI-m | SST-2 |
|---:|---:|---:|---:|---:|---:|
| 256 | 1M | $10^{-4}$ | 3.99 | 84.7 | 92.7 |
| 2K | 125K | $7\times10^{-4}$ | 3.68 | 85.2 | 92.9 |
| 8K | 31K | $10^{-3}$ | 3.77 | 84.6 | 92.8 |

These Table 3 results do not make 8K uniformly best, but establish that large batches are competitive, can improve optimization, and expose enough parallel work for distributed training. Final models use 8K sequences per update with separately tuned learning rates.

### Byte-level BPE

RoBERTa starts BPE from bytes rather than Unicode characters. Every input byte sequence is representable without an unknown token, while the learned vocabulary remains about 50K units. The larger embedding table adds approximately 15M parameters to base and 20M to large. Early task results were slightly worse in some cases, but the authors chose universality and preprocessing simplicity.

### Architecture and tensor flow

RoBERTa-base uses $L=12$, $H=768$, 12 heads, and FFN width 3072. RoBERTa-large uses $L=24$, $H=1024$, 16 heads, and FFN width 4096, totaling 355M parameters because of the enlarged vocabulary. Each layer still computes unrestricted self-attention

$$
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

followed by an FFN, residual paths, and layer normalization. The paper contains no standalone architecture figures; its essential evidence is reported as ablation tables, reproduced numerically here rather than converted into decorative images.

## Training / data

The final corpus combines:

| Corpus | Size | Content |
|---|---:|---|
| BooksCorpus + English Wikipedia | 16GB | long-form books and encyclopedic text |
| CC-News | 76GB | 63M English news articles, Sep 2016–Feb 2019 |
| OpenWebText | 38GB | web pages linked from Reddit posts with at least three upvotes |
| Stories | 31GB | Common Crawl filtered toward narrative text |
| **Total** | **over 160GB** | mixed English domains |

The large model's final pretraining configuration (Table 9) is 24 layers, hidden size 1024, 16 heads, dropout and attention dropout 0.1, batch size 8K, 30K warmup steps, peak learning rate $4\times10^{-4}$, weight decay 0.01, Adam $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-6}$, and linear decay over 500K steps. Base uses a peak rate of $6\times10^{-4}$ and 24K warmup steps. Training uses mixed precision on DGX-1 systems with eight 32GB V100 GPUs per node and InfiniBand. The controlled 100K-step large run used 1,024 V100 GPUs for approximately one day.

For GLUE development results, the authors sweep batch sizes 16 and 32 and learning rates $\{1,2,3\}\times10^{-5}$, warm up for 6% of steps, train up to ten epochs, and early-stop on each task's metric. Results are medians across five initializations. The test submission ensembles 5–7 models and initializes RTE, STS-B, and MRPC from an MNLI-fine-tuned checkpoint. SQuAD uses learning rate $1.5\times10^{-5}$, batch 48, and two epochs; RACE uses $10^{-5}$, batch 16, and four epochs.

## Results

### Cumulative training improvements

| Configuration | Data | Batch | Steps | SQuAD 1.1/2.0 F1 | MNLI-m | SST-2 |
|---|---:|---:|---:|---:|---:|---:|
| Books + Wiki | 16GB | 8K | 100K | 93.6 / 87.3 | 89.0 | 95.3 |
| + additional corpora | 160GB | 8K | 100K | 94.0 / 87.7 | 89.3 | 95.6 |
| + longer training | 160GB | 8K | 300K | 94.4 / 88.7 | 90.0 | 96.1 |
| + longest training | 160GB | 8K | 500K | **94.6 / 89.4** | **90.2** | **96.4** |

These Table 4 rows accumulate changes. More diverse data improves every reported metric, and extra updates continue to help through 500K without an observed overfitting turn.

### Headline benchmarks

| Benchmark | RoBERTa | BERT-large | Notes |
|---|---:|---:|---|
| GLUE test average | **88.5** | n/a | ensemble, Table 5; 88.4 for XLNet |
| MNLI dev matched/mismatched | **90.2 / 90.2** | 86.6 / n/a | single model, Table 5 |
| CoLA dev | **68.0** | 60.6 | Matthews correlation, Table 5 |
| SQuAD 1.1 dev | **88.9 EM / 94.6 F1** | 84.1 / 90.9 | no external QA data, Table 6 |
| SQuAD 2.0 dev | **86.5 EM / 89.4 F1** | 79.0 / 81.8 | Table 6 |
| RACE test | **83.2** | 72.0 | accuracy, Table 7 |

The central result is methodological rather than one score: unchanged MLM and essentially unchanged BERT-large architecture surpassed BERT and competed with XLNet once training was strengthened. Consequently, objective comparisons that use an undertrained BERT control can overestimate the benefit of the proposed objective.

## Limitations & follow-ups

- The final result changes data volume, composition, batch size, tokenizer, sequence construction, and update count. Table 4 shows cumulative gains but does not fully factorially isolate every interaction.
- Training on 1,024 V100 GPUs is not accessible to many researchers, and the paper reports no inference-efficiency gain over BERT-large.
- The model remains English-only, dense-attention, limited to 512 positions, and expensive to fine-tune or store per task.
- Dynamic masking was only slightly better in the controlled comparison; it should not receive credit for the full final gain.
- The best GLUE test number uses task-specific reformulations and ensembles, whereas the cleaner scientific comparison is the single-model development setting.
- [ALBERT](bert-efficient_2019_albert.md) attacks parameter storage and discourse objectives; [SpanBERT](bert-span_2019_spanbert.md) changes the corruption unit and explicitly trains span boundaries. These address different bottlenecks than RoBERTa's recipe optimization.

## Links

- **arXiv:** [abs](https://arxiv.org/abs/1907.11692v1) · [html](https://arxiv.org/html/1907.11692v1) · [pdf](https://arxiv.org/pdf/1907.11692v1)
- **Code:** [fairseq RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)
- **Hugging Face:** [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large)
- **Project page:** —
- **Blog posts:** —
- **Talks / videos:** —
- **OpenReview / venue page:** —
- **Papers-with-Code:** [RoBERTa](https://paperswithcode.com/method/roberta)
- **BibTeX:** [arXiv export](https://arxiv.org/bibtex/1907.11692)
- **Related / successor papers:** [BERT](bert-encoder_2018_bert-pretraining.md) · [ALBERT](bert-efficient_2019_albert.md) · [SpanBERT](bert-span_2019_spanbert.md)

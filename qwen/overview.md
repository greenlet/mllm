# Qwen Models — Overview

Qwen ("通义千问", *Tongyi Qianwen*) is the family of large language and multimodal foundation models developed by **Alibaba Cloud / Alibaba DAMO Academy**. The series spans dense and Mixture‑of‑Experts (MoE) LLMs, vision–language, audio–language, omni (any‑to‑any), code, math, and embedding/reranker models. Most checkpoints are released under permissive licenses (Apache‑2.0 for the majority of Qwen2 / Qwen2.5 / Qwen3 lineups; some early and the largest variants use a custom Tongyi Qianwen license).

---

## 1. Generations at a glance

| Generation | Released | Highlights | Context | Notable sizes |
|---|---|---|---|---|
| **Qwen (1.0)** | Aug 2023 | First open release; bilingual (zh/en); RoPE, SwiGLU, RMSNorm, untied embeddings | 8K (→32K with NTK) | 1.8B, 7B, 14B, 72B |
| **Qwen1.5** | Feb 2024 | Unified HF integration; GQA on larger sizes; first MoE (`A2.7B`) | 32K | 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B, MoE‑A2.7B |
| **Qwen2** | Jun 2024 | GQA across all sizes; YaRN long‑context; multilingual (29 langs); strong code/math | 32K–128K | 0.5B, 1.5B, 7B, 57B‑A14B (MoE), 72B |
| **Qwen2.5** | Sep 2024 | Pre‑trained on **18T tokens**; structured‑output & tool‑use upgrades; specialized lines (Coder, Math, VL, Omni) | 32K → 128K (1M with YaRN) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B |
| **Qwen3** | Apr 2025 | **Hybrid "thinking / non‑thinking" mode** in a single model; expanded MoE; 36T pre‑training tokens; 119 languages | 32K–128K (256K for some MoE) | 0.6B, 1.7B, 4B, 8B, 14B, 32B, **30B‑A3B**, **235B‑A22B** |

---

## 2. Architecture & training stack

### 2.1 Backbone (text)
- **Decoder‑only Transformer** with:
  - **Rotary Position Embeddings (RoPE)** — extended via NTK‑aware scaling and **YaRN** for long context.
  - **SwiGLU** feed‑forward activations.
  - **RMSNorm** (pre‑norm).
  - **Grouped‑Query Attention (GQA)** in Qwen2+ for cheaper KV cache.
  - **Untied input/output embeddings** (Qwen1/2; tied in some small variants).
  - **QK‑Norm** and **bias‑free linear layers** in Qwen3.
- **Tokenizer**: BPE (`tiktoken`‑style) with ~152K vocabulary, byte‑level fallback, strong CJK coverage.

### 2.2 Mixture‑of‑Experts (MoE)
- Fine‑grained experts with shared experts (DeepSeek‑MoE‑style routing) in Qwen2/2.5/3 MoE.
- Naming convention `Total‑A{Active}` — e.g. **Qwen3‑235B‑A22B** = 235 B total parameters, ~22 B active per token; **Qwen3‑30B‑A3B** = 30 B / 3 B active.
- Top‑k routing (typically k=8 of 128 experts in Qwen3 MoE) with auxiliary load‑balancing loss.

### 2.3 Long context
- Base context 32K, extended via **YaRN** + **Dual Chunk Attention (DCA)** to 128K and up to **1M tokens** (Qwen2.5‑Turbo / Qwen2.5‑1M).

### 2.4 Post‑training
- **SFT → DPO / GRPO / RLHF** pipeline.
- Qwen3 introduces **"thinking mode"** (chain‑of‑thought wrapped in `<think>…</think>`) toggled at inference via `enable_thinking=True` or the `/think` `/no_think` chat tags — one model, two regimes.

---

## 3. Inputs, outputs, modalities

### 3.1 Text‑only LLMs (Qwen, Qwen1.5, Qwen2, Qwen2.5, Qwen3)
- **Input**: tokenized text (BPE), ChatML‑style template (`<|im_start|>role … <|im_end|>`).
- **Output**: text; structured JSON / tool‑calls supported natively.

### 3.2 Qwen‑VL / Qwen2‑VL / Qwen2.5‑VL — Vision–Language
- **Inputs**: interleaved **text + images + video**.
- **Vision encoder**: **ViT** (~675 M params in Qwen2.5‑VL) with **2D RoPE** and **window attention**.
- **Native dynamic resolution**: image is split into 14×14 patches, then **2×2 patches merged**, producing a variable number of visual tokens proportional to the image area (no forced resize). Qwen2.5‑VL supports **absolute time encoding** for video frames and processes long videos via sparse frame sampling + **M‑RoPE** (multimodal RoPE jointly encoding time / height / width / text position).
- **Cross‑modal fusion**: visual tokens are projected by an MLP adapter into the LLM embedding space and concatenated with text tokens — the LLM attends to them like any other tokens.
- **Capabilities**: VQA, OCR (incl. multilingual / handwriting / documents), grounding (bounding boxes & points), chart/table understanding, GUI agent / "computer use", long‑video reasoning.

### 3.3 Qwen‑Audio / Qwen2‑Audio
- **Inputs**: audio waveform + text.
- **Audio encoder**: **Whisper‑Large‑v3** initialized encoder; 16 kHz mel‑spectrogram → audio tokens projected into the LLM.
- **Capabilities**: ASR, speech translation, audio captioning, sound‑event / music analysis, voice‑chat (free‑form spoken dialogue without ASR pre‑step).

### 3.4 Qwen2.5‑Omni — Any‑to‑Any
- **Inputs**: text, image, video (with audio track), and raw audio — processed in real time.
- **Architecture**: novel **"Thinker–Talker"** design.
  - *Thinker*: multimodal Transformer that consumes all modalities and produces text + high‑level semantic representations.
  - *Talker*: autoregressive **speech codec** decoder that streams audio tokens conditioned on Thinker's hidden states.
  - **TMRoPE** (Time‑aligned Multimodal RoPE) synchronizes video frames with audio.
- **Outputs**: streaming **text + natural speech** simultaneously.

### 3.5 Specialist lines
- **Qwen2.5‑Coder** (0.5B–32B): trained on 5.5T code tokens; FIM, repo‑level context, 92 languages.
- **Qwen2.5‑Math** / **Qwen3‑Math**: tool‑integrated reasoning (TIR) with Python, chain‑of‑thought + self‑consistency.
- **Qwen3‑Embedding / Qwen3‑Reranker** (0.6B / 4B / 8B): dense + multi‑vector embeddings, top of MTEB multilingual leaderboard at release.
- **QwQ‑32B** (Nov 2024): research preview reasoning model, precursor to Qwen3 thinking mode.
- **QVQ‑72B‑Preview**: visual reasoning ("think with images").

---

## 4. Benchmarks (representative, as reported by Alibaba)

> Scores are from the official Qwen technical reports / model cards; exact numbers vary slightly across revisions. Use as relative indicators.

### 4.1 Qwen2.5‑72B‑Instruct vs peers
| Benchmark | Qwen2.5‑72B‑Inst | Llama‑3.1‑70B‑Inst | Mistral‑Large‑2 |
|---|---|---|---|
| MMLU‑Pro | **71.1** | 66.4 | 69.4 |
| MMLU‑redux | **86.8** | 83.0 | 83.0 |
| GPQA | **49.0** | 46.7 | 52.0 |
| MATH | **83.1** | 68.0 | 69.9 |
| HumanEval | **86.6** | 80.5 | 92.1 |
| MBPP | **88.2** | 84.2 | 80.0 |
| LiveCodeBench | **55.5** | 32.1 | 41.0 |
| IFEval (strict‑prompt) | **84.1** | 83.6 | 64.5 |

### 4.2 Qwen3 flagship (Qwen3‑235B‑A22B, thinking mode)
| Benchmark | Qwen3‑235B‑A22B | DeepSeek‑R1 | OpenAI o1 |
|---|---|---|---|
| AIME’24 | **85.7** | 79.8 | 74.3 |
| AIME’25 | **81.5** | 70.0 | 79.2 |
| LiveCodeBench v5 | **70.7** | 64.3 | 63.4 |
| CodeForces (Elo) | **2056** | 2029 | 1891 |
| Arena‑Hard | **95.6** | 93.2 | 92.1 |
| BFCL v3 (tool use) | **70.8** | 56.9 | 67.8 |

### 4.3 Qwen2.5‑VL‑72B (vision)
| Benchmark | Qwen2.5‑VL‑72B | GPT‑4o | Claude‑3.5‑Sonnet |
|---|---|---|---|
| MMMU | 70.2 | 69.1 | 70.4 |
| DocVQA | **96.4** | 92.8 | 95.2 |
| ChartQA | **89.5** | 85.7 | 90.8 |
| MathVista | 74.8 | 63.8 | 65.4 |
| OCRBench | **885** | 736 | 788 |
| VideoMME (w/ subs) | **73.3** | 71.9 | — |

### 4.4 Qwen2.5‑Coder‑32B‑Instruct
- HumanEval **92.7**, MBPP **90.2**, LiveCodeBench **31.4**, BigCodeBench **38.3**, Spider (text2SQL) **85.1** — on par with GPT‑4o on coding tasks.

### 4.5 Qwen3‑Embedding‑8B
- **MTEB Multilingual avg ≈ 70.6** (SOTA at release, June 2025), beating `gte‑Qwen2‑7B‑instruct`, `multilingual‑e5‑large`, and `bge‑m3`.

---

## 5. Key papers & technical reports

| Year | Title | Link |
|---|---|---|
| 2023 | *Qwen Technical Report* | https://arxiv.org/abs/2309.16609 |
| 2023 | *Qwen‑VL: A Versatile Vision‑Language Model for Understanding, Localization, Text Reading, and Beyond* | https://arxiv.org/abs/2308.12966 |
| 2023 | *Qwen‑Audio: Advancing Universal Audio Understanding via Unified Large‑Scale Audio‑Language Models* | https://arxiv.org/abs/2311.07919 |
| 2024 | *Qwen2 Technical Report* | https://arxiv.org/abs/2407.10671 |
| 2024 | *Qwen2‑VL: Enhancing Vision‑Language Model's Perception of the World at Any Resolution* | https://arxiv.org/abs/2409.12191 |
| 2024 | *Qwen2‑Audio Technical Report* | https://arxiv.org/abs/2407.10759 |
| 2024 | *Qwen2.5 Technical Report* | https://arxiv.org/abs/2412.15115 |
| 2024 | *Qwen2.5‑Coder Technical Report* | https://arxiv.org/abs/2409.12186 |
| 2024 | *Qwen2.5‑Math Technical Report* | https://arxiv.org/abs/2409.12122 |
| 2025 | *Qwen2.5‑VL Technical Report* | https://arxiv.org/abs/2502.13923 |
| 2025 | *Qwen2.5‑1M Technical Report* (long context) | https://arxiv.org/abs/2501.15383 |
| 2025 | *Qwen2.5‑Omni Technical Report* | https://arxiv.org/abs/2503.20215 |
| 2025 | *Qwen3 Technical Report* | https://arxiv.org/abs/2505.09388 |
| 2025 | *Qwen3‑Embedding / Qwen3‑Reranker* | https://arxiv.org/abs/2506.05176 |

### Foundational / closely related work
- **RoPE** — Su et al., *RoFormer*, https://arxiv.org/abs/2104.09864
- **YaRN** — Peng et al., https://arxiv.org/abs/2309.00071
- **GQA** — Ainslie et al., https://arxiv.org/abs/2305.13245
- **SwiGLU** — Shazeer, https://arxiv.org/abs/2002.05202
- **RMSNorm** — Zhang & Sennrich, https://arxiv.org/abs/1910.07467
- **Mixture‑of‑Experts routing** — Switch Transformer (https://arxiv.org/abs/2101.03961), DeepSeek‑MoE (https://arxiv.org/abs/2401.06066)
- **Whisper** (audio encoder used in Qwen‑Audio) — Radford et al., https://arxiv.org/abs/2212.04356
- **ViT** (vision backbone) — Dosovitskiy et al., https://arxiv.org/abs/2010.11929
- **DPO** — Rafailov et al., https://arxiv.org/abs/2305.18290
- **GRPO** — DeepSeekMath, https://arxiv.org/abs/2402.03300

---

## 6. Ecosystem & tooling

- **Hugging Face**: https://huggingface.co/Qwen
- **ModelScope**: https://modelscope.cn/organization/qwen
- **GitHub**: https://github.com/QwenLM (Qwen, Qwen2, Qwen2.5, Qwen3, Qwen‑Agent, Qwen‑VL, Qwen2.5‑Omni, Qwen3‑Coder)
- **Blog / docs**: https://qwen.readthedocs.io , https://qwenlm.github.io/blog/
- **Inference**: native `transformers`, `vLLM`, `SGLang`, `TGI`, `llama.cpp` (GGUF), `MLX`, `Ollama`, `LMStudio`.
- **Fine‑tuning**: `LLaMA‑Factory`, `Axolotl`, `Unsloth`, `ms‑swift` (Alibaba’s official trainer).

---

## 7. Licensing summary

| Model line | License |
|---|---|
| Qwen‑72B, Qwen1.5‑72B/110B | Tongyi Qianwen License (free for ≤100M MAU) |
| Qwen2 / 2.5 / 3 (most sizes), Coder, Math, VL ≤32B, Embedding | **Apache‑2.0** |
| Qwen2.5‑VL‑72B, Qwen3‑235B‑A22B | Tongyi Qianwen License |
| Qwen2.5‑Omni | Apache‑2.0 |

Always check the specific model card before commercial deployment.

# Qwen Models — Overview

Qwen ("通义千问", *Tongyi Qianwen*) is the family of large language and multimodal foundation models developed by **Alibaba Cloud / Alibaba DAMO Academy**. The series spans dense and Mixture‑of‑Experts ([MoE][MoE]) LLMs, vision–language, audio–language, omni (any‑to‑any), code, math, and embedding/reranker models. Most checkpoints are released under permissive licenses (Apache‑2.0 for the majority of Qwen2 / Qwen2.5 / Qwen3 lineups; some early and the largest variants use a custom Tongyi Qianwen license).

---

## 1. Generations at a glance

| Generation | Released | Highlights | Context | Notable sizes |
|---|---|---|---|---|
| **Qwen (1.0)** | Aug 2023 | First open release; bilingual (zh/en); [RoPE][RoPE], [SwiGLU][SwiGLU], [RMSNorm][RMSNorm], untied embeddings | 8K (→32K with [NTK][NTK]) | 1.8B, 7B, 14B, 72B |
| **Qwen1.5** | Feb 2024 | Unified HF integration; [GQA][GQA] on larger sizes; first [MoE][MoE] (`A2.7B`) | 32K | 0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B, MoE‑A2.7B |
| **Qwen2** | Jun 2024 | [GQA][GQA] across all sizes; [YaRN][YaRN] long‑context; multilingual (29 langs); strong code/math | 32K–128K | 0.5B, 1.5B, 7B, 57B‑A14B (MoE), 72B |
| **Qwen2.5** | Sep 2024 | Pre‑trained on **18T tokens**; structured‑output & tool‑use upgrades; specialized lines (Coder, Math, VL, Omni) | 32K → 128K (1M with [YaRN][YaRN]) | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B |
| **Qwen3** | Apr 2025 | **Hybrid "thinking / non‑thinking" mode** in a single model; expanded [MoE][MoE]; 36T pre‑training tokens; 119 languages | 32K–128K (256K for some MoE) | 0.6B, 1.7B, 4B, 8B, 14B, 32B, **30B‑A3B**, **235B‑A22B** |

---

## 2. Architecture & training stack

### 2.1 Backbone (text)
- **Decoder‑only Transformer** ([Vaswani et al.][Transformer]) with:
  - **Rotary Position Embeddings** ([RoPE][RoPE]) — extended via [NTK][NTK]‑aware scaling and [YaRN][YaRN] for long context.
  - **[SwiGLU][SwiGLU]** feed‑forward activations.
  - **[RMSNorm][RMSNorm]** (pre‑norm).
  - **Grouped‑Query Attention** ([GQA][GQA]) in Qwen2+ for cheaper KV cache.
  - **Untied input/output embeddings** (Qwen1/2; tied in some small variants — see [Press & Wolf][TiedEmb]).
  - **[QK‑Norm][QKNorm]** and **bias‑free linear layers** in Qwen3.
  - **[Flash‑Attention 2][FlashAttn]** kernels for training and inference.

### 2.2 Tokenization, vocabulary, embedding & intrinsic dimensions

Qwen uses a **byte‑level BPE** tokenizer ([Sennrich et al.][BPE], [Wang et al. byte‑level BPE][ByteBPE]) implemented in the [`tiktoken`][tiktoken] style, derived from the open‑sourced **[Qwen tokenizer][QwenTokenizer]** (a.k.a. `qwen.tiktoken` / `Qwen2Tokenizer`).

| Generation | Vocab size | Special tokens | Notes |
|---|---|---|---|
| Qwen (1.0)   | **151,851** | `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, 205 reserved | byte‑level BPE, balanced for zh/en/code |
| Qwen1.5 / 2 / 2.5 | **151,936** (padded) | adds tool / FIM / vision placeholders | same merges as Qwen, padded to a multiple of 128 for tensor‑parallel friendliness |
| Qwen3        | **151,936** | adds `<think>` / `</think>` for hybrid reasoning | tokenizer compatible with Qwen2.5 |
| Qwen2.5‑Coder | 151,936 | adds `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|repo_name|>`, `<|file_sep|>` | enables [FIM][FIM] & repo‑level training |
| Qwen2‑VL / 2.5‑VL | 151,936 | `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, `<|video_pad|>` | visual tokens injected between markers |
| Qwen2‑Audio / 2.5‑Omni | 151,936 | `<|audio_bos|>`, `<|audio_eos|>`, codec tokens for the Talker | speech generated via a discrete audio codec ([Encodec][Encodec]‑style) |

**Hidden size, FFN, heads, layers (representative dense models):**

| Model | Layers | Hidden $d_{\text{model}}$ | FFN $d_{\text{ff}}$ | Heads (Q / KV) | Head dim | Tied emb. |
|---|---|---|---|---|---|---|
| Qwen2‑0.5B   | 24 | 896  | 4,864  | 14 / 2 | 64  | yes |
| Qwen2‑1.5B   | 28 | 1,536 | 8,960 | 12 / 2 | 128 | yes |
| Qwen2‑7B     | 28 | 3,584 | 18,944 | 28 / 4 | 128 | no |
| Qwen2‑72B    | 80 | 8,192 | 29,568 | 64 / 8 | 128 | no |
| Qwen2.5‑3B   | 36 | 2,048 | 11,008 | 16 / 2 | 128 | yes |
| Qwen2.5‑14B  | 48 | 5,120 | 13,824 | 40 / 8 | 128 | no |
| Qwen2.5‑32B  | 64 | 5,120 | 27,648 | 40 / 8 | 128 | no |
| Qwen3‑8B     | 36 | 4,096 | 12,288 | 32 / 8 | 128 | no |
| Qwen3‑32B    | 64 | 5,120 | 27,648 | 64 / 8 | 128 | no |
| Qwen3‑30B‑A3B (MoE) | 48 | 2,048 | 768 (per expert) × 128 experts, top‑8 | 32 / 4 | 128 | no |
| Qwen3‑235B‑A22B (MoE) | 94 | 4,096 | 1,536 × 128 experts, top‑8 | 64 / 4 | 128 | no |

> $d_{\text{model}}$ is the *embedding / residual stream* dimension. The KV‑cache footprint per token is $2 \cdot L \cdot H_{kv} \cdot d_{\text{head}}$ (much smaller than $2 L d_{\text{model}}$ thanks to [GQA][GQA]).

**Embedding (representation) dimensionality for the dedicated embedding line:**

| Embedding model | Backbone | Output dim | Pooling | Max length |
|---|---|---|---|---|
| `gte‑Qwen2‑1.5B‑instruct` | Qwen2‑1.5B | 1,536 | last‑token | 32K |
| `gte‑Qwen2‑7B‑instruct`   | Qwen2‑7B   | 3,584 | last‑token | 32K |
| `Qwen3‑Embedding‑0.6B`    | Qwen3‑0.6B | 1,024 (Matryoshka, [MRL][MRL]) | last‑token | 32K |
| `Qwen3‑Embedding‑4B`      | Qwen3‑4B   | 2,560 (MRL) | last‑token | 32K |
| `Qwen3‑Embedding‑8B`      | Qwen3‑8B   | 4,096 (MRL, truncatable to 1024/2048) | last‑token | 32K |

**Intrinsic dimensionality.** Although the residual stream is 4K–8K wide, empirical analyses with [intrinsic‑dimension probes][IntrinsicDim] (TwoNN, MLE, PR‑rank) on Qwen2/3 hidden states report effective dimensions of roughly **80–250** in middle layers and **<100** near the output — a pattern consistent with other large LMs ([Valeriani et al.][GeometryLLM], [Cheng et al. *LLM‑ID*][LLMID]) and with the [Linear Representation Hypothesis][LRH] / superposition view of [Elhage et al.][Superposition]. For task‑adaptation, [Aghajanyan et al.][IntrinsicDimSGD] showed that fine‑tuning a 7B‑class model on a downstream task only needs an *intrinsic* update of dimension ~$10^2$–$10^3$, which is exactly the regime exploited by [LoRA][LoRA] and friends (see §6).

### 2.3 Mixture‑of‑Experts ([MoE][MoE])
- **Fine‑grained experts with shared experts** ([DeepSeek‑MoE][DeepSeekMoE]) in Qwen2 / 2.5 / 3 MoE.
- Naming convention `Total‑A{Active}` — e.g. **Qwen3‑235B‑A22B** = 235 B total parameters, ~22 B active per token; **Qwen3‑30B‑A3B** = 30 B / 3 B active.
- **Top‑k routing** (typically k=8 of 128 experts in Qwen3 MoE) with auxiliary load‑balancing loss ([Switch Transformer][Switch], [GShard][GShard]).

### 2.4 Long context
- Base context 32K, extended via [YaRN][YaRN] + **Dual Chunk Attention** ([DCA][DCA]) to 128K, and up to **1M tokens** in **Qwen2.5‑Turbo / Qwen2.5‑1M** ([Qwen2.5‑1M Tech Report][Qwen2.5-1M]).

### 2.5 Post‑training
- **SFT → preference optimization** with [DPO][DPO] / [GRPO][GRPO] / classical [RLHF][RLHF] (PPO, [Schulman et al.][PPO]).
- Qwen3 introduces **"thinking mode"** (chain‑of‑thought wrapped in `<think>…</think>`, à la [CoT][CoT] and [DeepSeek‑R1][R1]) toggled at inference via `enable_thinking=True` or the `/think` `/no_think` chat tags — one model, two regimes.

---

## 3. Inputs, outputs, modalities

### 3.1 Text‑only LLMs (Qwen, Qwen1.5, Qwen2, Qwen2.5, Qwen3)
- **Input**: tokenized text ([byte‑level BPE][BPE]), **[ChatML][ChatML]**‑style template (`<|im_start|>role … <|im_end|>`).
- **Output**: text; structured JSON / tool‑calls supported natively (see [Qwen‑Agent][QwenAgent]).

### 3.2 Qwen‑VL / Qwen2‑VL / Qwen2.5‑VL — Vision–Language
- **Inputs**: interleaved **text + images + video**.
- **Vision encoder**: a [ViT][ViT] (~675 M params in Qwen2.5‑VL) with **2D [RoPE][RoPE]** and **window attention** ([Swin][Swin]).
- **Native dynamic resolution**: image is split into 14×14 patches, then **2×2 patches merged** via an MLP ([PixelShuffle‑like][PixelShuffle]), producing a variable number of visual tokens proportional to image area (no forced resize). Qwen2.5‑VL adds **absolute time encoding** for video frames and processes long videos via sparse frame sampling + **[M‑RoPE][MRoPE]** (multimodal RoPE jointly encoding time / height / width / text position).
- **Cross‑modal fusion**: visual tokens are projected by an MLP adapter (cf. [LLaVA][LLaVA]) into the LLM embedding space and concatenated with text tokens — the LLM attends to them like any other tokens.
- **Capabilities**: VQA, OCR, grounding (boxes & points), chart/table understanding, GUI agent / "computer use" ([Qwen2.5‑VL Tech Report][Qwen2.5-VL]).

### 3.3 Qwen‑Audio / Qwen2‑Audio
- **Inputs**: audio waveform + text.
- **Audio encoder**: initialized from **[Whisper‑Large‑v3][Whisper]**; 16 kHz log‑mel spectrogram → audio tokens → MLP projector → LLM.
- **Capabilities**: ASR, speech translation, audio captioning, sound‑event / music analysis, voice‑chat (no separate ASR step).

### 3.4 Qwen2.5‑Omni — Any‑to‑Any
- **Inputs**: text, image, video (with audio), and raw audio — streamed in real time.
- **Architecture**: novel **"Thinker–Talker"** design ([Qwen2.5‑Omni Tech Report][Qwen2.5-Omni]).
  - *Thinker*: multimodal Transformer that consumes all modalities and produces text + high‑level semantic states.
  - *Talker*: autoregressive **neural audio codec** decoder (cf. [Encodec][Encodec], [SoundStream][SoundStream]) that streams speech tokens conditioned on Thinker hidden states.
  - **TMRoPE** (Time‑aligned Multimodal [RoPE][RoPE]) synchronizes video frames with audio.
- **Outputs**: streaming **text + natural speech** simultaneously.

### 3.5 Specialist lines
- **Qwen2.5‑Coder** (0.5B–32B): 5.5T code tokens; [FIM][FIM], repo‑level context, 92 languages ([Qwen2.5‑Coder Tech Report][Qwen2.5-Coder]).
- **Qwen2.5‑Math** / **Qwen3‑Math**: **tool‑integrated reasoning** ([TIR][TIR]) with Python, [CoT][CoT] + [self‑consistency][SelfConsistency].
- **Qwen3‑Embedding / Qwen3‑Reranker** (0.6B / 4B / 8B): dense + multi‑vector embeddings with [MRL][MRL], top of [MTEB][MTEB] multilingual leaderboard at release.
- **[QwQ‑32B][QwQ]** (Nov 2024): research preview reasoning model, precursor to Qwen3 thinking mode.
- **[QVQ‑72B‑Preview][QVQ]**: visual reasoning ("think with images").

---

## 4. Benchmarks (representative, as reported by Alibaba)

> Scores are from official Qwen technical reports / model cards; exact numbers vary slightly across revisions. Use as relative indicators.

### 4.1 Qwen2.5‑72B‑Instruct vs peers
| Benchmark | Qwen2.5‑72B‑Inst | Llama‑3.1‑70B‑Inst | Mistral‑Large‑2 |
|---|---|---|---|
| [MMLU‑Pro][MMLUPro] | **71.1** | 66.4 | 69.4 |
| [MMLU][MMLU]‑redux | **86.8** | 83.0 | 83.0 |
| [GPQA][GPQA] | **49.0** | 46.7 | 52.0 |
| [MATH][MATH] | **83.1** | 68.0 | 69.9 |
| [HumanEval][HumanEval] | **86.6** | 80.5 | 92.1 |
| [MBPP][MBPP] | **88.2** | 84.2 | 80.0 |
| [LiveCodeBench][LiveCodeBench] | **55.5** | 32.1 | 41.0 |
| [IFEval][IFEval] (strict‑prompt) | **84.1** | 83.6 | 64.5 |

### 4.2 Qwen3 flagship (Qwen3‑235B‑A22B, thinking mode)
| Benchmark | Qwen3‑235B‑A22B | DeepSeek‑R1 | OpenAI o1 |
|---|---|---|---|
| [AIME][AIME]’24 | **85.7** | 79.8 | 74.3 |
| AIME’25 | **81.5** | 70.0 | 79.2 |
| [LiveCodeBench][LiveCodeBench] v5 | **70.7** | 64.3 | 63.4 |
| CodeForces (Elo) | **2056** | 2029 | 1891 |
| [Arena‑Hard][ArenaHard] | **95.6** | 93.2 | 92.1 |
| [BFCL][BFCL] v3 (tool use) | **70.8** | 56.9 | 67.8 |

### 4.3 Qwen2.5‑VL‑72B (vision)
| Benchmark | Qwen2.5‑VL‑72B | GPT‑4o | Claude‑3.5‑Sonnet |
|---|---|---|---|
| [MMMU][MMMU] | 70.2 | 69.1 | 70.4 |
| [DocVQA][DocVQA] | **96.4** | 92.8 | 95.2 |
| [ChartQA][ChartQA] | **89.5** | 85.7 | 90.8 |
| [MathVista][MathVista] | 74.8 | 63.8 | 65.4 |
| [OCRBench][OCRBench] | **885** | 736 | 788 |
| [VideoMME][VideoMME] (w/ subs) | **73.3** | 71.9 | — |

### 4.4 Qwen2.5‑Coder‑32B‑Instruct
- [HumanEval][HumanEval] **92.7**, [MBPP][MBPP] **90.2**, [LiveCodeBench][LiveCodeBench] **31.4**, [BigCodeBench][BigCodeBench] **38.3**, [Spider][Spider] (text2SQL) **85.1** — on par with GPT‑4o on coding tasks.

### 4.5 Qwen3‑Embedding‑8B
- **[MTEB][MTEB] Multilingual avg ≈ 70.6** (SOTA at release, June 2025), beating `gte‑Qwen2‑7B‑instruct`, `multilingual‑e5‑large`, and `bge‑m3`.

---

## 5. Key Qwen papers & technical reports

| Year | Title | Link |
|---|---|---|
| 2023 | *Qwen Technical Report* | [arXiv:2309.16609][Qwen1] |
| 2023 | *Qwen‑VL: A Versatile Vision‑Language Model …* | [arXiv:2308.12966][Qwen-VL] |
| 2023 | *Qwen‑Audio: Universal Audio Understanding via Unified LALMs* | [arXiv:2311.07919][Qwen-Audio] |
| 2024 | *Qwen2 Technical Report* | [arXiv:2407.10671][Qwen2] |
| 2024 | *Qwen2‑VL: Perception of the World at Any Resolution* | [arXiv:2409.12191][Qwen2-VL] |
| 2024 | *Qwen2‑Audio Technical Report* | [arXiv:2407.10759][Qwen2-Audio] |
| 2024 | *Qwen2.5 Technical Report* | [arXiv:2412.15115][Qwen2.5] |
| 2024 | *Qwen2.5‑Coder Technical Report* | [arXiv:2409.12186][Qwen2.5-Coder] |
| 2024 | *Qwen2.5‑Math Technical Report* | [arXiv:2409.12122][Qwen2.5-Math] |
| 2025 | *Qwen2.5‑VL Technical Report* | [arXiv:2502.13923][Qwen2.5-VL] |
| 2025 | *Qwen2.5‑1M Technical Report* (long context) | [arXiv:2501.15383][Qwen2.5-1M] |
| 2025 | *Qwen2.5‑Omni Technical Report* | [arXiv:2503.20215][Qwen2.5-Omni] |
| 2025 | *Qwen3 Technical Report* | [arXiv:2505.09388][Qwen3] |
| 2025 | *Qwen3‑Embedding / Qwen3‑Reranker* | [arXiv:2506.05176][Qwen3-Emb] |

---

## 6. Fine‑tuning Qwen

### 6.1 Full / continued pre‑training
- Standard causal‑LM objective on additional domain corpora; usually combined with **replay** of general data to mitigate [catastrophic forgetting][CatForget].
- Long‑context extension via [YaRN][YaRN] + [DCA][DCA] needs only ~1B tokens of long sequences; see the recipe in [Qwen2.5‑1M][Qwen2.5-1M].

### 6.2 Supervised Fine‑Tuning (SFT)
- [ChatML][ChatML] formatting, loss masked on the assistant turn (cf. the [InstructGPT][RLHF] / [Alpaca][Alpaca] / [Vicuna][Vicuna] SFT recipe).
- Recipes:
  - Alibaba's official **[ms‑swift][ms-swift]** trainer (SFT, [DPO][DPO], [GRPO][GRPO], [LoRA][LoRA], [QLoRA][QLoRA], multi‑node via [DeepSpeed][DeepSpeed] / [FSDP][FSDP]).
  - **[LLaMA‑Factory][LLaMAFactory]** — the most popular community framework with first‑class Qwen support.
  - **[Axolotl][Axolotl]**, **[Unsloth][Unsloth]** (2× faster [QLoRA][QLoRA]), **[TRL][TRL]** from Hugging Face.

### 6.3 Parameter‑efficient fine‑tuning (PEFT)
Exploits the *low intrinsic dimensionality* of the adaptation problem ([Aghajanyan et al.][IntrinsicDimSGD]).

| Method | Reference | Notes for Qwen |
|---|---|---|
| Adapters | [Houlsby et al. 2019][Adapters] | rarely used today |
| Prefix / Prompt tuning | [Li & Liang][PrefixTuning], [Lester et al.][PromptTuning] | works but underperforms LoRA at 7B+ |
| **LoRA** | [Hu et al. 2021][LoRA] | default; rank 8–64, target `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| **QLoRA** | [Dettmers et al.][QLoRA] | 4‑bit NF4 base + LoRA — fine‑tune Qwen2.5‑72B on a single 80 GB GPU |
| **DoRA** | [Liu et al.][DoRA] | weight‑decomposed LoRA; small but consistent gains |
| **LoRA+** | [Hayou et al.][LoRAplus] | different LR for A/B matrices |
| **GaLore** | [Zhao et al.][GaLore] | full‑parameter training in low‑rank gradient subspace |
| **IA³** | [Liu et al.][IA3] | ultra‑light scaling vectors |

### 6.4 Preference / RL fine‑tuning
- **[RLHF][RLHF]** (reward model + [PPO][PPO]) — used for early Qwen‑Chat.
- **[DPO][DPO]** — direct preference optimization, default for Qwen2 / 2.5 alignment.
- **[KTO][KTO]** — Kahneman‑Tversky objective; tolerates unpaired data.
- **[ORPO][ORPO]** — odd‑ratio PO, single‑stage SFT+preference.
- **[GRPO][GRPO]** — group‑relative PO (no value model), introduced in DeepSeekMath, used for Qwen2.5‑Math and Qwen3 thinking‑mode RL.
- **RLAIF** — AI‑labelled preferences ([Bai et al. *Constitutional AI*][CAI], [Lee et al. *RLAIF*][RLAIF]).

### 6.5 Multimodal fine‑tuning
- Vision SFT: freeze [ViT][ViT], train projector + LLM (LoRA or full); cf. the [LLaVA][LLaVA] / [LLaVA‑1.5][LLaVA15] recipes adapted to Qwen2‑VL in `ms‑swift` and `LLaMA‑Factory`.
- Audio SFT: similar, freeze [Whisper][Whisper] encoder, train projector + LLM.
- Video / Omni: stage‑wise (image → video → audio → joint) following [Qwen2.5‑Omni][Qwen2.5-Omni].

### 6.6 Quantization (post‑/during training)
- **[GPTQ][GPTQ]**, **[AWQ][AWQ]**, **[SmoothQuant][SmoothQuant]**, **[BitsAndBytes NF4][QLoRA]**, **[GGUF k‑quants][llamacpp]** — official 4/8‑bit GGUF & GPTQ checkpoints are published for most Qwen sizes.

### 6.7 Distillation & merging
- Qwen3 small models are trained via **[knowledge distillation][KD]** from the 235B‑A22B teacher.
- Community model merging: **[Model Soups][ModelSoups]**, **[TIES‑Merging][TIES]**, **[DARE][DARE]** — broadly compatible with Qwen base + fine‑tunes since they share tokenizer and architecture.

---

## 7. Ecosystem & tooling

- **Hugging Face**: <https://huggingface.co/Qwen>
- **ModelScope**: <https://modelscope.cn/organization/qwen>
- **GitHub**: <https://github.com/QwenLM>
- **Docs / blog**: <https://qwen.readthedocs.io>, <https://qwenlm.github.io/blog/>
- **Inference**: native [`transformers`][HFTransformers], [`vLLM`][vLLM], [`SGLang`][SGLang], [`TGI`][TGI], [`llama.cpp`][llamacpp] (GGUF), [`MLX`][MLX], `Ollama`, `LMStudio`.
- **Fine‑tuning**: [`ms‑swift`][ms-swift], [`LLaMA‑Factory`][LLaMAFactory], [`Axolotl`][Axolotl], [`Unsloth`][Unsloth], [`TRL`][TRL].

---

## 8. Licensing summary

| Model line | License |
|---|---|
| Qwen‑72B, Qwen1.5‑72B/110B | Tongyi Qianwen License (free for ≤100M MAU) |
| Qwen2 / 2.5 / 3 (most sizes), Coder, Math, VL ≤32B, Embedding | **Apache‑2.0** |
| Qwen2.5‑VL‑72B, Qwen3‑235B‑A22B | Tongyi Qianwen License |
| Qwen2.5‑Omni | Apache‑2.0 |

Always check the specific model card before commercial deployment.

---

## 9. Threads

In-depth threads grouping related references by lineage. Each thread doc has an evolution table and Qwen-specific notes; per-paper recaps live under [`docs/papers/`](../papers/).

- [Positional encoding & long-context scaling](positional/positional.md) — RoPE → NTK-aware → YaRN → DCA. *Used in:* every Qwen text/vision model; powers the 32k→128k→1M context ladder.

*(More threads will be added as their papers are recapped.)*

---

## References

### Qwen technical reports
- [Qwen Technical Report (2023)][Qwen1]
- [Qwen-VL: A Versatile Vision-Language Model (2023)][Qwen-VL]
- [Qwen-Audio: Universal Audio Understanding via Unified LALMs (2023)][Qwen-Audio]
- [Qwen2 Technical Report (2024)][Qwen2]
- [Qwen2-VL: Perception of the World at Any Resolution (2024)][Qwen2-VL]
- [Qwen2-Audio Technical Report (2024)][Qwen2-Audio]
- [Qwen2.5 Technical Report (2024)][Qwen2.5]
- [Qwen2.5-Coder Technical Report (2024)][Qwen2.5-Coder]
- [Qwen2.5-Math Technical Report (2024)][Qwen2.5-Math]
- [Qwen2.5-VL Technical Report (2025)][Qwen2.5-VL]
- [Qwen2.5-1M Technical Report — long context (2025)][Qwen2.5-1M]
- [Qwen2.5-Omni Technical Report (2025)][Qwen2.5-Omni]
- [Qwen3 Technical Report (2025)][Qwen3]
- [Qwen3-Embedding / Qwen3-Reranker (2025)][Qwen3-Emb]
- [Qwen-Agent framework (GitHub)][QwenAgent]

### Architecture & attention

**Thread:** [Positional encoding & long-context scaling](positional/positional.md)

- [Vaswani et al., *Attention Is All You Need* (2017)][Transformer]
- [Su et al., *RoFormer: Rotary Position Embedding* (2021)][RoPE]
- [bloc97, *NTK-aware scaled RoPE* (2023)][NTK]
- [Peng et al., *YaRN: Efficient Context Window Extension* (2023)][YaRN]
- [Ainslie et al., *GQA: Generalized Multi-Query Attention* (2023)][GQA]
- [Shazeer, *GLU Variants Improve Transformer (SwiGLU)* (2020)][SwiGLU]
- [Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)][RMSNorm]
- [Henry et al., *Query-Key Normalization for Transformers* (2020)][QKNorm]
- [Dao, *FlashAttention-2* (2023)][FlashAttn]
- [Press & Wolf, *Using the Output Embedding to Improve LMs* (2016)][TiedEmb]
- [An et al., *Training-Free Long-Context Scaling via Dual Chunk Attention* (2024)][DCA]

### Tokenization
- [Sennrich et al., *Neural MT of Rare Words with Subword Units (BPE)* (2015)][BPE]
- [Wang et al., *Neural MT with Byte-Level Subwords* (2019)][ByteBPE]
- [OpenAI `tiktoken` BPE library (GitHub)][tiktoken]
- [Bavarian et al., *Efficient Training of LMs to Fill in the Middle (FIM)* (2022)][FIM]

### Mixture-of-Experts
- [Shazeer et al., *Outrageously Large Neural Networks: Sparsely-Gated MoE* (2017)][MoE]
- [Lepikhin et al., *GShard* (2020)][GShard]
- [Fedus et al., *Switch Transformer* (2021)][Switch]
- [Dai et al., *DeepSeek-MoE* (2024)][DeepSeekMoE]

### Multimodal building blocks
- [Dosovitskiy et al., *An Image Is Worth 16×16 Words (ViT)* (2020)][ViT]
- [Liu et al., *Swin Transformer* (2021)][Swin]
- [Shi et al., *Sub-Pixel CNN (PixelShuffle)* (2016)][PixelShuffle]
- [Liu et al., *Visual Instruction Tuning (LLaVA)* (2023)][LLaVA]
- [Liu et al., *Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)* (2023)][LLaVA15]
- [M-RoPE — introduced in Qwen2-VL (2024)][MRoPE]
- [Radford et al., *Whisper* (2022)][Whisper]
- [Défossez et al., *High Fidelity Neural Audio Compression (Encodec)* (2022)][Encodec]
- [Zeghidour et al., *SoundStream* (2021)][SoundStream]

### Reasoning, alignment & RL
- [Wei et al., *Chain-of-Thought Prompting* (2022)][CoT]
- [Wang et al., *Self-Consistency Improves CoT* (2022)][SelfConsistency]
- [Ouyang et al., *InstructGPT / RLHF* (2022)][RLHF]
- [Schulman et al., *PPO* (2017)][PPO]
- [Rafailov et al., *Direct Preference Optimization (DPO)* (2023)][DPO]
- [Ethayarajh et al., *Kahneman-Tversky Optimization (KTO)* (2024)][KTO]
- [Hong et al., *ORPO: Monolithic Preference Optimization* (2024)][ORPO]
- [Shao et al., *DeepSeekMath / GRPO* (2024)][GRPO]
- [Bai et al., *Constitutional AI* (2022)][CAI]
- [Lee et al., *RLAIF* (2023)][RLAIF]
- [DeepSeek-R1 (2025)][R1]

### PEFT & quantization
- [Houlsby et al., *Parameter-Efficient Transfer Learning (Adapters)* (2019)][Adapters]
- [Li & Liang, *Prefix-Tuning* (2021)][PrefixTuning]
- [Lester et al., *The Power of Scale for Prompt Tuning* (2021)][PromptTuning]
- [Hu et al., *LoRA: Low-Rank Adaptation* (2021)][LoRA]
- [Dettmers et al., *QLoRA* (2023)][QLoRA]
- [Liu et al., *DoRA: Weight-Decomposed LoRA* (2024)][DoRA]
- [Hayou et al., *LoRA+* (2024)][LoRAplus]
- [Zhao et al., *GaLore* (2024)][GaLore]
- [Liu et al., *Few-Shot PEFT (IA³)* (2022)][IA3]
- [Frantar et al., *GPTQ* (2022)][GPTQ]
- [Lin et al., *AWQ* (2023)][AWQ]
- [Xiao et al., *SmoothQuant* (2022)][SmoothQuant]
- [Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)][KD]
- [Wortsman et al., *Model Soups* (2022)][ModelSoups]
- [Yadav et al., *TIES-Merging* (2023)][TIES]
- [Yu et al., *DARE* (2023)][DARE]
- [Kirkpatrick et al., *Overcoming Catastrophic Forgetting (EWC)* (2016)][CatForget]

### Embeddings & representation geometry
- [Kusupati et al., *Matryoshka Representation Learning* (2022)][MRL]
- [Muennighoff et al., *MTEB* (2022)][MTEB]
- [Facco et al., *Estimating the intrinsic dimension via TwoNN* (2017)][IntrinsicDim]
- [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning* (2020)][IntrinsicDimSGD]
- [Valeriani et al., *The Geometry of Hidden Representations of LLMs* (2023)][GeometryLLM]
- [Cheng et al., *Emergence of a High-Dimensional Abstraction Phase in LLMs* (2024)][LLMID]
- [Park et al., *The Linear Representation Hypothesis* (2023)][LRH]
- [Elhage et al., *Toy Models of Superposition* (Anthropic, 2022)][Superposition]

### Benchmarks
- [Hendrycks et al., *MMLU* (2020)][MMLU]
- [Wang et al., *MMLU-Pro* (2024)][MMLUPro]
- [Rein et al., *GPQA* (2023)][GPQA]
- [Hendrycks et al., *MATH* (2021)][MATH]
- [Chen et al., *HumanEval / Codex* (2021)][HumanEval]
- [Austin et al., *MBPP* (2021)][MBPP]
- [Jain et al., *LiveCodeBench* (2024)][LiveCodeBench]
- [Zhuo et al., *BigCodeBench* (2024)][BigCodeBench]
- [Yu et al., *Spider text-to-SQL* (2018)][Spider]
- [Zhou et al., *IFEval* (2023)][IFEval]
- [LMSYS, *Arena-Hard* (2024)][ArenaHard]
- [Berkeley Function Calling Leaderboard (BFCL)][BFCL]
- [AIME competition problems][AIME]
- [Yue et al., *MMMU* (2023)][MMMU]
- [Mathew et al., *DocVQA* (2020)][DocVQA]
- [Masry et al., *ChartQA* (2022)][ChartQA]
- [Lu et al., *MathVista* (2023)][MathVista]
- [Liu et al., *OCRBench* (2023)][OCRBench]
- [Fu et al., *Video-MME* (2024)][VideoMME]

### Tooling
- [Hugging Face Transformers][HFTransformers]
- [vLLM][vLLM]
- [SGLang][SGLang]
- [Text Generation Inference (TGI)][TGI]
- [llama.cpp / GGUF][llamacpp]
- [Apple MLX][MLX]
- [Alibaba ms-swift][ms-swift]
- [LLaMA-Factory][LLaMAFactory]
- [Axolotl][Axolotl]
- [Unsloth][Unsloth]
- [Hugging Face TRL][TRL]
- [DeepSpeed][DeepSpeed]
- [PyTorch FSDP][FSDP]

### Misc references
- [OpenAI ChatML format spec][ChatML]
- [Qwen tokenizer (GitHub)][QwenTokenizer]
- [QwQ-32B model card][QwQ]
- [QVQ-72B-Preview model card][QVQ]
- [Taori et al., *Stanford Alpaca* (2023)][Alpaca]
- [Chiang et al., *Vicuna* (2023)][Vicuna]
- [Gou et al., *ToRA / Tool-Integrated Reasoning* (2023)][TIR]

---

<!-- Link reference definitions (invisible in rendered output) -->

[Qwen1]: https://arxiv.org/abs/2309.16609 "Qwen Technical Report (2023)"
[Qwen-VL]: https://arxiv.org/abs/2308.12966 "Qwen-VL (2023)"
[Qwen-Audio]: https://arxiv.org/abs/2311.07919 "Qwen-Audio (2023)"
[Qwen2]: https://arxiv.org/abs/2407.10671 "Qwen2 Technical Report (2024)"
[Qwen2-VL]: https://arxiv.org/abs/2409.12191 "Qwen2-VL (2024)"
[Qwen2-Audio]: https://arxiv.org/abs/2407.10759 "Qwen2-Audio (2024)"
[Qwen2.5]: https://arxiv.org/abs/2412.15115 "Qwen2.5 Technical Report (2024)"
[Qwen2.5-Coder]: https://arxiv.org/abs/2409.12186 "Qwen2.5-Coder (2024)"
[Qwen2.5-Math]: https://arxiv.org/abs/2409.12122 "Qwen2.5-Math (2024)"
[Qwen2.5-VL]: https://arxiv.org/abs/2502.13923 "Qwen2.5-VL (2025)"
[Qwen2.5-1M]: https://arxiv.org/abs/2501.15383 "Qwen2.5-1M (2025)"
[Qwen2.5-Omni]: https://arxiv.org/abs/2503.20215 "Qwen2.5-Omni (2025)"
[Qwen3]: https://arxiv.org/abs/2505.09388 "Qwen3 Technical Report (2025)"
[Qwen3-Emb]: https://arxiv.org/abs/2506.05176 "Qwen3-Embedding/Reranker (2025)"
[QwenAgent]: https://github.com/QwenLM/Qwen-Agent "Qwen-Agent framework"
[Transformer]: https://arxiv.org/abs/1706.03762 "Vaswani et al., Attention Is All You Need (2017)"
[RoPE]: ../papers/p000_2021_positional_rope-roformer.md "Su et al., RoFormer: Rotary Position Embedding (2021) — local recap"
[NTK]: ../papers/p001_2023_positional_ntk-aware-rope.md "NTK-aware RoPE scaling (bloc97, 2023) — local recap"
[YaRN]: ../papers/p002_2023_positional_yarn-context-extension.md "Peng et al., YaRN: Efficient Context Window Extension (2023) — local recap"
[GQA]: https://arxiv.org/abs/2305.13245 "Ainslie et al., GQA (2023)"
[SwiGLU]: https://arxiv.org/abs/2002.05202 "Shazeer, GLU Variants Improve Transformer (2020)"
[RMSNorm]: https://arxiv.org/abs/1910.07467 "Zhang & Sennrich, Root Mean Square Layer Normalization (2019)"
[QKNorm]: https://arxiv.org/abs/2010.04245 "Henry et al., Query-Key Normalization for Transformers (2020)"
[FlashAttn]: https://arxiv.org/abs/2307.08691 "Dao, FlashAttention-2 (2023)"
[TiedEmb]: https://arxiv.org/abs/1608.05859 "Press & Wolf, Using the Output Embedding to Improve LMs (2016)"
[DCA]: ../papers/p003_2024_positional_dca-dual-chunk-attention.md "An et al., Training-Free Long-Context Scaling via Dual Chunk Attention (2024) — local recap"
[BPE]: https://arxiv.org/abs/1508.07909 "Sennrich et al., Neural MT of Rare Words with Subword Units (BPE, 2015)"
[ByteBPE]: https://arxiv.org/abs/1909.03341 "Wang et al., Neural MT with Byte-Level Subwords (2019)"
[tiktoken]: https://github.com/openai/tiktoken "OpenAI tiktoken BPE library"
[FIM]: https://arxiv.org/abs/2207.14255 "Bavarian et al., Efficient Training of LMs to Fill in the Middle (2022)"
[MoE]: https://arxiv.org/abs/1701.06538 "Shazeer et al., Outrageously Large Neural Networks: Sparsely-Gated MoE (2017)"
[GShard]: https://arxiv.org/abs/2006.16668 "Lepikhin et al., GShard (2020)"
[Switch]: https://arxiv.org/abs/2101.03961 "Fedus et al., Switch Transformer (2021)"
[DeepSeekMoE]: https://arxiv.org/abs/2401.06066 "Dai et al., DeepSeek-MoE (2024)"
[ViT]: https://arxiv.org/abs/2010.11929 "Dosovitskiy et al., An Image Is Worth 16x16 Words (ViT, 2020)"
[Swin]: https://arxiv.org/abs/2103.14030 "Liu et al., Swin Transformer (2021)"
[PixelShuffle]: https://arxiv.org/abs/1609.05158 "Shi et al., Sub-Pixel CNN (PixelShuffle, 2016)"
[LLaVA]: https://arxiv.org/abs/2304.08485 "Liu et al., Visual Instruction Tuning (LLaVA, 2023)"
[LLaVA15]: https://arxiv.org/abs/2310.03744 "Liu et al., Improved Baselines with Visual Instruction Tuning (LLaVA-1.5, 2023)"
[MRoPE]: https://arxiv.org/abs/2409.12191 "M-RoPE — introduced in Qwen2-VL (2024)"
[Whisper]: https://arxiv.org/abs/2212.04356 "Radford et al., Robust Speech Recognition via Large-Scale Weak Supervision (Whisper, 2022)"
[Encodec]: https://arxiv.org/abs/2210.13438 "Défossez et al., High Fidelity Neural Audio Compression (Encodec, 2022)"
[SoundStream]: https://arxiv.org/abs/2107.03312 "Zeghidour et al., SoundStream (2021)"
[CoT]: https://arxiv.org/abs/2201.11903 "Wei et al., Chain-of-Thought Prompting (2022)"
[SelfConsistency]: https://arxiv.org/abs/2203.11171 "Wang et al., Self-Consistency Improves CoT (2022)"
[RLHF]: https://arxiv.org/abs/2203.02155 "Ouyang et al., InstructGPT / RLHF (2022)"
[PPO]: https://arxiv.org/abs/1707.06347 "Schulman et al., PPO (2017)"
[DPO]: https://arxiv.org/abs/2305.18290 "Rafailov et al., Direct Preference Optimization (2023)"
[KTO]: https://arxiv.org/abs/2402.01306 "Ethayarajh et al., Kahneman-Tversky Optimization (2024)"
[ORPO]: https://arxiv.org/abs/2403.07691 "Hong et al., ORPO: Monolithic Preference Optimization (2024)"
[GRPO]: https://arxiv.org/abs/2402.03300 "Shao et al., DeepSeekMath / GRPO (2024)"
[CAI]: https://arxiv.org/abs/2212.08073 "Bai et al., Constitutional AI (2022)"
[RLAIF]: https://arxiv.org/abs/2309.00267 "Lee et al., RLAIF (2023)"
[R1]: https://arxiv.org/abs/2501.12948 "DeepSeek-R1 (2025)"
[Adapters]: https://arxiv.org/abs/1902.00751 "Houlsby et al., Parameter-Efficient Transfer Learning (Adapters, 2019)"
[PrefixTuning]: https://arxiv.org/abs/2101.00190 "Li & Liang, Prefix-Tuning (2021)"
[PromptTuning]: https://arxiv.org/abs/2104.08691 "Lester et al., The Power of Scale for Prompt Tuning (2021)"
[LoRA]: https://arxiv.org/abs/2106.09685 "Hu et al., LoRA: Low-Rank Adaptation (2021)"
[QLoRA]: https://arxiv.org/abs/2305.14314 "Dettmers et al., QLoRA (2023)"
[DoRA]: https://arxiv.org/abs/2402.09353 "Liu et al., DoRA: Weight-Decomposed LoRA (2024)"
[LoRAplus]: https://arxiv.org/abs/2402.12354 "Hayou et al., LoRA+ (2024)"
[GaLore]: https://arxiv.org/abs/2403.03507 "Zhao et al., GaLore (2024)"
[IA3]: https://arxiv.org/abs/2205.05638 "Liu et al., Few-Shot PEFT is Better and Cheaper than ICL (IA³, 2022)"
[GPTQ]: https://arxiv.org/abs/2210.17323 "Frantar et al., GPTQ (2022)"
[AWQ]: https://arxiv.org/abs/2306.00978 "Lin et al., AWQ (2023)"
[SmoothQuant]: https://arxiv.org/abs/2211.10438 "Xiao et al., SmoothQuant (2022)"
[KD]: https://arxiv.org/abs/1503.02531 "Hinton et al., Distilling the Knowledge in a Neural Network (2015)"
[ModelSoups]: https://arxiv.org/abs/2203.05482 "Wortsman et al., Model Soups (2022)"
[TIES]: https://arxiv.org/abs/2306.01708 "Yadav et al., TIES-Merging (2023)"
[DARE]: https://arxiv.org/abs/2311.03099 "Yu et al., DARE (2023)"
[CatForget]: https://arxiv.org/abs/1612.00796 "Kirkpatrick et al., Overcoming Catastrophic Forgetting (EWC, 2016)"
[MRL]: https://arxiv.org/abs/2205.13147 "Kusupati et al., Matryoshka Representation Learning (2022)"
[MTEB]: https://arxiv.org/abs/2210.07316 "Muennighoff et al., MTEB (2022)"
[IntrinsicDim]: https://www.nature.com/articles/s41598-017-11873-y "Facco et al., Estimating the intrinsic dimension via TwoNN (2017)"
[IntrinsicDimSGD]: https://arxiv.org/abs/2012.13255 "Aghajanyan et al., Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning (2020)"
[GeometryLLM]: https://arxiv.org/abs/2302.00294 "Valeriani et al., The Geometry of Hidden Representations of LLMs (2023)"
[LLMID]: https://arxiv.org/abs/2405.06915 "Cheng et al., Emergence of a High-Dimensional Abstraction Phase in LLMs (2024)"
[LRH]: https://arxiv.org/abs/2311.03658 "Park et al., The Linear Representation Hypothesis (2023)"
[Superposition]: https://transformer-circuits.pub/2022/toy_model/index.html "Elhage et al., Toy Models of Superposition (Anthropic, 2022)"
[MMLU]: https://arxiv.org/abs/2009.03300 "Hendrycks et al., MMLU (2020)"
[MMLUPro]: https://arxiv.org/abs/2406.01574 "Wang et al., MMLU-Pro (2024)"
[GPQA]: https://arxiv.org/abs/2311.12022 "Rein et al., GPQA (2023)"
[MATH]: https://arxiv.org/abs/2103.03874 "Hendrycks et al., MATH (2021)"
[HumanEval]: https://arxiv.org/abs/2107.03374 "Chen et al., HumanEval / Codex (2021)"
[MBPP]: https://arxiv.org/abs/2108.07732 "Austin et al., MBPP (2021)"
[LiveCodeBench]: https://arxiv.org/abs/2403.07974 "Jain et al., LiveCodeBench (2024)"
[BigCodeBench]: https://arxiv.org/abs/2406.15877 "Zhuo et al., BigCodeBench (2024)"
[Spider]: https://arxiv.org/abs/1809.08887 "Yu et al., Spider text-to-SQL (2018)"
[IFEval]: https://arxiv.org/abs/2311.07911 "Zhou et al., IFEval (2023)"
[ArenaHard]: https://lmsys.org/blog/2024-04-19-arena-hard/ "LMSYS, Arena-Hard (2024)"
[BFCL]: https://gorilla.cs.berkeley.edu/leaderboard.html "Berkeley Function Calling Leaderboard"
[AIME]: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions "AIME competition problems"
[MMMU]: https://arxiv.org/abs/2311.16502 "Yue et al., MMMU (2023)"
[DocVQA]: https://arxiv.org/abs/2007.00398 "Mathew et al., DocVQA (2020)"
[ChartQA]: https://arxiv.org/abs/2203.10244 "Masry et al., ChartQA (2022)"
[MathVista]: https://arxiv.org/abs/2310.02255 "Lu et al., MathVista (2023)"
[OCRBench]: https://arxiv.org/abs/2305.07895 "Liu et al., OCRBench (2023)"
[VideoMME]: https://arxiv.org/abs/2405.21075 "Fu et al., Video-MME (2024)"
[HFTransformers]: https://github.com/huggingface/transformers "Hugging Face Transformers"
[vLLM]: https://github.com/vllm-project/vllm "vLLM"
[SGLang]: https://github.com/sgl-project/sglang "SGLang"
[TGI]: https://github.com/huggingface/text-generation-inference "Text Generation Inference"
[llamacpp]: https://github.com/ggerganov/llama.cpp "llama.cpp (GGUF)"
[MLX]: https://github.com/ml-explore/mlx "Apple MLX"
[ms-swift]: https://github.com/modelscope/ms-swift "Alibaba ms-swift"
[LLaMAFactory]: https://github.com/hiyouga/LLaMA-Factory "LLaMA-Factory"
[Axolotl]: https://github.com/OpenAccess-AI-Collective/axolotl "Axolotl"
[Unsloth]: https://github.com/unslothai/unsloth "Unsloth"
[TRL]: https://github.com/huggingface/trl "Hugging Face TRL"
[DeepSpeed]: https://github.com/microsoft/DeepSpeed "Microsoft DeepSpeed"
[FSDP]: https://pytorch.org/docs/stable/fsdp.html "PyTorch Fully Sharded Data Parallel (FSDP)"
[ChatML]: https://github.com/openai/openai-python/blob/release-v0.28.1/chatml.md "OpenAI ChatML format specification"
[QwenTokenizer]: https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md "Qwen tokenizer (qwen.tiktoken / Qwen2Tokenizer)"
[QwQ]: https://huggingface.co/Qwen/QwQ-32B-Preview "QwQ-32B-Preview model card"
[QVQ]: https://huggingface.co/Qwen/QVQ-72B-Preview "QVQ-72B-Preview model card"
[Alpaca]: https://crfm.stanford.edu/2023/03/13/alpaca.html "Taori et al., Stanford Alpaca (2023)"
[Vicuna]: https://lmsys.org/blog/2023-03-30-vicuna/ "Chiang et al., Vicuna (2023)"
[TIR]: https://arxiv.org/abs/2309.17452 "Gou et al., ToRA: Tool-Integrated Reasoning Agent (2023)"

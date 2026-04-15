from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(kw_only=True)
class QnaBatch:
    """Batch of QnA items prepared for MixedDecoder training.

    Context chunks are encoded by the BERT encoder to produce CLS embeddings.
    These embeddings serve as the context prefix for the autoregressive decoder.
    The prompt encodes the question (and optionally conversation history for multi-turn).
    The answer is the generation target (empty string for unanswerable questions).
    """
    # --- Context ---
    # (total_chunks_in_batch, inp_len) — all context chunk tokens across batch items
    ctx_chunks_toks: torch.Tensor
    # (total_chunks_in_batch, inp_len) — attention masks for context chunks
    ctx_chunks_att_mask: torch.Tensor
    # Number of context chunks per QnA item (list of length batch_size)
    ctx_chunk_counts: List[int]

    # --- Prompt (question / conversation history) ---
    # (batch_size, max_prompt_len) — tokenized prompt, right-padded
    prompt_toks: torch.Tensor
    # (batch_size, max_prompt_len) — attention mask for prompts
    prompt_att_mask: torch.Tensor
    # Actual token lengths of each prompt before padding (list of length batch_size)
    prompt_lengths: List[int]

    # --- Answer ---
    # (batch_size, max_ans_len) — target answer tokens (with special tokens)
    ans_toks: torch.Tensor
    # (batch_size, max_ans_len) — attention mask for answer tokens
    ans_att_mask: torch.Tensor

    # --- Metadata ---
    # Whether each item is answerable (list of length batch_size)
    answerable: Optional[List[bool]] = None

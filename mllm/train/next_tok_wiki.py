"""Next-token prediction batch and dataset iterator for MixedDecoder training with Wikipedia data.

For each Wikipedia article:
1. Tokenize the full article text.
2. Determine how many complete chunks fit while reserving at least ``min_next_toks`` tokens
   for the prediction target.
3. Randomly select a contiguous window of chunks as context (encoded by BERT → CLS embeddings).
4. The tokens immediately following the context window become the autoregressive target.
A fixed "Continue:" prompt bridges context embeddings and target tokens in the decoder input.
"""

from dataclasses import dataclass, field
from enum import Enum
from math import floor
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from mllm.data.wiki.itwiki import get_split_wiki_ds


# In fixed-target mode the prediction target must yield EXACTLY ``fixed_target_toks``
# decoder tokens. Decoder (BPE) tokens can be coarser than encoder (WordPiece) tokens,
# so we reserve a document tail of ``NEXT_TARGET_TAIL_FACTOR * fixed_target_toks``
# encoder tokens to make the uniform random offset draw valid across the whole document
# (otherwise near-end offsets are rejected, biasing the context toward the doc start).
NEXT_TARGET_TAIL_FACTOR = 2


@dataclass(kw_only=True)
class NextTokBatch:
    """Batch of next-token prediction items prepared for MixedDecoder training.

    Context chunks are encoded by the BERT encoder to produce CLS embeddings.
    These embeddings serve as the context prefix for the autoregressive decoder.
    """
    # (total_chunks_in_batch, inp_len) – all context chunk tokens concatenated across batch items
    ctx_chunks_toks: torch.Tensor
    # (total_chunks_in_batch, inp_len) – attention masks for context chunks
    ctx_chunks_att_mask: torch.Tensor
    # Number of context chunks per item (list of length batch_size)
    ctx_chunk_counts: List[int]
    # (batch_size, prompt_len) – tokenized "Continue:" prompt, right-padded
    prompt_toks: torch.Tensor
    # (batch_size, prompt_len) – attention mask for prompts
    prompt_att_mask: torch.Tensor
    # Actual token lengths of each prompt before padding (list of length batch_size)
    prompt_lengths: List[int]
    # (batch_size, max_target_len) – target tokens (with CLS at start, SEP at end, right-padded)
    target_toks: torch.Tensor
    # (batch_size, max_target_len) – attention mask for target tokens
    target_att_mask: torch.Tensor


class NextTokWikiDataset:
    """Dataset that yields :class:`NextTokBatch` from Wikipedia articles.

    Each article is tokenized into sub-word tokens.  A contiguous window of
    ``win_size`` chunks (each ``inp_len`` tokens with CLS/SEP) is selected as
    context, and the subsequent tokens form the prediction target.
    """

    def __init__(
            self, ds: Dataset, inds: np.ndarray, tkz_enc: PreTrainedTokenizer,
            inp_len: int, min_next_toks: int,
            emb_win_min_size: int, emb_win_max_size: int,
            max_target_toks: int = 128, device: Optional[torch.device] = None,
            tkz_dec: Optional[PreTrainedTokenizer] = None,
            fixed_win_size: Optional[int] = None,
            fixed_target_toks: Optional[int] = None,
            deterministic: bool = False,
            text_field: str = 'text',
            prompt: str = 'Continue:',
            seed: Optional[int] = None,
    ):
        self.ds = ds
        self.inds = inds.copy()
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.tkz = tkz_enc  # backward-compat alias
        # Name of the document text column in ``ds`` (varies per corpus, e.g.
        # 'text' for wiki/pg19/gutenberg, 'article' for arXiv, 'report' for GovReport).
        self.text_field = text_field
        self.inp_len = inp_len
        self.min_next_toks = min_next_toks
        self.emb_win_min_size = max(emb_win_min_size, 1)
        self.emb_win_max_size = max(emb_win_max_size, 1)
        self.max_target_toks = max_target_toks
        # --- Controlled-comparison mode (soft-context vs raw-context perplexity) ---
        # When fixed_win_size / fixed_target_toks are set, every emitted item has an
        # identical number of context chunks (=> identical context-token count N) and
        # an identical number of decoder target tokens K. Combined with
        # deterministic=True (in-order document iteration, start=0, no reshuffle),
        # this yields a reproducible sample stream so a soft-context model and a
        # raw-context model can be scored on exactly the same (N, K) inputs.
        self.fixed_win_size = fixed_win_size if (fixed_win_size is None or fixed_win_size > 0) else None
        self.fixed_target_toks = fixed_target_toks if (fixed_target_toks is None or fixed_target_toks > 0) else None
        self.deterministic = deterministic
        self.device = device if device is not None else torch.device('cpu')
        self.enc_pad_token_id = tkz_enc.pad_token_id
        self.dec_pad_token_id = self.tkz_dec.pad_token_id
        if self.dec_pad_token_id is None:
            self.dec_pad_token_id = self.tkz_dec.eos_token_id
        self.cls_token_id = tkz_enc.cls_token_id
        self.sep_token_id = tkz_enc.sep_token_id
        self.chunk_content_len = inp_len - 2  # reserve 1 for CLS and 1 for SEP
        self.size = len(self.inds)
        # Decoder-side specials for wrapping the target sequence.
        if self.tkz_dec.cls_token_id is not None and self.tkz_dec.sep_token_id is not None:
            # BERT-like decoder: wrap with CLS .. SEP.
            self.dec_target_prefix: List[int] = [self.tkz_dec.cls_token_id]
            self.dec_target_suffix: List[int] = [self.tkz_dec.sep_token_id]
        elif self.tkz_dec.eos_token_id is not None:
            # GPT-2: only end-of-text token; use as target ending.
            self.dec_target_prefix = []
            self.dec_target_suffix = [self.tkz_dec.eos_token_id]
        else:
            self.dec_target_prefix = []
            self.dec_target_suffix = []
        # Reserve room for prefix+suffix in the target budget.
        self.dec_target_content_budget = max(
            self.max_target_toks - len(self.dec_target_prefix) - len(self.dec_target_suffix), 1,
        )
        # Fixed prompt tokenized once with the decoder tokenizer. An empty prompt
        # string yields no prompt tokens => pure [context | target] layout (both the
        # soft and decoder-only model paths already handle prompt_len == 0).
        self.prompt = prompt
        self.prompt_toks_fixed: List[int] = (
            self.tkz_dec(prompt, add_special_tokens=False).input_ids if prompt else []
        )
        # Internal pointer for skipping short docs
        self._ptr = 0
        # Per-dataset RNG so context-offset, window-size and shuffle draws are
        # reproducible and independent of the global numpy RNG (seeded per source
        # via shuffle(seed) / build_stacked_next_tok_datasets).
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _tokenize_doc(self, text: str) -> List[int]:
        return self.tkz_enc(text, add_special_tokens=False).input_ids

    def _make_chunks(self, doc_toks: List[int], start: int, win_size: int) -> List[List[int]]:
        """Split ``win_size`` consecutive content-length segments into CLS+content+SEP chunks."""
        chunks: List[List[int]] = []
        for i in range(win_size):
            seg_start = start + i * self.chunk_content_len
            seg_end = seg_start + self.chunk_content_len
            content = doc_toks[seg_start:seg_end]
            chunk = [self.cls_token_id] + content + [self.sep_token_id]
            chunks.append(chunk)
        return chunks

    def _sample_item(self) -> Optional[Tuple[List[List[int]], List[int]]]:
        """Try to sample one valid item from the current pointer position.

        Returns ``None`` if the current document is too short (caller should
        advance the pointer and retry).
        """
        idx = self.inds[self._ptr].item()
        row = self.ds[idx]
        text: str = row[self.text_field]
        doc_toks = self._tokenize_doc(text)
        doc_toks_num = len(doc_toks)

        # Tail budget reserved for the prediction target after the context window.
        # Reserving only min_next_toks biased the random offset toward the document
        # start in fixed-target mode (near-end offsets couldn't fill K decoder tokens
        # and were rejected). Reserve a K-proportional encoder-token tail so the
        # uniform offset draw is valid across the whole document.
        if self.fixed_target_toks is not None:
            required_tail = max(self.min_next_toks, NEXT_TARGET_TAIL_FACTOR * self.fixed_target_toks)
        else:
            required_tail = self.min_next_toks

        doc_chunks_num = max(floor((doc_toks_num - required_tail) / self.chunk_content_len), 0)

        # Window size: fixed (controlled mode) or randomized within [min, max].
        if self.fixed_win_size is not None:
            if doc_chunks_num < self.fixed_win_size:
                return None
            win_size = self.fixed_win_size
        else:
            if doc_chunks_num < self.emb_win_min_size:
                return None
            win_max_size_new = min(doc_chunks_num, self.emb_win_max_size)
            win_size = int(self._rng.integers(self.emb_win_min_size, win_max_size_new + 1))

        ctx_tok_count = win_size * self.chunk_content_len
        max_start = doc_toks_num - ctx_tok_count - required_tail
        if max_start < 0:
            return None
        # Deterministic mode pins the context to the document start so the sample
        # stream is reproducible across models; otherwise pick a random window.
        start = 0 if self.deterministic else int(self._rng.integers(0, max_start + 1))

        chunks = self._make_chunks(doc_toks, start, win_size)

        # Target: tokens right after the context window. We slice the encoder
        # tokenization, decode to text, and re-tokenize with the decoder vocab.
        tgt_start = start + ctx_tok_count
        if self.fixed_target_toks is not None:
            # Controlled mode: emit EXACTLY fixed_target_toks decoder content tokens.
            # Over-slice on the encoder side (decoder tokens are usually coarser),
            # then require enough continuation to fill K; otherwise skip the doc.
            k = self.fixed_target_toks
            tgt_end = min(tgt_start + k * 8, doc_toks_num)
            tgt_text = self.tkz_enc.decode(doc_toks[tgt_start:tgt_end], skip_special_tokens=True)
            tgt_content = self.tkz_dec(tgt_text, add_special_tokens=False).input_ids
            if len(tgt_content) < k:
                return None
            tgt_content = tgt_content[:k]
        else:
            tgt_end = min(tgt_start + self.max_target_toks * 4, doc_toks_num)
            tgt_text = self.tkz_enc.decode(doc_toks[tgt_start:tgt_end], skip_special_tokens=True)
            tgt_content = self.tkz_dec(tgt_text, add_special_tokens=False).input_ids
            tgt_content = tgt_content[:self.dec_target_content_budget]
        target = self.dec_target_prefix + tgt_content + self.dec_target_suffix

        return chunks, target

    def _advance_ptr(self) -> None:
        self._ptr += 1
        if self._ptr >= self.size:
            # Deterministic mode keeps a stable document order so the emitted
            # sample stream is identical across evaluation runs / models.
            if not self.deterministic:
                self._rng.shuffle(self.inds)
            self._ptr = 0

    def _sample_valid_item(self) -> Tuple[List[List[int]], List[int]]:
        """Keep sampling until a document long enough is found."""
        while True:
            result = self._sample_item()
            self._advance_ptr()
            if result is not None:
                return result

    def get_batch(self, batch_size: int) -> NextTokBatch:
        all_chunks: List[List[int]] = []
        chunk_counts: List[int] = []
        prompt_toks_list: List[List[int]] = []
        target_toks_list: List[List[int]] = []

        for _ in range(batch_size):
            chunks, target = self._sample_valid_item()
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
            prompt_toks_list.append(list(self.prompt_toks_fixed))
            target_toks_list.append(target)

        # Pad context chunks to inp_len (encoder vocab)
        total_chunks = len(all_chunks)
        ctx_chunks_t = torch.full((total_chunks, self.inp_len), self.enc_pad_token_id, dtype=torch.long, device=self.device)
        ctx_chunks_att = torch.zeros((total_chunks, self.inp_len), dtype=torch.long, device=self.device)
        for i, chunk in enumerate(all_chunks):
            n = min(len(chunk), self.inp_len)
            ctx_chunks_t[i, :n] = torch.tensor(chunk[:n], dtype=torch.long, device=self.device)
            ctx_chunks_att[i, :n] = 1

        # Right-pad prompts (decoder vocab; fixed length, but keep interface consistent)
        prompt_lengths = [len(p) for p in prompt_toks_list]
        max_prompt_len = max(prompt_lengths)
        prompt_t = torch.full((batch_size, max_prompt_len), self.dec_pad_token_id, dtype=torch.long, device=self.device)
        prompt_att = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(prompt_toks_list):
            n = len(toks)
            prompt_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            prompt_att[i, :n] = 1

        # Pad target tokens (decoder vocab)
        max_target_len = max(len(t) for t in target_toks_list)
        target_t = torch.full((batch_size, max_target_len), self.dec_pad_token_id, dtype=torch.long, device=self.device)
        target_att = torch.zeros((batch_size, max_target_len), dtype=torch.long, device=self.device)
        for i, toks in enumerate(target_toks_list):
            n = len(toks)
            target_t[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            target_att[i, :n] = 1

        return NextTokBatch(
            ctx_chunks_toks=ctx_chunks_t,
            ctx_chunks_att_mask=ctx_chunks_att,
            ctx_chunk_counts=chunk_counts,
            prompt_toks=prompt_t,
            prompt_att_mask=prompt_att,
            prompt_lengths=prompt_lengths,
            target_toks=target_t,
            target_att_mask=target_att,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'NextTokWikiDataset':
        if seed is not None:
            # Reseed the per-dataset RNG so subsequent offset/window/shuffle draws
            # are reproducible from this seed.
            self._rng = np.random.default_rng(seed)
        self._rng.shuffle(self.inds)
        self._ptr = 0
        return self


def load_split_wiki_for_next(
        data_path: Path, val_ratio: float = 0.05, random_seed: int = 100,
) -> Tuple[Dataset, np.ndarray, np.ndarray]:
    """Load Wikipedia and split into train/val index arrays.

    Returns:
        Tuple of (ds, inds_train, inds_val) where *ds* is the HuggingFace
        ``Dataset`` object and the index arrays select documents for each split.
    """
    ds, inds_train, inds_val = get_split_wiki_ds(data_path, val_ratio=val_ratio, rand_seed=random_seed)
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Wikipedia for NextTok: n_total={len(ds)}. n_train={len(inds_train)}. n_val={len(inds_val)}.')
    return ds, inds_train, inds_val


def create_next_tok_dataloader(
        dataset: 'NextTokWikiDataset | StackedNextTokDataset', batch_size: int,
) -> Generator[NextTokBatch, None, None]:
    """Create infinite-loop generator yielding :class:`NextTokBatch` instances."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create NextTokWikiDataset dataloader. batch_size={batch_size}.')
    while True:
        batch = dataset.get_batch(batch_size)
        yield batch


# Backward-compatible corpus-agnostic alias.
NextTokDataset = NextTokWikiDataset


@dataclass(frozen=True)
class NextTokSourceSpec:
    """Static description of a long-document corpus usable for next-token training."""
    hf_id: str                       # HuggingFace dataset id
    hf_config: Optional[str]         # dataset config / subset name (None if not needed)
    split: str                       # split holding the documents (usually 'train')
    text_field: str                  # column name holding the document text


# Registry of supported next-token corpora. Verify exact ids/fields against the
# HuggingFace Hub before large downloads; centralised here so additions are one line.
SOURCE_REGISTRY: Dict[str, NextTokSourceSpec] = {
    'wiki': NextTokSourceSpec('wikimedia/wikipedia', '20231101.en', 'train', 'text'),
    'pg19': NextTokSourceSpec('deepmind/pg19', None, 'train', 'text'),
    'bookcorpusopen': NextTokSourceSpec('lucadiliello/bookcorpusopen', None, 'train', 'text'),
    'arxiv': NextTokSourceSpec('ccdv/arxiv-summarization', None, 'train', 'article'),
    'govreport': NextTokSourceSpec('ccdv/govreport-summarization', None, 'train', 'report'),
    'gutenberg': NextTokSourceSpec('manu/project_gutenberg', None, 'en', 'text'),
}


def load_split_source_for_next(
        source: str, data_path: Path, val_ratio: float = 0.05, random_seed: int = 100,
) -> Tuple[Dataset, np.ndarray, np.ndarray, str]:
    """Load a registered corpus and split it into train/val document indices.

    Wikipedia reuses the existing dedicated loader; every other source is loaded
    generically from the HuggingFace Hub via :data:`SOURCE_REGISTRY`.

    Returns:
        Tuple of (ds, inds_train, inds_val, text_field).
    """
    if source not in SOURCE_REGISTRY:
        raise ValueError(
            f'Unknown next-token source {source!r}. Known: {sorted(SOURCE_REGISTRY)}'
        )
    spec = SOURCE_REGISTRY[source]
    rank = dist.get_rank() if dist.is_initialized() else 0
    if source == 'wiki':
        ds, inds_train, inds_val = load_split_wiki_for_next(
            data_path, val_ratio=val_ratio, random_seed=random_seed,
        )
        return ds, inds_train, inds_val, spec.text_field

    dss = load_dataset(
        spec.hf_id, spec.hf_config, cache_dir=str(data_path), trust_remote_code=True,
    )
    ds = dss[spec.split]
    n_docs = len(ds)
    doc_inds = np.arange(n_docs)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(doc_inds)
    n_docs_val = int(n_docs * val_ratio)
    n_docs_train = n_docs - n_docs_val
    inds_train = doc_inds[:n_docs_train].copy()
    inds_val = doc_inds[n_docs_train:].copy()
    print(
        f'R{rank}. Source {source!r} ({spec.hf_id}) for NextTok: '
        f'n_total={n_docs}. n_train={len(inds_train)}. n_val={len(inds_val)}.'
    )
    return ds, inds_train, inds_val, spec.text_field


class StackedNextTokDataset:
    """Stack several :class:`NextTokDataset` sub-datasets behind one interface.

    Each :meth:`get_batch` picks exactly one sub-dataset (so a batch is drawn from a
    single homogeneous corpus) with probability proportional to that source's split
    size, then delegates to it. This keeps every emitted object a plain
    :class:`NextTokBatch`, so the model and dataloader are unchanged.
    """

    def __init__(
            self, datasets: Sequence[NextTokDataset], names: Optional[Sequence[str]] = None,
            weights: Optional[Sequence[float]] = None, seed: Optional[int] = None,
    ):
        if not datasets:
            raise ValueError('StackedNextTokDataset requires at least one sub-dataset.')
        self.datasets: List[NextTokDataset] = list(datasets)
        self.names: List[str] = (
            list(names) if names is not None else [f'src{i}' for i in range(len(self.datasets))]
        )
        if weights is None:
            weights_arr = np.array([float(len(d)) for d in self.datasets], dtype=np.float64)
        else:
            weights_arr = np.array([float(w) for w in weights], dtype=np.float64)
        total = weights_arr.sum()
        if total <= 0:
            weights_arr = np.ones(len(self.datasets), dtype=np.float64)
            total = weights_arr.sum()
        self.weights = weights_arr / total
        self._rng = np.random.default_rng(seed)
        self.device = self.datasets[0].device

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def get_batch(self, batch_size: int) -> NextTokBatch:
        idx = int(self._rng.choice(len(self.datasets), p=self.weights))
        return self.datasets[idx].get_batch(batch_size)

    def shuffle(self, seed: Optional[int] = None) -> 'StackedNextTokDataset':
        for i, d in enumerate(self.datasets):
            d.shuffle(None if seed is None else seed + i)
        return self


def build_stacked_next_tok_datasets(
        sources: Sequence[str],
        sources_data: Dict[str, Tuple[Dataset, np.ndarray, np.ndarray, str]],
        split: str,
        tkz_enc: PreTrainedTokenizer, inp_len: int, min_next_toks: int,
        emb_win_min_size: int, emb_win_max_size: int,
        max_target_toks: int = 128, device: Optional[torch.device] = None,
        tkz_dec: Optional[PreTrainedTokenizer] = None,
        fixed_win_size: Optional[int] = None,
        fixed_target_toks: Optional[int] = None,
        deterministic: bool = False, prompt: str = 'Continue:',
        seed: Optional[int] = None,
) -> StackedNextTokDataset:
    """Build a :class:`StackedNextTokDataset` for the requested *sources*.

    *sources_data* maps each source name to the tuple returned by
    :func:`load_split_source_for_next` (ds, inds_train, inds_val, text_field).
    *split* selects which index array to use: ``'train'`` or ``'val'``.
    """
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    sub_datasets: List[NextTokDataset] = []
    names: List[str] = []
    weights: List[float] = []
    for src in sources:
        if src not in sources_data:
            raise ValueError(f'Source {src!r} requested but not present in sources_data.')
        ds, inds_train, inds_val, text_field = sources_data[src]
        inds = inds_train if split == 'train' else inds_val
        sub = NextTokDataset(
            ds=ds, inds=inds, tkz_enc=tkz_enc, inp_len=inp_len,
            min_next_toks=min_next_toks, emb_win_min_size=emb_win_min_size,
            emb_win_max_size=emb_win_max_size, max_target_toks=max_target_toks,
            device=device, tkz_dec=tkz_dec, fixed_win_size=fixed_win_size,
            fixed_target_toks=fixed_target_toks, deterministic=deterministic,
            text_field=text_field, prompt=prompt,
            seed=(None if seed is None else seed + len(sub_datasets)),
        )
        sub_datasets.append(sub)
        names.append(src)
        weights.append(float(len(inds)))
    return StackedNextTokDataset(sub_datasets, names=names, weights=weights, seed=seed)

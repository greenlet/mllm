"""Shared primitives for the synthetic structured-extraction datasets.

All five structured-extraction builders (keyval / jsonfield / jsonata / xmlxpath /
sql) plus the ``cite`` builder used to duplicate the same scaffolding:

* split WordPiece ids into whole words,
* a random "special-token pool" used to top up short articles,
* encoder-vocab -> decoder-vocab re-tokenization,
* the final ``TokensSubsetV2`` packaging (CLS/SEP, decoder-vocab mirror, end token),
* the ``Dataset`` wrapper (``_to_tensor`` / ``get_batch`` / ``shuffle``) and the
  round-robin dataloader generator.

This module centralizes those pieces so each builder can focus on its own
*spec -> realize* logic. The key design rule for testability:

    **All randomness lives in a ``sample_*_spec`` step. ``realize_*`` is a pure,
    deterministic function of (spec, WordFeed).**

``WordFeed`` is the deterministic content source: words are consumed
*sequentially* (not sampled by random position), with a deterministic pool
fallback when the article runs out.
"""
from dataclasses import dataclass
import json
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.encdec_graph_bert import MaskedCiteBatch


# ---------------------------------------------------------------------------
# Word splitting + deterministic content feed
# ---------------------------------------------------------------------------

def split_words(ids: List[int], tkz_enc: PreTrainedTokenizer) -> List[List[int]]:
    """Group WordPiece ids into whole words.

    A word = a non-``##`` token followed by its ``##`` continuations.
    """
    toks = tkz_enc.convert_ids_to_tokens(ids)
    words: List[List[int]] = []
    cur: List[int] = []
    for tid, tok in zip(ids, toks):
        if tok.startswith('##') and cur:
            cur.append(tid)
        else:
            if cur:
                words.append(cur)
            cur = [tid]
    if cur:
        words.append(cur)
    return words


def make_pool(rng: np.random.Generator, n_special_toks: int, vocab_size: int) -> np.ndarray:
    """A shuffled pool of non-special token ids. The shuffle (randomness) is done
    here, OUTSIDE item realization; the pool is then consumed deterministically."""
    return rng.permutation(np.arange(n_special_toks, vocab_size))


class WordFeed:
    """Deterministic, sequential provider of real words drawn from one article.

    Holds parallel ``ids`` / ``text`` lists plus a precomputed special-token
    ``pool``. Every accessor advances a cursor; when the article is exhausted it
    falls back to the pool deterministically (no RNG inside).
    """

    def __init__(
            self, words: List[List[int]], texts: List[str], pool: Sequence[int],
            tkz_enc: PreTrainedTokenizer,
    ):
        assert len(words) == len(texts)
        self.words = words
        self.texts = texts
        self.pool = list(pool)
        self.tkz_enc = tkz_enc
        self.wi = 0   # word cursor
        self.pi = 0   # pool cursor

    # -- pool --------------------------------------------------------------
    def next_pool_token(self) -> int:
        if not self.pool:
            return 0
        tid = int(self.pool[self.pi])
        self.pi += 1
        if self.pi >= len(self.pool):
            self.pi = 0
        return tid

    def _pool_word(self) -> Tuple[List[int], str]:
        tid = self.next_pool_token()
        txt = self.tkz_enc.decode([tid], skip_special_tokens=True).strip()
        if not txt:
            txt = f'v{tid}'
        return [tid], txt

    # -- words -------------------------------------------------------------
    def reset(self) -> None:
        """Rewind both cursors. Used by retry-until-fits builders so each
        attempt consumes the same words from the start (still deterministic)."""
        self.wi = 0
        self.pi = 0

    def remaining(self) -> int:
        return max(0, len(self.words) - self.wi)

    def next_word(self) -> Tuple[List[int], str]:
        """One word (ids, text). Pool fallback when exhausted."""
        if self.wi < len(self.words):
            ids, txt = self.words[self.wi], self.texts[self.wi]
            self.wi += 1
            return list(ids), txt
        return self._pool_word()

    def next_distinct_word(
            self, used: set, is_valid=None, max_scan: int = 64,
    ) -> Tuple[List[int], str]:
        """Next word whose lowercased text is unused (and passes ``is_valid``).

        Deterministic forward scan; pool fallback if none qualifies within
        ``max_scan`` words. ``used`` is updated with the chosen text.
        """
        scanned = 0
        while self.wi < len(self.words) and scanned < max_scan:
            ids, txt = self.words[self.wi], self.texts[self.wi]
            self.wi += 1
            scanned += 1
            low = txt.strip().lower()
            if low and low not in used and (is_valid is None or is_valid(txt)):
                used.add(low)
                return list(ids), txt
        ids, txt = self._pool_word()
        used.add(txt.lower())
        return ids, txt

    def next_span(self, n_words: int) -> Tuple[List[int], str]:
        """Concatenated ids of the next ``n_words`` words, plus the decoded text
        of that concatenation (so it matches what the encoder will see)."""
        n_words = max(1, n_words)
        ids: List[int] = []
        for _ in range(n_words):
            wids, _ = self.next_word()
            ids.extend(wids)
        txt = self.tkz_enc.decode(ids, skip_special_tokens=True).strip()
        if not txt:
            txt = f'v{ids[0] if ids else 0}'
        return ids, txt


def word_feed_from_text(
        text: str, tkz_enc: PreTrainedTokenizer, pool: Sequence[int],
) -> Tuple[WordFeed, List[int]]:
    """Build a :class:`WordFeed` from raw text. Returns (feed, src_ids)."""
    ids_src = tkz_enc(text, add_special_tokens=False).input_ids
    words = split_words(ids_src, tkz_enc)
    texts = [tkz_enc.decode(w, skip_special_tokens=True).strip() for w in words]
    feed = WordFeed(words, texts, pool, tkz_enc)
    return feed, ids_src


# ---------------------------------------------------------------------------
# Encoder<->decoder vocab bridge
# ---------------------------------------------------------------------------

def enc_to_dec(
        enc_ids: List[int], tkz_enc: PreTrainedTokenizer, tkz_dec: PreTrainedTokenizer,
        same_vocab: bool,
) -> List[int]:
    if same_vocab:
        return list(enc_ids)
    if not enc_ids:
        return []
    text = tkz_enc.decode(enc_ids, skip_special_tokens=True)
    return tkz_dec(text, add_special_tokens=False).input_ids


def dec_end_token_id(tkz_dec: PreTrainedTokenizer) -> Optional[int]:
    return tkz_dec.sep_token_id if tkz_dec.sep_token_id is not None else tkz_dec.eos_token_id


# ---------------------------------------------------------------------------
# Composite-aware value serialization
# ---------------------------------------------------------------------------

# JSON separator policies, mirrored between the record and the serialized target
# so that a composite (object/array) answer remains a verbatim substring of the
# record (the needle-in-context invariant).
JSON_SEPS_COMPACT = (',', ':')
JSON_SEPS_SPACED = (', ', ': ')


def json_seps(compact: bool) -> Tuple[str, str]:
    return JSON_SEPS_COMPACT if compact else JSON_SEPS_SPACED


def json_target_text(value, compact: bool) -> str:
    """Serialize a JSON value for the decoder target.

    * A scalar ``str`` is returned raw (no surrounding quotes).
    * Every other value (int / bool / None / list / dict) is ``json.dumps``-ed
      with the RECORD's separator policy, so composite targets stay grounded.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, separators=json_seps(compact))


# ---------------------------------------------------------------------------
# Shared JSON shape: content-agnostic spec + deterministic materialization
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class JsonNodeSpec:
    """One node of a content-agnostic JSON shape tree (shared by jsonfield/jsonata)."""
    kind: str  # 'text' | 'int' | 'bool' | 'null' | 'array' | 'object'
    word_count: int = 0          # kind == 'text'
    int_lit: int = 0             # kind == 'int'
    bool_lit: bool = False       # kind == 'bool'
    elems: Optional[List['JsonNodeSpec']] = None        # kind == 'array'
    array_query_idx: int = 0     # kind == 'array'
    fields: Optional[List['JsonNodeSpec']] = None       # kind == 'object'


def sample_json_shape(rng: np.random.Generator, cfg) -> JsonNodeSpec:
    """Sample a JSON shape (no content). ``cfg`` must expose ``min_fields``,
    ``max_fields``, ``max_depth``, ``max_array_len``, ``value_min_words``,
    ``value_max_words``, ``nested_prob``, ``array_prob``, ``number_prob``,
    ``bool_null_prob``. All randomness lives here."""
    state = {'nf': int(rng.integers(cfg.min_fields, cfg.max_fields + 1))}

    def scalar() -> JsonNodeSpec:
        r = float(rng.random())
        if r < cfg.bool_null_prob * 0.5:
            return JsonNodeSpec(kind='bool', bool_lit=bool(rng.integers(2)))
        if r < cfg.bool_null_prob:
            return JsonNodeSpec(kind='null')
        if r < cfg.bool_null_prob + cfg.number_prob:
            return JsonNodeSpec(kind='int', int_lit=int(rng.integers(0, 1_000_000)))
        n = int(rng.integers(cfg.value_min_words, cfg.value_max_words + 1))
        return JsonNodeSpec(kind='text', word_count=n)

    def obj(depth: int) -> JsonNodeSpec:
        fields: List[JsonNodeSpec] = []
        while state['nf'] > 0:
            state['nf'] -= 1
            r = float(rng.random())
            can_nest = depth < cfg.max_depth and state['nf'] > 0
            if can_nest and r < cfg.nested_prob:
                fields.append(obj(depth + 1))
            elif r < cfg.nested_prob + cfg.array_prob:
                arr_len = int(rng.integers(1, cfg.max_array_len + 1))
                elems = [scalar() for _ in range(arr_len)]
                fields.append(JsonNodeSpec(
                    kind='array', elems=elems, array_query_idx=int(rng.integers(arr_len)),
                ))
            else:
                fields.append(scalar())
            if float(rng.random()) < 0.35:
                break
        if not fields:
            fields.append(scalar())
        return JsonNodeSpec(kind='object', fields=fields)

    return obj(1)


def normalize_key(txt: str) -> str:
    out = []
    for ch in txt.lower():
        if ch.isalnum() or ch == '_':
            out.append(ch)
        elif ch in ('-', ' '):
            out.append('_')
    return ''.join(out).strip('_')


def is_json_key(key: str, min_chars: int) -> bool:
    return len(key) >= min_chars and any(c.isalpha() for c in key)


def next_json_key(feed: WordFeed, used_norm: set, min_key_chars: int) -> str:
    """Next normalized, unused JSON key drawn sequentially from the feed."""
    for _ in range(64):
        _, txt = feed.next_word()
        k = normalize_key(txt)
        if is_json_key(k, min_key_chars) and k not in used_norm:
            used_norm.add(k)
            return k
    for _ in range(256):
        tid = feed.next_pool_token()
        k = normalize_key(feed.tkz_enc.decode([tid], skip_special_tokens=True))
        if not is_json_key(k, min_key_chars):
            k = f'k_{tid}'
        if k not in used_norm:
            used_norm.add(k)
            return k
    k = f'k_{len(used_norm)}'
    used_norm.add(k)
    return k


def materialize_json(
        node: JsonNodeSpec, feed: WordFeed, min_key_chars: int, path: List,
        leaves: List[Tuple[List, object]], arrays: List[Tuple[List, list]],
        composites: List[Tuple[List, object]],
):
    """Deterministically build a JSON value from a shape spec + word feed.

    Records, in-place: ``leaves`` (scalar/array-element targets), ``arrays``
    (array path/value pairs for transforms), ``composites`` (non-root object /
    array subtrees for composite targets).
    """
    k = node.kind
    if k == 'text':
        _, txt = feed.next_span(node.word_count)
        return txt
    if k == 'int':
        return node.int_lit
    if k == 'bool':
        return node.bool_lit
    if k == 'null':
        return None
    if k == 'array':
        arr = [
            materialize_json(en, feed, min_key_chars, path + [i], leaves, arrays, composites)
            for i, en in enumerate(node.elems or [])
        ]
        if arr:
            qi = node.array_query_idx % len(arr)
            leaves.append((path + [qi], arr[qi]))
            arrays.append((list(path), arr))
            if path:
                composites.append((list(path), arr))
        return arr
    d: dict = {}
    used: set = set()
    for child in (node.fields or []):
        key = next_json_key(feed, used, min_key_chars)
        v = materialize_json(child, feed, min_key_chars, path + [key], leaves, arrays, composites)
        d[key] = v
        if child.kind in ('text', 'int', 'bool', 'null'):
            leaves.append((path + [key], v))
    if path and d:
        composites.append((list(path), d))
    return d


def build_json_value(node: JsonNodeSpec, feed: WordFeed, min_key_chars: int):
    """Convenience wrapper returning ``(value, leaves, arrays, composites)``."""
    leaves: List[Tuple[List, object]] = []
    arrays: List[Tuple[List, list]] = []
    composites: List[Tuple[List, object]] = []
    value = materialize_json(node, feed, min_key_chars, [], leaves, arrays, composites)
    return value, leaves, arrays, composites


def json_path_strings(path: List) -> Tuple[str, str]:
    """(JSONPath ``$.a.b[0]``, dotted ``a.b[0]``) for a path of keys/indices."""
    jsonpath = '$'
    dotted_parts: List[str] = []
    for p in path:
        if isinstance(p, int):
            jsonpath += f'[{p}]'
            if dotted_parts:
                dotted_parts[-1] = f'{dotted_parts[-1]}[{p}]'
            else:
                dotted_parts.append(f'[{p}]')
        else:
            jsonpath += f'.{p}'
            dotted_parts.append(str(p))
    return jsonpath, '.'.join(dotted_parts)


def path_to_dot(path: List) -> str:
    parts: List[str] = []
    for p in path:
        if isinstance(p, int):
            if parts:
                parts[-1] = f'{parts[-1]}[{p}]'
            else:
                parts.append(f'[{p}]')
        else:
            parts.append(str(p))
    return '.'.join(parts)


def path_to_jq(path: List) -> str:
    out = '.'
    for p in path:
        if isinstance(p, int):
            out += f'[{p}]'
        else:
            out += p if out == '.' else f'.{p}'
    return out


def sql_row_text(cells: Sequence) -> str:
    """Serialize one SQL row (composite SELECT *) as a comma-joined tuple."""
    return ', '.join(str(c) for c in cells)


def sql_rows_text(rows: Sequence[Sequence]) -> str:
    """Serialize multiple SQL rows: ``;``-joined rows of ``,``-joined cells."""
    return '; '.join(sql_row_text(r) for r in rows)


# ---------------------------------------------------------------------------
# Final TokensSubsetV2 packaging (shared by every structured builder)
# ---------------------------------------------------------------------------

def assemble_subset(
        *,
        ids_src: List[int],
        rec_ids: List[int],
        prompt: str,
        prompt_toks: List[int],
        tkz_enc: PreTrainedTokenizer,
        tkz_dec: PreTrainedTokenizer,
        same_vocab: bool,
        end_token_id: Optional[int],
        target_text: Optional[str] = None,
        target_enc: Optional[List[int]] = None,
        target_dec: Optional[List[int]] = None,
) -> TokensSubsetV2:
    """Pack a structured-extraction sample into a :class:`TokensSubsetV2`.

    Either pass ``target_text`` (it is tokenized with both vocabs) or provide
    precomputed ``target_enc`` / ``target_dec`` token lists (used by keyval,
    whose target is the exact value token run rather than a re-tokenized string).
    """
    if target_enc is None:
        assert target_text is not None
        target_enc = tkz_enc(target_text, add_special_tokens=False).input_ids
    if target_dec is None:
        if target_text is not None:
            target_dec = tkz_dec(target_text, add_special_tokens=False).input_ids
        else:
            target_dec = enc_to_dec(target_enc, tkz_enc, tkz_dec, same_vocab)

    end_suffix = [end_token_id] if end_token_id is not None else []
    cites_dec = list(target_dec) + end_suffix
    inp_dec = enc_to_dec(rec_ids, tkz_enc, tkz_dec, same_vocab) + end_suffix

    cls_id = tkz_enc.cls_token_id
    sep_id = tkz_enc.sep_token_id
    toks_inp = [cls_id] + rec_ids + [sep_id]

    return TokensSubsetV2(
        toks_src=ids_src,
        inp_beg_ind=0,
        inp_end_ind=len(ids_src),
        toks_inp=toks_inp,
        toks_inp_masked=list(toks_inp),
        cite_beg_ind=-1,
        cite_end_ind=-1,
        toks_cite=list(target_enc),
        toks_cite_masked=list(target_enc),
        toks_cite_beg=[],
        toks_cite_end=[],
        prompt=prompt,
        toks_prompt=prompt_toks,
        toks_inp_dec=inp_dec,
        toks_inp_masked_dec=list(inp_dec),
        toks_cite_dec=cites_dec,
        toks_cite_masked_dec=list(cites_dec),
    )


# ---------------------------------------------------------------------------
# Retry-until-fits / fill-to-budget build loop (shared by jsonfield/jsonata/
# xmlxpath/sql builders)
# ---------------------------------------------------------------------------

def build_to_budget(
        *,
        sample_spec: Callable[[], object],
        realize: Callable[[object, 'WordFeed', Optional[List[int]]], Tuple[TokensSubsetV2, int]],
        feed: 'WordFeed',
        ids_src: List[int],
        budget: int,
        n_attempts: int,
        fill_to_budget: bool = False,
        fill_frac: float = 0.85,
) -> TokensSubsetV2:
    """Sample/realize repeatedly and pick a record honoring the token budget.

    ``realize`` returns ``(subset, rec_len)`` where ``rec_len`` is the realized
    record length in encoder tokens (before CLS/SEP). The feed is rewound before
    each attempt so every attempt is deterministic given its spec.

    * ``fill_to_budget=False`` (default): legacy first-fit -- return the first
      attempt with ``rec_len <= budget`` (falls back to the smallest overflow).
    * ``fill_to_budget=True``: best-fit -- early-return as soon as an attempt
      reaches ``fill_frac * budget`` (and still fits); otherwise keep the
      *largest* record that fits, so chunks pack close to ``budget``.
    """
    target = int(budget * fill_frac) if fill_to_budget else 0
    best_fit: Optional[Tuple[TokensSubsetV2, int]] = None   # largest rec_len <= budget
    best_over: Optional[Tuple[TokensSubsetV2, int]] = None  # smallest rec_len > budget
    for _ in range(max(1, n_attempts)):
        feed.reset()
        spec = sample_spec()
        subset, rec_len = realize(spec, feed, ids_src)
        if rec_len <= budget:
            if rec_len >= target:
                return subset
            if best_fit is None or rec_len > best_fit[1]:
                best_fit = (subset, rec_len)
        elif best_over is None or rec_len < best_over[1]:
            best_over = (subset, rec_len)
    if best_fit is not None:
        return best_fit[0]
    assert best_over is not None
    return best_over[0]


# ---------------------------------------------------------------------------
# Base dataset + dataloader (shared wrapper)
# ---------------------------------------------------------------------------

class BaseRecallDataset:
    """Wraps a HuggingFace ``text`` dataset and yields :class:`MaskedCiteBatch`.

    Subclasses construct ``self.builder`` (a callable ``list[str] -> list[
    TokensSubsetV2]``) and call ``super().__init__``.
    """

    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            builder, device: Optional[torch.device] = None,
            tkz_dec: Optional[PreTrainedTokenizer] = None,
    ):
        self.dataset = dataset
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.size = len(dataset)
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')
        self.enc_pad_token_id = tkz_enc.pad_token_id
        self.dec_pad_token_id = self.tkz_dec.pad_token_id
        if self.dec_pad_token_id is None:
            self.dec_pad_token_id = self.tkz_dec.eos_token_id
        self.inds = np.arange(self.size)
        self.builder = builder

    def __len__(self):
        return self.size

    def _to_tensor(self, toks_list: List[List[int]], pad_token_id: int, with_att_mask: bool):
        batch_size = len(toks_list)
        max_len = max((len(t) for t in toks_list), default=1)
        max_len = max(max_len, 1)
        out = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=self.device)
        att = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device) if with_att_mask else None
        for i, toks in enumerate(toks_list):
            n = len(toks)
            if n:
                out[i, :n] = torch.tensor(toks, dtype=torch.long, device=self.device)
            if att is not None:
                att[i, :n] = 1
        return out, att

    def get_batch(self, inds: List[int]) -> MaskedCiteBatch:
        texts = [self.dataset[int(i)]['text'] for i in inds]
        subsets = self.builder(texts)
        batch_size = len(inds)
        enc_pad, dec_pad = self.enc_pad_token_id, self.dec_pad_token_id

        inp_toks, inp_att_mask = self._to_tensor([s.toks_inp for s in subsets], enc_pad, True)
        inp_masked_toks, _ = self._to_tensor([s.toks_inp_masked for s in subsets], enc_pad, False)
        prompts_toks, prompts_att_mask = self._to_tensor([s.toks_prompt for s in subsets], dec_pad, True)
        cites_toks, cites_att_mask = self._to_tensor([s.toks_cite_dec for s in subsets], dec_pad, True)
        cites_masked_toks, _ = self._to_tensor([s.toks_cite_masked_dec for s in subsets], dec_pad, False)
        inp_toks_dec, inp_dec_att_mask = self._to_tensor([s.toks_inp_dec for s in subsets], dec_pad, True)
        inp_masked_toks_dec, _ = self._to_tensor([s.toks_inp_masked_dec for s in subsets], dec_pad, False)
        edge_inds = torch.stack([
            torch.arange(batch_size, device=self.device),
            torch.full((batch_size,), batch_size, device=self.device),
        ])

        return MaskedCiteBatch(
            tokens_subsets=subsets,
            inp_toks=inp_toks,
            inp_masked_toks=inp_masked_toks,
            prompts_toks=prompts_toks,
            cites_masked_toks=cites_masked_toks,
            cites_toks=cites_toks,
            inp_att_mask=inp_att_mask,
            prompts_att_mask=prompts_att_mask,
            cites_att_mask=cites_att_mask,
            inp_toks_dec=inp_toks_dec,
            inp_masked_toks_dec=inp_masked_toks_dec,
            inp_dec_att_mask=inp_dec_att_mask,
            edge_inds=edge_inds,
        )

    def shuffle(self, seed: Optional[int] = None) -> 'BaseRecallDataset':
        if seed is not None:
            np.random.default_rng(seed).shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def create_recall_dataloader(
        dataset: BaseRecallDataset, batch_size: int, name: str = 'Recall',
) -> Generator[MaskedCiteBatch, None, None]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create {name}Dataset dataloader. batch_size={batch_size}.')
    start_ind = 0
    while True:
        end_ind = min(start_ind + batch_size, len(dataset))
        inds = dataset.inds[start_ind:end_ind].tolist()
        if len(inds) < batch_size:
            inds += dataset.inds[:(batch_size - len(inds))].tolist()
        batch = dataset.get_batch(inds)
        if end_ind == len(dataset):
            print(f'R{rank}. Shuffle dataset')
            dataset.shuffle()
        yield batch
        start_ind = end_ind % len(dataset)

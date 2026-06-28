"""XML/XPath recall dataset (structured-extraction family §2.4).

Automated, but grounded in real Wikipedia text: each sample builds a randomized
XML tree and asks for one value via an XPath-like query. The answer is a verbatim
attribute value, text-node value, or (composite) element subtree selected from
the serialized XML record.

Testability split
-----------------
``sample_spec`` fixes the tree *shape* (attribute presence/count, child
presence/count, text placement, per-value word counts) and the query indices —
all randomness lives here. :meth:`XmlXpathRecallTokenizer.realize` deterministic-
ally fills tag/attr names and values *sequentially* from the word feed.
"""
from dataclasses import dataclass, field
from html import escape
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.extraction_common import (
    BaseRecallDataset, WordFeed, assemble_subset, build_to_budget,
    create_recall_dataloader, dec_end_token_id, make_pool, word_feed_from_text,
)


@dataclass(kw_only=True)
class XmlXpathRecallCfg:
    min_nodes: int = 4
    max_nodes: int = 12
    max_depth: int = 4
    max_children: int = 4
    value_min_words: int = 1
    value_max_words: int = 3
    attr_prob: float = 0.35
    nested_prob: float = 0.45
    min_tag_chars: int = 2
    # Tag/attribute-name length in *words*. Multi-word names stay distinctive so
    # a frequent stopword cannot collide across elements / XPaths.
    name_min_words: int = 3
    name_max_words: int = 4
    # Probability of a composite (element subtree) target.
    composite_prob: float = 0.0
    # When True, grow records to ~fill the chunk (largest sampled record that
    # fits the token budget; early-stop at fill_frac * budget).
    fill_to_budget: bool = False
    fill_frac: float = 0.85


@dataclass(kw_only=True)
class XmlNodeSpec:
    """Content-agnostic shape of one XML element."""
    attr_word_counts: List[int] = field(default_factory=list)
    children: List['XmlNodeSpec'] = field(default_factory=list)
    text_word_count: int = 1
    text_before: bool = False  # only meaningful when children present


@dataclass(kw_only=True)
class XmlSpec:
    root: XmlNodeSpec
    name_words: int  # words per tag / attribute name
    query_idx: int  # modular index into selected (or composite) targets
    q_kind: int     # prompt phrasing: 0 / 1 / 2
    composite: bool = False


class XmlXpathRecallTokenizer:
    def __init__(
            self, tkz_enc: PreTrainedTokenizer, max_len: int, cfg: Optional[XmlXpathRecallCfg] = None,
            n_special_toks: int = 1000, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        self.tkz_enc = tkz_enc
        self.tkz_dec = tkz_dec if tkz_dec is not None else tkz_enc
        self.same_vocab = self.tkz_enc is self.tkz_dec or len(self.tkz_enc) == len(self.tkz_dec)
        self.max_len = max_len
        self.cfg = cfg if cfg is not None else XmlXpathRecallCfg()
        self.rng = np.random.default_rng(seed)
        self.pool = make_pool(self.rng, n_special_toks, len(self.tkz_enc))
        self.dec_end_token_id = dec_end_token_id(self.tkz_dec)

    # --- helpers -------------------------------------------------------------
    def _normalize_name(self, txt: str) -> str:
        out = []
        for ch in txt.lower():
            if ch.isalnum() or ch in ('_', '-'):
                out.append(ch)
            elif ch == ' ':
                out.append('_')
        name = ''.join(out).strip('_-')
        if not name:
            return name
        if name[0].isdigit():
            name = f'n_{name}'
        return name

    def _is_name(self, name: str) -> bool:
        return len(name) >= self.cfg.min_tag_chars and any(c.isalpha() for c in name)

    def _next_name(self, feed: WordFeed, used: set, prefix: str = 'n', n_words: int = 1) -> str:
        parts: List[str] = []
        for _ in range(max(1, n_words)):
            part = None
            for _ in range(64):
                _, txt = feed.next_word()
                n = self._normalize_name(txt)
                if self._is_name(n):
                    part = n
                    break
            if part is None:
                for _ in range(256):
                    tid = feed.next_pool_token()
                    n = self._normalize_name(self.tkz_enc.decode([tid], skip_special_tokens=True))
                    if self._is_name(n):
                        part = n
                        break
                if part is None:
                    part = f'{prefix}_{len(used)}'
            parts.append(part)
        name = '_'.join(parts)
        if name in used:
            suffix = 1
            while f'{name}_{suffix}' in used:
                suffix += 1
            name = f'{name}_{suffix}'
        used.add(name)
        return name

    # --- random: spec sampling ----------------------------------------------
    def sample_spec(self) -> XmlSpec:
        cfg = self.cfg
        rng = self.rng
        self._nn = int(rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
        root = self._sample_node(depth=1)
        return XmlSpec(
            root=root,
            name_words=int(rng.integers(cfg.name_min_words, cfg.name_max_words + 1)),
            query_idx=int(rng.integers(1 << 30)),
            q_kind=int(rng.integers(3)),
            composite=float(rng.random()) < cfg.composite_prob,
        )

    def _sample_node(self, depth: int) -> XmlNodeSpec:
        cfg = self.cfg
        rng = self.rng
        attr_word_counts: List[int] = []
        if float(rng.random()) < cfg.attr_prob:
            n_attrs = int(rng.integers(1, 3))
            attr_word_counts = [
                int(rng.integers(cfg.value_min_words, cfg.value_max_words + 1))
                for _ in range(n_attrs)
            ]
        children: List[XmlNodeSpec] = []
        can_nest = depth < cfg.max_depth and self._nn > 0
        if can_nest and float(rng.random()) < cfg.nested_prob:
            n_child = int(rng.integers(1, cfg.max_children + 1))
            for _ in range(n_child):
                if self._nn <= 0:
                    break
                self._nn -= 1
                children.append(self._sample_node(depth + 1))
        text_word_count = int(rng.integers(cfg.value_min_words, cfg.value_max_words + 1))
        text_before = bool(rng.integers(2)) if children else False
        return XmlNodeSpec(
            attr_word_counts=attr_word_counts, children=children,
            text_word_count=text_word_count, text_before=text_before,
        )

    # --- deterministic: realization -----------------------------------------
    def _emit(
            self, node: XmlNodeSpec, feed: WordFeed, path_parts: List[str], used_tags: set,
            selected: List[Tuple[str, str]], composites: List[Tuple[str, str]], name_words: int = 1,
    ) -> str:
        tag = path_parts[-1]
        attrs: List[Tuple[str, str]] = []
        used_attrs: set = set()
        for c in node.attr_word_counts:
            a = self._next_name(feed, used_attrs, prefix='a', n_words=name_words)
            _, v = feed.next_span(c)
            attrs.append((a, v))
            selected.append((f"/{'/'.join(path_parts)}/@{a}", v))

        children_xml: List[str] = []
        for child in node.children:
            child_tag = self._next_name(feed, used_tags, prefix='t', n_words=name_words)
            children_xml.append(self._emit(child, feed, path_parts + [child_tag], used_tags, selected, composites, name_words))

        _, text_value = feed.next_span(node.text_word_count)
        selected.append((f"/{'/'.join(path_parts)}/text()", text_value))

        attr_str = ''.join([f' {k}="{escape(v, quote=True)}"' for k, v in attrs])
        if children_xml:
            body = ''.join(children_xml)
            body = f'{escape(text_value)}{body}' if node.text_before else f'{body}{escape(text_value)}'
            xml = f'<{tag}{attr_str}>{body}</{tag}>'
        else:
            xml = f'<{tag}{attr_str}>{escape(text_value)}</{tag}>'
        composites.append((f"/{'/'.join(path_parts)}", xml))
        return xml

    def realize(self, spec: XmlSpec, feed: WordFeed, ids_src: Optional[List[int]] = None) -> Tuple[TokensSubsetV2, int]:
        budget = self.max_len - 2
        used_tags: set = set()
        selected: List[Tuple[str, str]] = []
        composites: List[Tuple[str, str]] = []
        root_tag = self._next_name(feed, used_tags, prefix='root', n_words=spec.name_words)
        xml_text = self._emit(spec.root, feed, [root_tag], used_tags, selected, composites, spec.name_words)

        rec_ids = self.tkz_enc(xml_text, add_special_tokens=False).input_ids
        rec_len = len(rec_ids)
        if rec_len > budget:
            rec_ids = rec_ids[:budget]
            xml_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        candidates = composites if (spec.composite and composites) else selected
        if not candidates:
            candidates = selected or [('/', xml_text)]
        n = len(candidates)
        xpath, value = candidates[spec.query_idx % n]
        for j in range(n):
            if value and value in xml_text:
                break
            xpath, value = candidates[(spec.query_idx + j + 1) % n]

        if spec.q_kind == 0:
            prompt = f'XPath: {xpath}. Return value only.'
        elif spec.q_kind == 1:
            prompt = f'XML query: extract {xpath}. Return value only.'
        else:
            prompt = f'What is the value selected by XPath {xpath}? Return value only.'
        prompt_toks = self.tkz_dec(prompt, add_special_tokens=False).input_ids

        subset = assemble_subset(
            ids_src=ids_src if ids_src is not None else list(rec_ids),
            rec_ids=rec_ids,
            prompt=prompt,
            prompt_toks=prompt_toks,
            tkz_enc=self.tkz_enc,
            tkz_dec=self.tkz_dec,
            same_vocab=self.same_vocab,
            end_token_id=self.dec_end_token_id,
            target_text=value,
        )
        return subset, rec_len

    # --- wrapper -------------------------------------------------------------
    def build(self, text: str) -> TokensSubsetV2:
        feed, ids_src = word_feed_from_text(text, self.tkz_enc, self.pool, rng=self.rng)
        return build_to_budget(
            sample_spec=self.sample_spec,
            realize=self.realize,
            feed=feed,
            ids_src=ids_src,
            budget=self.max_len - 2,
            n_attempts=20,
            fill_to_budget=self.cfg.fill_to_budget,
            fill_frac=self.cfg.fill_frac,
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class XmlXpathRecallDataset(BaseRecallDataset):
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[XmlXpathRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
    ):
        builder = XmlXpathRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=tkz_dec if tkz_dec is not None else tkz_enc, seed=seed,
        )
        super().__init__(dataset, tkz_enc, max_seq_len, builder, device=device, tkz_dec=tkz_dec)


def create_xml_xpath_recall_dataloader(dataset: XmlXpathRecallDataset, batch_size: int):
    return create_recall_dataloader(dataset, batch_size, name='XmlXpathRecall')

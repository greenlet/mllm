"""XML/XPath recall dataset (structured-extraction family §2.4).

Automated, but grounded in real Wikipedia text: each sample builds a randomized
XML tree and asks for one value via XPath-like query. The answer is a verbatim
attribute value or text node value selected from the serialized XML record.

The dataset emits ``MaskedCiteBatch`` so it routes through
``MixedDecoder.run_on_text_citation`` unchanged.
"""
from dataclasses import dataclass
from html import escape
from typing import Any, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import PreTrainedTokenizer

from mllm.data.utils import TokensSubsetV2
from mllm.train.encdec_graph_bert import MaskedCiteBatch


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
        self.pool = self.rng.permutation(np.arange(n_special_toks, len(self.tkz_enc)))
        self.pool_cur = 0
        td = self.tkz_dec
        self.dec_end_token_id = td.sep_token_id if td.sep_token_id is not None else td.eos_token_id

    def _next_pool_token(self) -> int:
        tid = int(self.pool[self.pool_cur])
        self.pool_cur += 1
        if self.pool_cur >= len(self.pool):
            self.rng.shuffle(self.pool)
            self.pool_cur = 0
        return tid

    def _enc_to_dec(self, enc_ids: List[int]) -> List[int]:
        if self.same_vocab:
            return list(enc_ids)
        if not enc_ids:
            return []
        text = self.tkz_enc.decode(enc_ids, skip_special_tokens=True)
        return self.tkz_dec(text, add_special_tokens=False).input_ids

    def _split_words(self, ids: List[int]) -> List[List[int]]:
        toks = self.tkz_enc.convert_ids_to_tokens(ids)
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

    def _pick_name(self, pool: List[str], used: set[str], prefix: str = 'n') -> str:
        for _ in range(max(8, len(pool))):
            if pool:
                n = pool[int(self.rng.integers(len(pool)))]
                if n not in used and self._is_name(n):
                    used.add(n)
                    return n
            else:
                break
        while True:
            n = f'{prefix}_{int(self.rng.integers(10_000_000))}'
            if n not in used:
                used.add(n)
                return n

    def _pick_text_value(self, words: List[List[int]]) -> str:
        for _ in range(10):
            if not words:
                tok = [self._next_pool_token()]
                txt = self.tkz_enc.decode(tok, skip_special_tokens=True).strip()
            else:
                n = int(self.rng.integers(self.cfg.value_min_words, self.cfg.value_max_words + 1))
                i0 = int(self.rng.integers(len(words)))
                ids: List[int] = []
                for w in words[i0:i0 + n]:
                    ids.extend(w)
                txt = self.tkz_enc.decode(ids, skip_special_tokens=True).strip()
            if txt and any(c.isalnum() for c in txt):
                return txt
        return f'v{int(self.rng.integers(100000))}'

    def _build_xml(self, words: List[List[int]], name_pool: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
        cfg = self.cfg
        n_nodes = int(self.rng.integers(cfg.min_nodes, cfg.max_nodes + 1))
        selected: List[Tuple[str, str]] = []

        used_tags: set[str] = set()
        root_tag = self._pick_name(name_pool, used_tags, prefix='root')

        def emit_node(path_parts: List[str], depth: int) -> str:
            nonlocal n_nodes
            tag = path_parts[-1]
            attrs: List[Tuple[str, str]] = []
            used_attrs: set[str] = set()
            if float(self.rng.random()) < cfg.attr_prob:
                n_attrs = int(self.rng.integers(1, 3))
                for _ in range(n_attrs):
                    a = self._pick_name(name_pool, used_attrs, prefix='a')
                    v = self._pick_text_value(words)
                    attrs.append((a, v))
                    selected.append((f"/{'/'.join(path_parts)}/@{a}", v))

            children: List[str] = []
            can_nest = depth < cfg.max_depth and n_nodes > 0
            if can_nest and float(self.rng.random()) < cfg.nested_prob:
                n_child = int(self.rng.integers(1, cfg.max_children + 1))
                for _ in range(n_child):
                    if n_nodes <= 0:
                        break
                    child_tag = self._pick_name(name_pool, used_tags, prefix='t')
                    n_nodes -= 1
                    children.append(emit_node(path_parts + [child_tag], depth + 1))

            text_value = self._pick_text_value(words)
            selected.append((f"/{'/'.join(path_parts)}/text()", text_value))

            attr_str = ''.join([f' {k}="{escape(v, quote=True)}"' for k, v in attrs])
            if children:
                body = ''.join(children)
                # Mix text-before/text-after to avoid fixed position shortcuts.
                if bool(self.rng.integers(2)):
                    body = f'{escape(text_value)}{body}'
                else:
                    body = f'{body}{escape(text_value)}'
                return f'<{tag}{attr_str}>{body}</{tag}>'
            return f'<{tag}{attr_str}>{escape(text_value)}</{tag}>'

        xml = emit_node([root_tag], depth=1)
        return xml, selected

    def build(self, text: str) -> TokensSubsetV2:
        ids_src = self.tkz_enc(text, add_special_tokens=False).input_ids
        words = self._split_words(ids_src)

        name_pool: List[str] = []
        seen = set()
        for w in words:
            n = self._normalize_name(self.tkz_enc.decode(w, skip_special_tokens=True).strip())
            if self._is_name(n) and n not in seen:
                seen.add(n)
                name_pool.append(n)

        budget = self.max_len - 2
        best = None
        for _ in range(20):
            xml_text, selected = self._build_xml(words, name_pool)
            rec_ids = self.tkz_enc(xml_text, add_special_tokens=False).input_ids
            if len(rec_ids) <= budget and selected:
                best = (xml_text, selected, rec_ids)
                break
            if best is None or len(rec_ids) < len(best[2]):
                best = (xml_text, selected, rec_ids)

        assert best is not None
        xml_text, selected, rec_ids = best
        if len(rec_ids) > budget:
            rec_ids = rec_ids[:budget]
            xml_text = self.tkz_enc.decode(rec_ids, skip_special_tokens=True)

        xpath, value = selected[int(self.rng.integers(len(selected)))]
        for _ in range(12):
            if value and value in xml_text:
                break
            xpath, value = selected[int(self.rng.integers(len(selected)))]

        q_kind = int(self.rng.integers(3))
        if q_kind == 0:
            prompt = f'XPath: {xpath}. Return value only.'
        elif q_kind == 1:
            prompt = f'XML query: extract {xpath}. Return value only.'
        else:
            prompt = f'What is the value selected by XPath {xpath}? Return value only.'

        prompt_toks = self.tkz_dec(prompt, add_special_tokens=False).input_ids
        target_dec = self.tkz_dec(value, add_special_tokens=False).input_ids
        end_suffix = [self.dec_end_token_id] if self.dec_end_token_id is not None else []
        cites_dec = target_dec + end_suffix
        inp_dec = self._enc_to_dec(rec_ids) + end_suffix

        cls_id = self.tkz_enc.cls_token_id
        sep_id = self.tkz_enc.sep_token_id
        toks_inp = [cls_id] + rec_ids + [sep_id]
        tgt_enc = self.tkz_enc(value, add_special_tokens=False).input_ids

        return TokensSubsetV2(
            toks_src=ids_src,
            inp_beg_ind=0,
            inp_end_ind=len(ids_src),
            toks_inp=toks_inp,
            toks_inp_masked=list(toks_inp),
            cite_beg_ind=-1,
            cite_end_ind=-1,
            toks_cite=tgt_enc,
            toks_cite_masked=list(tgt_enc),
            toks_cite_beg=[],
            toks_cite_end=[],
            prompt=prompt,
            toks_prompt=prompt_toks,
            toks_inp_dec=inp_dec,
            toks_inp_masked_dec=list(inp_dec),
            toks_cite_dec=cites_dec,
            toks_cite_masked_dec=list(cites_dec),
        )

    def __call__(self, texts: List[str]) -> List[TokensSubsetV2]:
        return [self.build(t) for t in texts]


class XmlXpathRecallDataset:
    def __init__(
            self, dataset: Dataset, tkz_enc: PreTrainedTokenizer, max_seq_len: int,
            n_special_toks: int = 1000, cfg: Optional[XmlXpathRecallCfg] = None,
            device: Optional[torch.device] = None, tkz_dec: Optional[PreTrainedTokenizer] = None,
            seed: Optional[int] = None,
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
        self.builder = XmlXpathRecallTokenizer(
            tkz_enc, max_len=max_seq_len, cfg=cfg, n_special_toks=n_special_toks,
            tkz_dec=self.tkz_dec, seed=seed,
        )

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

    def shuffle(self, seed: Optional[int] = None) -> 'XmlXpathRecallDataset':
        if seed is not None:
            np.random.default_rng(seed).shuffle(self.inds)
        else:
            np.random.shuffle(self.inds)
        return self


def create_xml_xpath_recall_dataloader(
        dataset: XmlXpathRecallDataset, batch_size: int,
) -> Generator[MaskedCiteBatch, None, None]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f'R{rank}. Create XmlXpathRecallDataset dataloader. batch_size={batch_size}.')
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

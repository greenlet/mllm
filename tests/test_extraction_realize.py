"""Deterministic ``realize`` tests for the structured-extraction builders.

These tests pin down the *testability* contract introduced by the spec/realize
refactor (docs/mixed_decoder + ``mllm/train/extraction_common.py``):

    All randomness lives in ``sample_spec``. ``realize(spec, feed, ...)`` is a
    PURE, DETERMINISTIC function of ``(spec, WordFeed)`` — words are consumed
    *sequentially* from the feed, never sampled by random position.

Consequently we can:

* hand-build a :class:`WordFeed` from a known list of words and a hand-built
  spec, and assert the *exact* serialized record / decoder target, and
* prove ``realize`` is pure by running it twice on two identical feeds and
  getting byte-identical token sequences.

The cite builder (``RandomInputTokenizerV2`` in ``mllm/data/utils.py``) follows
the same shape with :class:`CiteSpecV2` + ``sample_spec`` / ``realize``.
"""
from typing import List, Optional

import numpy as np
import pytest
from transformers import AutoTokenizer

from mllm.data.utils import RandomInputTokenizerV2, CiteSpecV2
from mllm.train.extraction_common import WordFeed, word_feed_from_text, make_pool
from mllm.train.mask_utils import MaskCfg

from mllm.train.key_val_recall import KeyValRecallTokenizer, KeyValRecallCfg, KeyValSpec
from mllm.train.json_field_recall import JsonFieldRecallTokenizer, JsonFieldRecallCfg
from mllm.train.jsonata_recall import JsonataRecallTokenizer, JsonataRecallCfg
from mllm.train.xml_xpath_recall import XmlXpathRecallTokenizer, XmlXpathRecallCfg
from mllm.train.sql_select_recall import SqlSelectRecallTokenizer, SqlSelectRecallCfg


BERT_NAME = 'bert-base-uncased'
MAX_LEN = 192
POOL = [1100, 1200, 1300, 1400, 1500, 1600]

# A word-rich passage for the generic purity tests.
RICH_TEXT = (
    'The Amazon rainforest is a moist broadleaf tropical forest covering most of the '
    'Amazon basin of South America across nine nations and many protected indigenous '
    'territories rich with biodiversity rivers wildlife plants insects and birds today.'
)


@pytest.fixture(scope='module')
def tkz():
    return AutoTokenizer.from_pretrained(BERT_NAME)


def _make_feed(tkz, words_str: List[str], pool=None) -> WordFeed:
    """A deterministic feed seeded from an explicit list of word strings."""
    words = [tkz(w, add_special_tokens=False).input_ids for w in words_str]
    texts = [tkz.decode(ids, skip_special_tokens=True).strip() for ids in words]
    return WordFeed(words, texts, pool if pool is not None else POOL, tkz)


def _dec(tkz, ids: List[int]) -> str:
    return tkz.decode(ids, skip_special_tokens=True).strip()


def _unwrap(out):
    """Builders that retry-until-fits return ``(subset, rec_len)``; others return
    just the subset. Normalize to the subset."""
    if isinstance(out, tuple):
        return out[0]
    return out


# ===========================================================================
# cite — RandomInputTokenizerV2.realize is a pure function of (ids, CiteSpecV2)
# ===========================================================================

def test_cite_realize_exact_span(tkz):
    tk = RandomInputTokenizerV2(tkz, max_len=64, n_random_toks=3, mask_cfg=None, tkz_dec=tkz)
    ids = tkz('alpha bravo charlie delta echo', add_special_tokens=False).input_ids
    # Cite the single word at offset 1 ('bravo').
    spec = CiteSpecV2(
        inp_beg_ind=0, inp_end_ind=len(ids), sub_off=1, n_cite_toks=1,
        toks_cite_beg=[2000, 2001, 2002], toks_cite_end=[2003, 2004, 2005],
        raw_mask=None, prompt_all=True,
    )
    sub = tk.realize(ids, spec)
    assert _dec(tkz, sub.toks_cite) == 'bravo'
    # Layout: [CLS] + prefix(sub_off=1) + beg + cite + end + suffix + [SEP].
    assert sub.toks_inp[2:5] == spec.toks_cite_beg
    assert sub.toks_inp[6:9] == spec.toks_cite_end


def test_cite_realize_is_deterministic(tkz):
    tk = RandomInputTokenizerV2(tkz, max_len=64, n_random_toks=3, mask_cfg=None, tkz_dec=tkz)
    ids = tkz('one two three four five six', add_special_tokens=False).input_ids
    spec = CiteSpecV2(
        inp_beg_ind=0, inp_end_ind=len(ids), sub_off=2, n_cite_toks=2,
        toks_cite_beg=[2000, 2001, 2002], toks_cite_end=[2003, 2004, 2005],
        raw_mask=None, prompt_all=False,
    )
    a = tk.realize(ids, spec)
    b = tk.realize(ids, spec)
    assert a.toks_inp == b.toks_inp
    assert a.toks_inp_masked == b.toks_inp_masked
    assert a.toks_cite == b.toks_cite
    assert a.toks_cite_dec == b.toks_cite_dec
    assert a.toks_prompt == b.toks_prompt


def test_cite_mask_hides_span_in_encoder_only(tkz):
    tk = RandomInputTokenizerV2(tkz, max_len=64, n_random_toks=3, mask_cfg=MaskCfg(), tkz_dec=tkz)
    ids = tkz('visible secret visible word here', add_special_tokens=False).input_ids
    spec = CiteSpecV2(
        inp_beg_ind=0, inp_end_ind=len(ids), sub_off=1, n_cite_toks=1,
        toks_cite_beg=[2000, 2001, 2002], toks_cite_end=[2003, 2004, 2005],
        raw_mask=np.array([True]), prompt_all=True,
    )
    sub = tk.realize(ids, spec)
    # Encoder side: the cited word is fully masked.
    assert all(t == tkz.mask_token_id for t in sub.toks_cite_masked)
    # Decoder target: the unmasked span is preserved.
    assert _dec(tkz, sub.toks_cite) == 'secret'


# ===========================================================================
# keyval — exact value retrieval + composite whole-record dump
# ===========================================================================

def _keyval_builder(tkz, **cfg_kw):
    cfg = KeyValRecallCfg(**cfg_kw)
    return KeyValRecallTokenizer(tkz, max_len=MAX_LEN, cfg=cfg, tkz_dec=tkz, seed=0)


def test_keyval_realize_exact_value(tkz):
    tk = _keyval_builder(tkz)
    feed = _make_feed(tkz, ['capital', 'paris', 'river', 'amazon'])
    spec = KeyValSpec(
        n_pairs=2, kv_sep_idx=0, pair_sep_idx=0,
        value_word_counts=[1, 1], query_pair_idx=0, composite=None,
    )
    sub = tk.realize(spec, feed, ids_src=None)
    # Query pair 0 -> value 'paris'.
    assert _dec(tkz, sub.toks_cite) == 'paris'
    # Both keys + values are present verbatim in the encoded record.
    rec_txt = _dec(tkz, sub.toks_src)
    for w in ('capital', 'paris', 'river', 'amazon'):
        assert w in rec_txt


def test_keyval_realize_second_pair(tkz):
    tk = _keyval_builder(tkz)
    feed = _make_feed(tkz, ['capital', 'paris', 'river', 'amazon'])
    spec = KeyValSpec(
        n_pairs=2, kv_sep_idx=0, pair_sep_idx=0,
        value_word_counts=[1, 1], query_pair_idx=1, composite=None,
    )
    sub = tk.realize(spec, feed, ids_src=None)
    assert _dec(tkz, sub.toks_cite) == 'amazon'


def test_keyval_composite_record_is_grounded(tkz):
    tk = _keyval_builder(tkz)
    feed = _make_feed(tkz, ['capital', 'paris', 'river', 'amazon'])
    spec = KeyValSpec(
        n_pairs=2, kv_sep_idx=0, pair_sep_idx=0,
        value_word_counts=[1, 1], query_pair_idx=0, composite='record',
    )
    sub = tk.realize(spec, feed, ids_src=None)
    # Composite target == the whole record (verbatim, grounded).
    assert _dec(tkz, sub.toks_cite) == _dec(tkz, sub.toks_src)


def test_keyval_realize_is_pure(tkz):
    tk = _keyval_builder(tkz)
    spec = KeyValSpec(
        n_pairs=3, kv_sep_idx=1, pair_sep_idx=2,
        value_word_counts=[2, 1, 3], query_pair_idx=2, composite=None,
    )
    words = ['capital', 'paris', 'france', 'river', 'amazon', 'mountain', 'andes', 'tall', 'peak']
    s1 = tk.realize(spec, _make_feed(tkz, words), ids_src=None)
    s2 = tk.realize(spec, _make_feed(tkz, words), ids_src=None)
    assert s1.toks_src == s2.toks_src
    assert s1.toks_cite == s2.toks_cite
    assert s1.toks_prompt == s2.toks_prompt


# ===========================================================================
# Generic purity: realize() is deterministic for every structured builder.
# ===========================================================================

STRUCTURED_BUILDERS = [
    pytest.param(lambda t: KeyValRecallTokenizer(
        t, max_len=MAX_LEN, cfg=KeyValRecallCfg(min_pairs=3, max_pairs=5), tkz_dec=t, seed=7), id='keyval'),
    pytest.param(lambda t: JsonFieldRecallTokenizer(
        t, max_len=MAX_LEN, cfg=JsonFieldRecallCfg(min_fields=3, max_fields=5), tkz_dec=t, seed=7), id='jsonfield'),
    pytest.param(lambda t: JsonataRecallTokenizer(
        t, max_len=MAX_LEN, cfg=JsonataRecallCfg(min_fields=3, max_fields=5), tkz_dec=t, seed=7), id='jsonata'),
    pytest.param(lambda t: XmlXpathRecallTokenizer(
        t, max_len=MAX_LEN, cfg=XmlXpathRecallCfg(min_nodes=3, max_nodes=5), tkz_dec=t, seed=7), id='xmlxpath'),
    pytest.param(lambda t: SqlSelectRecallTokenizer(
        t, max_len=MAX_LEN, cfg=SqlSelectRecallCfg(min_rows=3, max_rows=5), tkz_dec=t, seed=7), id='sql'),
]


@pytest.mark.parametrize('make_builder', STRUCTURED_BUILDERS)
def test_structured_realize_is_pure(make_builder, tkz):
    """Two identical feeds + the same spec must yield byte-identical output."""
    tk = make_builder(tkz)
    spec = tk.sample_spec()
    feed1, ids1 = word_feed_from_text(RICH_TEXT, tkz, tk.pool)
    feed2, ids2 = word_feed_from_text(RICH_TEXT, tkz, tk.pool)
    s1 = _unwrap(tk.realize(spec, feed1, ids_src=ids1))
    s2 = _unwrap(tk.realize(spec, feed2, ids_src=ids2))
    assert s1.toks_src == s2.toks_src
    assert s1.toks_inp == s2.toks_inp
    assert s1.toks_cite == s2.toks_cite
    assert s1.toks_cite_dec == s2.toks_cite_dec
    assert s1.toks_prompt == s2.toks_prompt
    assert s1.prompt == s2.prompt


@pytest.mark.parametrize('make_builder', STRUCTURED_BUILDERS)
def test_structured_build_runs(make_builder, tkz):
    """The thin ``build(text)`` wrapper (sample_spec -> realize) produces a
    non-empty record + target end-to-end."""
    tk = make_builder(tkz)
    sub = tk.build(RICH_TEXT)
    assert len(sub.toks_src) > 0
    assert len(sub.toks_cite_dec) > 0
    assert len(sub.toks_prompt) > 0

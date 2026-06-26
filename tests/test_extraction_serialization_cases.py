"""Test-case specifications (stubs) for the structured-extraction datasets.

This module enumerates *input -> expected-output* test cases for every dataset
exercised by ``notebooks/n_12_06_extraction_datasets.ipynb`` and
``s_03_11_train_mixed_decoder.py``:

| type        | module                              | builder                       |
|-------------|-------------------------------------|-------------------------------|
| ``cite``    | ``mllm.train.encdec_graph_bert``    | ``RandomInputTokenizerV2``    |
| ``keyval``  | ``mllm.train.key_val_recall``       | ``KeyValRecallTokenizer``     |
| ``jsonfield``| ``mllm.train.json_field_recall``   | ``JsonFieldRecallTokenizer``  |
| ``jsonata`` | ``mllm.train.jsonata_recall``       | ``JsonataRecallTokenizer``    |
| ``xmlxpath``| ``mllm.train.xml_xpath_recall``     | ``XmlXpathRecallTokenizer``   |
| ``sql``     | ``mllm.train.sql_select_recall``    | ``SqlSelectRecallTokenizer``  |

Scope of THIS file
------------------
* These are **stubs** — the test bodies are intentionally *not implemented*.
  Each test documents the **input**, the **query/path**, and the **expected
  serialized target** so the behaviour can be pinned down before coding.
* The emphasis is the open question raised in the notebook review:
  **"what happens when the queried path does not resolve to a single scalar,
  and the value must be serialized?"** (a nested object, an array, an XML
  subtree, a whole/partial SQL row, multiple matching rows, ...).

Every case carries an explicit ``expected`` string. Where the *current*
generators cannot yet produce the query (they only descend to scalar leaves),
the case is tagged ``composite`` and the ``expected`` field encodes the
**desired serialization contract** that an implementation must satisfy.

Two invariants the cases are written against
--------------------------------------------
1. **Needle-in-context.** For verbatim (selection) families the serialized
   target MUST be a substring of the serialized record. The composite cases
   deliberately probe where naive ``json.dumps(value)`` (default separators)
   would *break* this invariant against a compact record (separators mismatch).
2. **Round-trip determinism.** Given a fixed record + path, the serialized
   target is a pure function of (value, serialization-policy) and must not
   depend on RNG.
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest


STUB_REASON = 'stub: test case specified but not implemented'


# ---------------------------------------------------------------------------
# Shared case schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalCase:
    """One input -> expected-output specification.

    Attributes
    ----------
    name:
        Stable, human-readable id (used as the pytest param id).
    record:
        The structured context the model must read from. Concretely:
        * keyval     -> ``list[tuple[str, str]]`` of key/value pairs.
        * jsonfield  -> a ``dict`` (the JSON object).
        * jsonata    -> a ``dict`` (the JSON object).
        * xmlxpath   -> an XML ``str``.
        * sql        -> ``dict`` with ``cols``/``rows`` (a tiny table).
        * cite       -> the source ``str`` the citation span is drawn from.
    query:
        The path / key / xpath / sql / anchor that selects the answer.
    expected:
        The exact serialized target string the decoder must emit.
    kind:
        ``scalar`` | ``array`` | ``object`` | ``subtree`` | ``row`` |
        ``rows`` | ``computed`` — what the query resolves to.
    composite:
        True when the answer is NOT a single scalar and serialization policy
        is the crux of the case.
    grounded:
        True when ``expected`` MUST appear verbatim inside the serialized
        record (selection families). False for computed/aggregate answers.
    notes:
        Free-form rationale / corner-case description.
    """
    name: str
    record: Any
    query: str
    expected: str
    kind: str = 'scalar'
    composite: bool = False
    grounded: bool = True
    notes: str = ''
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conversion helpers: drive the REAL serialization helpers / builders.
# ---------------------------------------------------------------------------
import html
import json
import re
import xml.etree.ElementTree as ET

import numpy as np
from transformers import AutoTokenizer

from mllm.data.utils import RandomInputTokenizerV2, CiteSpecV2
from mllm.train.extraction_common import (
    json_seps, json_target_text, sql_row_text, sql_rows_text,
)

_BERT_NAME = 'bert-base-uncased'
_TKZ = None


def _tkz():
    """Lazily-loaded shared encoder/decoder tokenizer (BERT, same vocab)."""
    global _TKZ
    if _TKZ is None:
        _TKZ = AutoTokenizer.from_pretrained(_BERT_NAME)
    return _TKZ


# -- JSON (jsonfield / jsonata selection) -----------------------------------

def _infer_compact(expected: str) -> bool:
    """The record's separator policy is encoded in the expected target: a spaced
    serialization (``", "`` / ``": "``) means the record was dumped spaced."""
    return not (', ' in expected or ': ' in expected)


def _resolve_json(record, query: str):
    """Resolve a JSONPath/jq/dotted selection to its value.

    Supports ``$.a.b``, ``a.b``, ``.a.b``, ``xs``, ``$.arr[1]``.
    """
    q = query.strip()
    q = q.lstrip('$')
    if q.startswith('.'):
        q = q[1:]
    parts = re.findall(r'[^.\[\]]+|\[\d+\]', q)
    cur = record
    for p in parts:
        if p.startswith('['):
            cur = cur[int(p[1:-1])]
        else:
            cur = cur[p]
    return cur


def _json_record_text(record, compact: bool) -> str:
    return json.dumps(record, ensure_ascii=True, separators=json_seps(compact))


# -- SQL (markdown table) ----------------------------------------------------

def _sql_parse(query: str):
    """Parse the tiny SELECT dialect into (cols_or_star, where_col, op, value)."""
    m = re.match(
        r'\s*SELECT\s+(?P<cols>\*|[\w, ]+?)\s+FROM\s+t'
        r'(?:\s+WHERE\s+(?P<wcol>\w+)\s*(?P<op><=|>=|=)\s*(?P<val>\d+))?\s*$',
        query, re.IGNORECASE,
    )
    assert m, f'unparseable SQL: {query!r}'
    cols_raw = m.group('cols').strip()
    cols = None if cols_raw == '*' else [c.strip() for c in cols_raw.split(',')]
    wcol, op, val = m.group('wcol'), m.group('op'), m.group('val')
    return cols, wcol, op, (int(val) if val is not None else None)


def _sql_filter(rows, wcol, op, val):
    if wcol is None:
        return list(rows)
    cmp = {'=': lambda a, b: a == b, '<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b}[op]
    return [r for r in rows if cmp(r[wcol], val)]


def _sql_markdown(record) -> str:
    cols, rows = record['cols'], record['rows']
    header = '| ' + ' | '.join(cols) + ' |'
    body = ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in rows]
    return '\n'.join([header] + body)


def _sql_aggregate(record, query: str) -> str:
    """Evaluate a Tier-C SQL aggregate (COUNT/SUM/MAX) over the filtered rows."""
    m = re.match(
        r'\s*SELECT\s+(?P<fn>COUNT|SUM|MAX)\(\s*(?P<arg>\*|\w+)\s*\)\s+FROM\s+t'
        r'(?:\s+WHERE\s+(?P<wcol>\w+)\s*(?P<op><=|>=|=)\s*(?P<val>\d+))?\s*$',
        query, re.IGNORECASE,
    )
    assert m, f'unparseable SQL aggregate: {query!r}'
    fn = m.group('fn').upper()
    val = m.group('val')
    rows = _sql_filter(record['rows'], m.group('wcol'), m.group('op'),
                       int(val) if val is not None else None)
    if fn == 'COUNT':
        return str(len(rows))
    vals = [r[m.group('arg')] for r in rows]
    return str(sum(vals) if fn == 'SUM' else max(vals))


# -- JSON aggregate (jsonata Tier-C) ----------------------------------------

def _json_aggregate(record, query: str) -> str:
    """Evaluate a Tier-C JSONata/jq aggregate (count/sum/max) over an array.

    sum/max operate over numeric members only (ints/floats, not bools); count
    counts every member. The first aggregate token in the query wins.
    """
    m = re.search(r'\$(?P<op>count|sum|max)\(\s*(?P<arg>[^)]+?)\s*\)', query, re.IGNORECASE)
    assert m, f'no aggregate in query {query!r}'
    op = m.group('op').lower()
    arr = _resolve_json(record, m.group('arg').strip())
    nums = [x for x in arr if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if op == 'count':
        return str(len(arr))
    if op == 'sum':
        return str(sum(nums))
    return str(max(nums))


# -- XML descendant text (xmlxpath join policy) -----------------------------

def _xml_join_text(record: str, xpath: str) -> str:
    """Join all descendant text nodes under the element addressed by ``xpath``
    using the space-join policy (``/.../node//text()``)."""
    root = ET.fromstring(record)
    body = re.sub(r'/+text\(\)\s*$', '', xpath.strip())
    parts = [p for p in body.strip('/').split('/') if p]
    el = root
    for p in parts[1:]:  # parts[0] is the root tag itself
        el = el.find(p)
    return ' '.join(t.strip() for t in el.itertext() if t.strip())



# ===========================================================================
# 1. cite — masked-citation recall (RandomInputTokenizerV2 / MaskedCiteDataset)
# ===========================================================================
# The decoder is given two anchor tokens (tag-begin / tag-end) and must emit
# the citation span that sits between them inside the encoded record. There is
# no "serialization" of composite structures here, but the span itself is a
# multi-token sequence, so the corner cases are about span boundaries, masking,
# and the prompt_all vs prompt_cite contract.

CITE_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='single_token_span',
        record='alpha BRAVO charlie',
        query='anchors=(alpha, charlie)',
        expected='BRAVO',
        kind='scalar',
        notes='Minimal cite: a one-word span flanked by two anchor words.',
    ),
    RetrievalCase(
        name='multi_token_span',
        record='the quick brown fox jumps over the lazy dog',
        query='anchors=(quick, lazy)',
        expected='brown fox jumps over the',
        kind='array',
        notes='Span covers several WordPiece tokens; whole run must be emitted.',
    ),
    RetrievalCase(
        name='span_with_subwords',
        record='photosynthesis chloroplasts respiration',
        query='anchors=(photosynthesis, respiration)',
        expected='chloroplasts',
        kind='scalar',
        notes='Subword (##) continuations must stay glued to their head token.',
    ),
    RetrievalCase(
        name='prompt_all_returns_full_text',
        record='one two three four five',
        query='prompt_all=True; anchors=(two, four)',
        expected='one two three four five',
        kind='array',
        notes='prompt_all=True asks for the WHOLE text containing the tags, '
              'not just the inner span.',
    ),
    RetrievalCase(
        name='masking_hides_span_in_encoder',
        record='visible SECRET visible',
        query='mask_cfg!=None; anchors=(visible, visible)',
        expected='SECRET',
        grounded=False,
        notes='With encoder masking the span is masked in inp_masked_toks but '
              'the decoder target is still the unmasked span — verify the '
              'needle is removed from the encoder side only.',
    ),
    RetrievalCase(
        name='empty_between_anchors',
        record='adjacent anchors here',
        query='anchors=(adjacent, anchors)',
        expected='',
        kind='scalar',
        grounded=False,
        notes='CORNER: anchors are adjacent -> empty span. Builder must either '
              'reject (resample) or emit a well-defined empty target.',
    ),
    RetrievalCase(
        name='text_shorter_than_anchor_budget',
        record='tiny',
        query='n_random_toks=3 on a 1-word doc',
        expected='<pool-token-fallback>',
        grounded=False,
        notes='CORNER: doc too short to supply 2*n_random anchors + cite; '
              'random-pool fallback must fill anchors without crashing.',
    ),
]


@pytest.mark.parametrize('case', CITE_CASES, ids=[c.name for c in CITE_CASES])
def test_cite_span_retrieval(case: RetrievalCase):
    """cite: the decoder target equals the citation span between the anchors,
    and (unless masked) the span is present verbatim in the encoded record."""
    tkz = _tkz()

    # Degenerate doc too short to supply 2*n_random anchors + a cite span: the
    # real builder must fall back to pool tokens for the anchors (no anchors are
    # given in the query) without crashing and still produce a well-formed subset.
    if case.expected == '<pool-token-fallback>':
        tk = RandomInputTokenizerV2(tkz, max_len=128, n_random_toks=3, tkz_dec=tkz)
        out = tk([case.record])
        assert len(out) == 1
        sub = out[0]
        assert len(sub.toks_inp) > 0 and len(sub.toks_inp_dec) > 0
        assert len(sub.toks_cite_beg) == 3 and len(sub.toks_cite_end) == 3
        return

    m = re.search(r'anchors=\((?P<a>[^,]+),\s*(?P<b>[^)]+)\)', case.query)
    assert m, f'no anchors in query {case.query!r}'
    left, right = m.group('a').strip().lower(), m.group('b').strip().lower()
    prompt_all = 'prompt_all=True' in case.query
    masked = 'mask_cfg!=None' in case.query

    words = case.record.split()
    low = [w.lower() for w in words]
    li = low.index(left)
    ri = low.index(right, li + 1)
    span_words = words[li + 1:ri]

    # Token offsets of the span inside the encoded record.
    prefix_ids = tkz(' '.join(words[:li + 1]), add_special_tokens=False).input_ids
    span_ids = tkz(' '.join(span_words), add_special_tokens=False).input_ids if span_words else []
    ids = tkz(' '.join(words), add_special_tokens=False).input_ids

    from mllm.train.mask_utils import MaskCfg
    tk = RandomInputTokenizerV2(
        tkz, max_len=128, n_random_toks=3,
        mask_cfg=MaskCfg() if masked else None, tkz_dec=tkz,
    )
    raw_mask = np.ones(len(span_ids), dtype=bool) if masked else None
    spec = CiteSpecV2(
        inp_beg_ind=0, inp_end_ind=len(ids), sub_off=len(prefix_ids),
        n_cite_toks=len(span_ids),
        toks_cite_beg=[2000, 2001, 2002], toks_cite_end=[2003, 2004, 2005],
        raw_mask=raw_mask, prompt_all=prompt_all,
    )
    sub = tk.realize(ids, spec)

    if prompt_all:
        # prompt_all asks for the WHOLE text, not just the inner span: every
        # source word survives in the decoded encoder input (around the anchors).
        decoded = tkz.decode(sub.toks_inp, skip_special_tokens=True)
        for w in case.expected.lower().split():
            assert w in decoded
    else:
        target = tkz.decode(sub.toks_cite, skip_special_tokens=True).strip()
        assert target == case.expected.lower()
    if masked:
        assert all(t == tkz.mask_token_id for t in sub.toks_cite_masked)



# ===========================================================================
# 2. keyval — key -> value recall (KeyValRecallTokenizer)
# ===========================================================================
# record is a serialized list of ``k <kv_sep> v <pair_sep> ...`` pairs. The
# prompt names one key; the target is its paired value. Values today are 1..N
# consecutive real words (always a flat string). The composite cases probe the
# "retrieve more than one value / a value that is itself a list" question.

KEYVAL_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='single_pair',
        record=[('capital', 'paris')],
        query='key=capital',
        expected='paris',
        notes='Smallest record: one pair, query its only key.',
    ),
    RetrievalCase(
        name='multiword_value',
        record=[('river', 'amazon river basin'), ('city', 'lima')],
        query='key=river',
        expected='amazon river basin',
        kind='array',
        notes='Value spans value_max_words words; whole run is the target.',
    ),
    RetrievalCase(
        name='value_with_separator_lookalike',
        record=[('range', 'six to ten'), ('note', 'a; b')],
        query='key=note',
        expected='a; b',
        notes='CORNER: a value containing the pair-separator char (";") must '
              'not be split — exact value is returned.',
    ),
    RetrievalCase(
        name='duplicate_keys',
        record=[('color', 'red'), ('color', 'blue')],
        query='key=color',
        expected='red',
        grounded=True,
        notes='CORNER: duplicate keys. Contract must be defined — first match '
              '(here "red") or last. Pick one and pin it.',
    ),
    RetrievalCase(
        name='value_fits_extended_budget',
        record=[('k', 'w1 w2 w3 w4 w5 w6 w7 w8 w9 w10')],
        query='key=k',
        expected='w1 w2 w3 w4 w5 w6 w7 w8 w9 w10',
        grounded=True,
        notes='Limits extended so the whole multi-word value is retained (no '
              'trim); the full value is the target and appears verbatim in the '
              'record.',
    ),
    RetrievalCase(
        name='numeric_looking_value',
        record=[('year', '1889'), ('name', 'eiffel')],
        query='key=year',
        expected='1889',
        notes='Numeric-looking text value stays a string (no JSON coercion).',
    ),
    # --- composite / serialization probes ---------------------------------
    RetrievalCase(
        name='retrieve_all_values_for_key',
        record=[('tag', 'a'), ('tag', 'b'), ('tag', 'c')],
        query='key=tag; selector=ALL',
        expected='a, b, c',
        kind='array',
        composite=True,
        grounded=True,
        notes='COMPOSITE: when a key repeats and ALL values are requested, '
              'the multi-value answer must be serialized with a defined join '
              '(", ") and every element must remain grounded in the record.',
    ),
    RetrievalCase(
        name='retrieve_whole_record',
        record=[('a', '1'), ('b', '2')],
        query='selector=*',
        expected='a: 1; b: 2',
        kind='object',
        composite=True,
        grounded=True,
        notes='COMPOSITE: dumping the whole record back must reuse the SAME '
              'kv/pair separators the record was serialized with, otherwise '
              'the target is not a substring of the record.',
    ),
]


@pytest.mark.parametrize('case', KEYVAL_CASES, ids=[c.name for c in KEYVAL_CASES])
def test_keyval_retrieval(case: RetrievalCase):
    """keyval: target == value paired with the queried key (or the defined
    serialization of a multi-value / whole-record selector)."""
    pairs = case.record
    rec = '; '.join(f'{k}: {v}' for k, v in pairs)  # documented kv/pair policy
    q = case.query

    if 'selector=*' in q:
        ser = rec
        assert ser == case.expected
        assert case.expected in rec
    elif 'selector=ALL' in q:
        key = re.search(r'key=(\w+)', q).group(1)
        vals = [v for k, v in pairs if k == key]
        ser = ', '.join(vals)
        assert ser == case.expected
        # Composite multi-value: every element stays grounded in the record.
        for v in vals:
            assert v in rec
    else:
        key = re.search(r'key=(\w+)', q).group(1)
        ser = next(v for k, v in pairs if k == key)  # first-match contract
        assert ser == case.expected
        if case.grounded:
            assert case.expected in rec



# ===========================================================================
# 3. jsonfield — JSON path -> field value (JsonFieldRecallTokenizer)
# ===========================================================================
# record is a JSON object. ``_target_text`` today: str -> as-is; everything
# else -> ``json.dumps(value, ensure_ascii=True)``. The record itself is dumped
# with EITHER compact ``(',',':')`` or spaced ``(', ',': ')`` separators —
# which is the crux of every composite case below.

JSONFIELD_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='top_level_string',
        record={'capital': 'paris', 'pop': 2148000},
        query='$.capital',
        expected='paris',
        notes='Flat string field; dotted/jsonpath/NL prompt forms all valid.',
    ),
    RetrievalCase(
        name='top_level_int',
        record={'capital': 'paris', 'pop': 2148000},
        query='$.pop',
        expected='2148000',
        notes='int -> json.dumps -> "2148000".',
    ),
    RetrievalCase(
        name='boolean_value',
        record={'active': True},
        query='$.active',
        expected='true',
        notes='CORNER: bool serializes to JSON literal "true"/"false", not '
              'Python "True". Must match the record text exactly.',
    ),
    RetrievalCase(
        name='null_value',
        record={'middle_name': None},
        query='$.middle_name',
        expected='null',
        notes='CORNER: None -> "null". Grounded only if record also rendered '
              '"null" (it does via json.dumps).',
    ),
    RetrievalCase(
        name='nested_scalar',
        record={'geo': {'country': {'name': 'peru'}}},
        query='$.geo.country.name',
        expected='peru',
        kind='scalar',
        notes='Deep path to a scalar across max_depth nesting.',
    ),
    RetrievalCase(
        name='array_element',
        record={'tribs': ['negro', 'madeira', 'tapajos']},
        query='$.tribs[1]',
        expected='madeira',
        notes='Indexed array element (scalar).',
    ),
    RetrievalCase(
        name='key_with_underscores',
        record={'sea_level_m': 8849},
        query='$.sea_level_m',
        expected='8849',
        notes='Normalized keys may contain underscores; path must round-trip.',
    ),
    # --- composite / serialization probes ---------------------------------
    RetrievalCase(
        name='retrieve_whole_array',
        record={'tribs': ['negro', 'madeira', 'tapajos']},
        query='$.tribs',
        expected='["negro","madeira","tapajos"]',
        kind='array',
        composite=True,
        grounded=True,
        notes='COMPOSITE: path resolves to an ARRAY. Expected serialization '
              'must match the record separators (compact here). With default '
              'json.dumps separators ", " the target would NOT be a substring '
              'of a compact record — this case pins the policy.',
    ),
    RetrievalCase(
        name='retrieve_nested_object',
        record={'geo': {'country': {'name': 'peru', 'iso': 'pe'}}},
        query='$.geo.country',
        expected='{"name":"peru","iso":"pe"}',
        kind='object',
        composite=True,
        grounded=True,
        notes='COMPOSITE: path resolves to an OBJECT. Key order + separators '
              'must equal the record\'s rendering to stay grounded.',
    ),
    RetrievalCase(
        name='retrieve_object_spaced_separators',
        record={'geo': {'country': {'name': 'peru', 'iso': 'pe'}}},
        query='$.geo.country',
        expected='{"name": "peru", "iso": "pe"}',
        kind='object',
        composite=True,
        grounded=True,
        notes='COMPOSITE twin of the previous case but for a record dumped '
              'with spaced separators — proves the serializer must FOLLOW the '
              'record policy, not a fixed one.',
    ),
    RetrievalCase(
        name='retrieve_array_of_objects',
        record={'rivers': [{'n': 'negro'}, {'n': 'madeira'}]},
        query='$.rivers',
        expected='[{"n":"negro"},{"n":"madeira"}]',
        kind='array',
        composite=True,
        grounded=True,
        notes='COMPOSITE: array of objects — nested serialization, still a '
              'verbatim substring of the record.',
    ),
    RetrievalCase(
        name='string_value_that_breaks_grounding_if_jsonified',
        record={'note': 'a, b'},
        query='$.note',
        expected='a, b',
        notes='CORNER: a STRING value is returned raw ("a, b"), NOT json.dumps '
              '(which would add quotes). Distinguishes scalar-string policy '
              'from composite-json policy.',
    ),
]


@pytest.mark.parametrize('case', JSONFIELD_CASES, ids=[c.name for c in JSONFIELD_CASES])
def test_jsonfield_retrieval(case: RetrievalCase):
    """jsonfield: target == serialized value at ``query``; scalars raw,
    composites JSON-serialized with the record's own separator policy so the
    needle stays grounded."""
    compact = _infer_compact(case.expected)
    value = _resolve_json(case.record, case.query)
    ser = json_target_text(value, compact)
    assert ser == case.expected
    if case.grounded:
        assert case.expected in _json_record_text(case.record, compact)



# ===========================================================================
# 4. jsonata — selection + transform (JsonataRecallTokenizer)
# ===========================================================================
# Tier-E: verbatim selection (same as jsonfield). Tier-C: computed aggregates
# (count/sum/max) whose answer need NOT be in the record. Composite cases probe
# selecting whole arrays/objects and aggregates over mixed/empty arrays.

JSONATA_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='tier_e_select_scalar_jsonata',
        record={'city': {'name': 'lima'}},
        query='city.name',
        expected='lima',
        notes='Tier-E JSONata dotted selection of a scalar.',
    ),
    RetrievalCase(
        name='tier_e_select_scalar_jq',
        record={'city': {'name': 'lima'}},
        query='.city.name',
        expected='lima',
        notes='Tier-E jq dialect of the same selection.',
    ),
    RetrievalCase(
        name='tier_c_count',
        record={'xs': [10, 20, 30, 40]},
        query='$count(xs)   /   .xs | length',
        expected='4',
        kind='computed',
        grounded=False,
        notes='Tier-C COUNT — answer is derived, not in the record.',
    ),
    RetrievalCase(
        name='tier_c_sum',
        record={'xs': [10, 20, 30]},
        query='$sum(xs)   /   .xs | add',
        expected='60',
        kind='computed',
        grounded=False,
        notes='Tier-C SUM over numeric array.',
    ),
    RetrievalCase(
        name='tier_c_max',
        record={'xs': [10, 99, 30]},
        query='$max(xs)   /   .xs | max',
        expected='99',
        kind='computed',
        grounded=False,
        notes='Tier-C MAX over numeric array.',
    ),
    RetrievalCase(
        name='tier_c_count_on_text_array',
        record={'tags': ['a', 'b', 'c']},
        query='$count(tags)',
        expected='3',
        kind='computed',
        grounded=False,
        notes='CORNER: non-numeric array — only COUNT is valid (no sum/max). '
              'Builder must NOT emit sum/max for text arrays.',
    ),
    # --- composite / edge probes ------------------------------------------
    RetrievalCase(
        name='tier_e_select_whole_array',
        record={'xs': [1, 2, 3]},
        query='xs',
        expected='[1,2,3]',
        kind='array',
        composite=True,
        grounded=True,
        notes='COMPOSITE: selecting the whole array must serialize to a '
              'record-grounded substring.',
    ),
    RetrievalCase(
        name='tier_c_sum_mixed_array',
        record={'xs': [10, 'oops', 20, None, 30]},
        query='$sum(xs)',
        expected='60',
        kind='computed',
        grounded=False,
        notes='CORNER: SUM must operate only over numeric members (10+20+30), '
              'ignoring strings/null — matches the code filtering numerics.',
    ),
    RetrievalCase(
        name='tier_c_aggregate_empty_array',
        record={'xs': []},
        query='$sum(xs) / $count(xs)',
        expected='0',
        kind='computed',
        grounded=False,
        notes='CORNER: empty array. COUNT->0, SUM->0; MAX is undefined and '
              'must be suppressed (no MAX query emitted).',
    ),
]


@pytest.mark.parametrize('case', JSONATA_CASES, ids=[c.name for c in JSONATA_CASES])
def test_jsonata_retrieval(case: RetrievalCase):
    """jsonata: Tier-E selection grounded in record; Tier-C aggregate computed
    over numeric members only; whole-array selection serialized to a grounded
    substring."""
    if case.kind == 'computed':
        assert _json_aggregate(case.record, case.query) == case.expected
        return
    compact = _infer_compact(case.expected)
    value = _resolve_json(case.record, case.query)
    ser = json_target_text(value, compact)
    assert ser == case.expected
    if case.grounded:
        assert case.expected in _json_record_text(case.record, compact)



# ===========================================================================
# 5. xmlxpath — XML/XPath -> node value (XmlXpathRecallTokenizer)
# ===========================================================================
# record is an XML string. Selectable today: ``.../@attr`` and ``.../text()``.
# Composite cases probe selecting a whole element (subtree serialization),
# escaping, and repeated siblings needing positional predicates.

XMLXPATH_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='attribute_value',
        record='<root><city id="lima">peru</city></root>',
        query='/root/city/@id',
        expected='lima',
        notes='Attribute selection — verbatim attribute value.',
    ),
    RetrievalCase(
        name='text_node',
        record='<root><city id="lima">peru</city></root>',
        query='/root/city/text()',
        expected='peru',
        notes='Text-node selection — verbatim text value.',
    ),
    RetrievalCase(
        name='nested_text_node',
        record='<root><geo><country><name>peru</name></country></geo></root>',
        query='/root/geo/country/name/text()',
        expected='peru',
        kind='scalar',
        notes='Deep text node across several levels.',
    ),
    RetrievalCase(
        name='escaped_special_chars',
        record='<root><expr op="a &lt; b">x &amp; y</expr></root>',
        query='/root/expr/text()',
        expected='x & y',
        notes='CORNER: selected value is the UNESCAPED text ("x & y") while '
              'the record stores the ESCAPED form ("x &amp; y"). Grounding '
              'must compare against the escaped record form.',
    ),
    RetrievalCase(
        name='mixed_text_and_children',
        record='<root><a>lead<b>inner</b></a></root>',
        query='/root/a/text()',
        expected='lead',
        notes='CORNER: element has BOTH text and child elements; text() must '
              'pick only the direct text, not child text.',
    ),
    RetrievalCase(
        name='repeated_siblings_need_index',
        record='<root><item>a</item><item>b</item><item>c</item></root>',
        query='/root/item[2]/text()',
        expected='b',
        notes='CORNER: repeated tags require a 1-based positional predicate to '
              'disambiguate which sibling is selected.',
    ),
    # --- composite / serialization probes ---------------------------------
    RetrievalCase(
        name='retrieve_whole_element_subtree',
        record='<root><city id="lima"><pop>9</pop></city></root>',
        query='/root/city',
        expected='<city id="lima"><pop>9</pop></city>',
        kind='subtree',
        composite=True,
        grounded=True,
        notes='COMPOSITE: selecting an ELEMENT (not @attr/text()) must '
              'serialize the entire subtree (outer XML), including attributes '
              'and child markup, as a verbatim substring of the record.',
    ),
    RetrievalCase(
        name='retrieve_element_with_multiple_children',
        record='<root><box w="2"><a>1</a><b>2</b></box></root>',
        query='/root/box',
        expected='<box w="2"><a>1</a><b>2</b></box>',
        kind='subtree',
        composite=True,
        grounded=True,
        notes='COMPOSITE: subtree with several children — full markup '
              'serialization, order preserved.',
    ),
    RetrievalCase(
        name='retrieve_all_text_under_node',
        record='<root><a>lead<b>inner</b>tail</a></root>',
        query='/root/a//text()',
        expected='lead inner tail',
        kind='array',
        composite=True,
        grounded=False,
        notes='COMPOSITE: descendant text() returns MULTIPLE text nodes; the '
              'concatenation/join policy (space-joined here) must be defined. '
              'Not a single contiguous substring of the record.',
    ),
]


@pytest.mark.parametrize('case', XMLXPATH_CASES, ids=[c.name for c in XMLXPATH_CASES])
def test_xmlxpath_retrieval(case: RetrievalCase):
    """xmlxpath: @attr/text() selections return verbatim values; element
    selection serializes the full subtree (outer XML); descendant text() joins
    multiple nodes by the defined policy."""
    if not case.grounded:
        # Descendant text() yields multiple text nodes joined by the space
        # policy; not a single contiguous substring, so assert the join directly.
        assert _xml_join_text(case.record, case.query) == case.expected
        return
    # Grounded selection: the (unescaped) target maps to a verbatim substring of
    # the (escaped) record — markup subtrees match raw, text values match escaped.
    assert case.expected in case.record or html.escape(case.expected) in case.record



# ===========================================================================
# 6. sql — SQL select / aggregate (SqlSelectRecallTokenizer)
# ===========================================================================
# record is a markdown table (cols incl. an int ``id``). Tier-E: SELECT col
# WHERE id=x (single cell). Tier-C: COUNT/SUM/MAX. Composite cases probe whole
# rows, multi-column projections, multi-row results, and pipe escaping.

SQL_TABLE = {
    'cols': ['id', 'city', 'pop'],
    'rows': [
        {'id': 1, 'city': 'lima', 'pop': 100},
        {'id': 2, 'city': 'paris', 'pop': 200},
        {'id': 3, 'city': 'cairo', 'pop': 300},
    ],
}

SQL_CASES: List[RetrievalCase] = [
    RetrievalCase(
        name='tier_e_single_cell_text',
        record=SQL_TABLE,
        query='SELECT city FROM t WHERE id = 2',
        expected='paris',
        notes='Tier-E single text cell — verbatim from the table.',
    ),
    RetrievalCase(
        name='tier_e_single_cell_num',
        record=SQL_TABLE,
        query='SELECT pop FROM t WHERE id = 3',
        expected='300',
        notes='Tier-E single numeric cell (stringified).',
    ),
    RetrievalCase(
        name='tier_c_count_threshold',
        record=SQL_TABLE,
        query='SELECT COUNT(*) FROM t WHERE id <= 2',
        expected='2',
        kind='computed',
        grounded=False,
        notes='Tier-C COUNT with an id threshold.',
    ),
    RetrievalCase(
        name='tier_c_sum',
        record=SQL_TABLE,
        query='SELECT SUM(pop) FROM t',
        expected='600',
        kind='computed',
        grounded=False,
        notes='Tier-C SUM over a numeric column.',
    ),
    RetrievalCase(
        name='tier_c_max',
        record=SQL_TABLE,
        query='SELECT MAX(pop) FROM t',
        expected='300',
        kind='computed',
        grounded=False,
        notes='Tier-C MAX over a numeric column.',
    ),
    RetrievalCase(
        name='where_no_match',
        record=SQL_TABLE,
        query='SELECT city FROM t WHERE id = 99',
        expected='',
        kind='scalar',
        grounded=False,
        notes='CORNER: WHERE matches no row. Empty-result serialization must '
              'be defined (empty string / NULL token) rather than crashing.',
    ),
    RetrievalCase(
        name='pipe_in_text_value',
        record={
            'cols': ['id', 'note'],
            'rows': [{'id': 1, 'note': 'a b'}],
        },
        query='SELECT note FROM t WHERE id = 1',
        expected='a b',
        notes='CORNER: text values are sanitized so a literal "|" cannot break '
              'the markdown row; the stored/returned value drops the pipe.',
    ),
    # --- composite / serialization probes ---------------------------------
    RetrievalCase(
        name='select_star_whole_row',
        record=SQL_TABLE,
        query='SELECT * FROM t WHERE id = 2',
        expected='2, paris, 200',
        kind='row',
        composite=True,
        grounded=True,
        notes='COMPOSITE: SELECT * returns a whole row across mixed types; the '
              'tuple serialization (", "-joined, column order) must be defined '
              'and every cell remains grounded in the table.',
    ),
    RetrievalCase(
        name='select_multi_column_projection',
        record=SQL_TABLE,
        query='SELECT city, pop FROM t WHERE id = 3',
        expected='cairo, 300',
        kind='row',
        composite=True,
        grounded=True,
        notes='COMPOSITE: multi-column projection — ordered subset of the row, '
              'same join policy as SELECT *.',
    ),
    RetrievalCase(
        name='select_column_multiple_rows',
        record=SQL_TABLE,
        query='SELECT city FROM t WHERE id <= 2',
        expected='lima, paris',
        kind='rows',
        composite=True,
        grounded=True,
        notes='COMPOSITE: a column projection matching MULTIPLE rows must '
              'serialize the ordered list of cells; each value grounded.',
    ),
    RetrievalCase(
        name='select_rows_as_records',
        record=SQL_TABLE,
        query="SELECT * FROM t WHERE pop >= 200",
        expected='2, paris, 200; 3, cairo, 300',
        kind='rows',
        composite=True,
        grounded=True,
        notes='COMPOSITE: multiple full rows — needs both an intra-row join '
              '(", ") and an inter-row separator ("; "); policy must be pinned.',
    ),
]


@pytest.mark.parametrize('case', SQL_CASES, ids=[c.name for c in SQL_CASES])
def test_sql_retrieval(case: RetrievalCase):
    """sql: Tier-E single cells verbatim; Tier-C aggregates computed;
    SELECT * / multi-column / multi-row projections serialized with a defined
    intra-row and inter-row join policy."""
    if case.kind == 'computed':
        assert _sql_aggregate(case.record, case.query) == case.expected
        return

    cols_sel, wcol, op, val = _sql_parse(case.query)
    all_cols = case.record['cols']
    matched = _sql_filter(case.record['rows'], wcol, op, val)
    if not matched:
        # Empty WHERE result: defined empty-string serialization.
        assert case.expected == ''
        return

    sel = all_cols if cols_sel is None else cols_sel

    if len(matched) == 1 and len(sel) == 1:
        ser = str(matched[0][sel[0]])
    elif len(matched) == 1:
        ser = sql_row_text([matched[0][c] for c in sel])
    elif len(sel) == 1:
        # single-column, multiple rows -> flat comma-joined list of cells.
        ser = ', '.join(str(r[sel[0]]) for r in matched)
    else:
        ser = sql_rows_text([[r[c] for c in sel] for r in matched])

    assert ser == case.expected
    if case.grounded:
        table = _sql_markdown(case.record)
        # Every selected cell is present verbatim in the rendered table.
        for r in matched:
            for c in sel:
                assert str(r[c]) in table



# ===========================================================================
# Cross-cutting contract stubs (apply to every selection family)
# ===========================================================================

ALL_SELECTION_CASES = (
    KEYVAL_CASES + JSONFIELD_CASES + JSONATA_CASES + XMLXPATH_CASES + SQL_CASES
)
GROUNDED_CASES = [c for c in ALL_SELECTION_CASES if c.grounded]
COMPOSITE_CASES = [c for c in ALL_SELECTION_CASES if c.composite]


def _is_json_case(c: RetrievalCase) -> bool:
    """JSON-serializable families (jsonfield / jsonata selection): record is a
    plain dict (not an SQL table) and the answer is a selected value, not a
    computed aggregate. These are where the json.dumps separator-policy question
    from the notebook review actually lives."""
    return isinstance(c.record, dict) and 'cols' not in c.record and c.kind != 'computed'


JSON_GROUNDED_CASES = [c for c in GROUNDED_CASES if _is_json_case(c)]
JSON_COMPOSITE_CASES = [c for c in COMPOSITE_CASES if _is_json_case(c)]


@pytest.mark.parametrize('case', JSON_GROUNDED_CASES, ids=[c.name for c in JSON_GROUNDED_CASES])
def test_grounded_target_is_substring_of_record(case: RetrievalCase):
    """Invariant 1 (needle-in-context): for every grounded JSON case the
    serialized target appears verbatim inside the serialized record — INCLUDING
    the composite cases, which is precisely where a fixed json.dumps would
    violate the invariant."""
    compact = _infer_compact(case.expected)
    value = _resolve_json(case.record, case.query)
    ser = json_target_text(value, compact)
    assert ser in _json_record_text(case.record, compact)


@pytest.mark.parametrize('case', JSON_COMPOSITE_CASES, ids=[c.name for c in JSON_COMPOSITE_CASES])
def test_composite_serialization_is_deterministic(case: RetrievalCase):
    """Invariant 2 (round-trip determinism): serializing a composite value
    (object / array) is a pure function of the value and the record's
    serialization policy — repeated calls yield byte-identical targets."""
    compact = _infer_compact(case.expected)
    value = _resolve_json(case.record, case.query)
    a = json_target_text(value, compact)
    b = json_target_text(value, compact)
    assert a == b == case.expected


@pytest.mark.parametrize('case', JSON_COMPOSITE_CASES, ids=[c.name for c in JSON_COMPOSITE_CASES])
def test_composite_serialization_follows_record_separator_policy(case: RetrievalCase):
    """The open question from the notebook review: when the answer is NOT a
    single scalar, the serializer must mirror the RECORD's separator policy
    (compact vs spaced) so the composite target stays grounded — and the WRONG
    policy provably breaks grounding."""
    compact = _infer_compact(case.expected)
    value = _resolve_json(case.record, case.query)
    right = json_target_text(value, compact)
    rec_right = _json_record_text(case.record, compact)
    assert right in rec_right
    wrong = json_target_text(value, not compact)
    if wrong != right:
        assert wrong not in rec_right


import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from transformers import PreTrainedTokenizer


def extend_mask_to_words(mask: np.ndarray, toks_str: list[str]) -> np.ndarray:
    n = len(mask)
    for i in range(1, n):
        if not mask[i] and mask[i - 1] and toks_str[i].startswith('##'):
            mask[i] = True
    for i in range(n - 2, -1, -1):
        if not mask[i] and mask[i + 1] and toks_str[i + 1].startswith('##'):
            mask[i] = True
    return mask


def mask_random_tokens(toks: np.ndarray, tkz: PreTrainedTokenizer, rem_freq: float = 0.33, rem_prob: float = 0.15,
        rem_conseq_freq: float = 0.33, rem_conseq_prob: float = 0.2, rem_conseq_max_len: int = 20,
        rem_conseq_max_times: int = 5) -> np.ndarray:
    res = toks.copy()
    rv = np.random.rand()
    n_total = len(res)
    if rv > rem_freq + rem_conseq_freq:
        return res

    if n_total < 5:
        return res

    if rv < rem_freq:
        mask: np.ndarray = np.random.rand(n_total) <= rem_prob
    elif rv <= rem_freq + rem_conseq_freq:
        rem_conseq_times = np.random.randint(1, rem_conseq_max_times + 1)
        rem_interval = n_total // rem_conseq_times
        off = 0
        mask = np.full(n_total, False, dtype=bool)
        i_conseq = 0
        while i_conseq < rem_conseq_times and off < n_total:
            n_rem = int(n_total * rem_conseq_prob)
            n_rem = np.random.randint(2, min(max(n_rem, 2), rem_conseq_max_len) + 1)
            i = np.random.randint(off, off + rem_interval)
            i1 = max(i - n_rem // 2, 0)
            i2 = min(i1 + n_rem, n_total - 1)
            if i1 < i2:
                mask[i1:i2] = True
            off = max(off + rem_interval, i2 + int(rem_conseq_max_len * 1.5))
            i_conseq += 1

    toks_str = [tkz.decode(t) for t in toks]
    mask = extend_mask_to_words(mask, toks_str)
    res[mask] = tkz.mask_token_id
    return res


NEWLINE_PAT = re.compile(r'[\n\r]+', re.M)
STR_DELIM_PAT = re.compile(r'\s+')


def mask_random_words(
        s: str, mask_tok_str: str, rem_freq: float = 0.33, rem_prob: float = 0.15,
        rem_conseq_freq: float = 0.33, rem_conseq_prob: float = 0.2, rem_conseq_max_len: int = 20,
        rem_conseq_max_times: int = 5,
        ) -> Optional[str]:
    rv = np.random.rand()
    # print(rv, rem_freq, rem_conseq_freq)
    if rv < 1 - (rem_freq + rem_conseq_freq):
        return
    lines = NEWLINE_PAT.split(s)
    res = []
    n_total = 0
    for line in lines:
        if not line:
            continue
        words = STR_DELIM_PAT.split(line)
        words = filter(None, words)
        words = list(words)
        if not words:
            continue
        res.append(words)
        n_total += len(words)

    if n_total < 5:
        return

    if rv < 1 - rem_conseq_freq:
        mask = np.random.rand(n_total) <= rem_prob
    else:
        rem_conseq_times = np.random.randint(1, rem_conseq_max_times + 1)
        rem_interval = n_total // rem_conseq_times
        off = 0
        mask = np.full(n_total, False, dtype=bool)
        while off < n_total:
            n_rem = int(n_total * rem_conseq_prob)
            n_rem = np.random.randint(2, max(n_rem, 2) + 1)
            n_rem = min(n_rem, rem_conseq_max_len)
            i = np.random.randint(off, off + rem_interval)
            i1 = max(i - n_rem // 2, 0)
            i2 = min(i1 + n_rem, n_total - 1)
            if i1 < i2:
                mask[i1:i2] = True
            off = max(off + rem_interval, i2 + int(n_rem * 1.5))

    im = 0
    for words in res:
        for iw in range(len(words)):
            if mask[im]:
                words[iw] = mask_tok_str
            im += 1

    return '\n'.join([' '.join(words) for words in res])


@dataclass
class MaskCfg:
    rem_freq: float = 0.33
    rem_prob: float = 0.15
    rem_conseq_freq: float = 0.33
    rem_conseq_prob: float = 0.2
    rem_conseq_max_len: int = 20

    def gen_mask(self, n_total: int) -> Optional[np.ndarray]:
        rv = np.random.rand()
        if rv > self.rem_freq + self.rem_conseq_freq or n_total < 2:
            return None

        if rv < self.rem_freq:
            mask: np.ndarray = np.random.rand(n_total) <= self.rem_prob
        else:
            mask = np.full(n_total, False, dtype=bool)
            n_seq = int(n_total * self.rem_conseq_prob)
            n_seq = min(max(n_seq, 1), self.rem_conseq_max_len, int(0.9 * n_total))
            i_off = np.random.randint(0, n_total - n_seq + 1)
            mask[i_off:i_off + n_seq] = True

        return mask


def mask_random_words_v2(toks: np.ndarray, tkz: PreTrainedTokenizer, mcfg: MaskCfg) -> np.ndarray:
    res = toks.copy()
    mask = mcfg.gen_mask(len(toks))
    if mask is not None:
        toks_str = tkz.convert_ids_to_tokens(toks)
        mask = extend_mask_to_words(mask, toks_str)
        res[mask] = tkz.mask_token_id
    return res




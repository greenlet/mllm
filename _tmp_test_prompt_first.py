"""Temporary verification of prompt_first config + sequence ordering. Deleted after use."""
import types
import torch
from torch import nn

from mllm.config.model import (
    create_mixed_decoder_cfg, copy_override_mixed_decoder_cfg, gen_prefpostfix_mixed_decoder,
)
from mllm.model.mixed_decoder import MixedDecoder

# --- 1. Config round-trip + folder naming ---
cfg = create_mixed_decoder_cfg(prompt_first=True)
assert cfg.prompt_first is True
cfg2 = copy_override_mixed_decoder_cfg(cfg)
assert cfg2.prompt_first is True
cfg_def = create_mixed_decoder_cfg()
assert cfg_def.prompt_first is False
_, post_t = gen_prefpostfix_mixed_decoder(cfg)
_, post_f = gen_prefpostfix_mixed_decoder(cfg_def)
assert 'pfirstT' in post_t, post_t
assert 'pfirst' not in post_f, post_f
ovr = copy_override_mixed_decoder_cfg(cfg_def, prompt_first=True)
assert ovr.prompt_first is True
print('CONFIG OK:', post_t.split("-")[-2:], '|', post_f.split("-")[-2:])

# --- 2. build_decoder_input ordering via lightweight stub ---
d_dec = 8
vocab = 50
stub = types.SimpleNamespace()
stub.enc_proj = None
stub.sep_token_id = 7
stub.word_embeddings = nn.Embedding(vocab, d_dec)
stub.pos_emb = None
stub.d_dec = d_dec

def make_cfg(use_sep, prompt_first, max_seq_len=100):
    return types.SimpleNamespace(use_sep=use_sep, prompt_first=prompt_first, max_seq_len=max_seq_len)

B, N, P, T = 2, 3, 4, 5
ctx_embs = torch.randn(B, N, d_dec)
prompt_toks = torch.randint(0, vocab, (B, P))
prompt_att = torch.ones(B, P, dtype=torch.long)
target_toks = torch.randint(0, vocab, (B, T))
target_att = torch.ones(B, T, dtype=torch.long)

fn = MixedDecoder.build_decoder_input

for use_sep in (True, False):
    for prompt_first in (False, True):
        stub.cfg = make_cfg(use_sep, prompt_first)
        emb, mask, labels, tsi = fn(
            stub, ctx_embs, prompt_toks, prompt_att, target_toks, target_att, include_prompt=True,
        )
        sep_len = 1 if use_sep else 0
        prefix = N + sep_len + P
        total = prefix + (T - 1)
        assert emb.shape == (B, total, d_dec), emb.shape
        assert tsi == prefix - 1, (tsi, prefix)
        # labels: -100 before target_start, target tokens after
        assert torch.all(labels[:, :tsi] == -100)
        assert torch.all(labels[:, tsi:tsi + T] == target_toks)
        # Verify ordering: reconstruct expected prefix embeddings (no pos emb added).
        sep_emb = stub.word_embeddings(torch.full((B, 1), stub.sep_token_id))
        p_emb = stub.word_embeddings(prompt_toks)
        if prompt_first:
            blocks = [p_emb] + ([sep_emb] if use_sep else []) + [ctx_embs]
        else:
            blocks = [ctx_embs] + ([sep_emb] if use_sep else []) + [p_emb]
        expected_prefix = torch.cat(blocks, dim=1)
        assert torch.allclose(emb[:, :prefix], expected_prefix, atol=1e-6), (use_sep, prompt_first)
        print(f'BUILD OK use_sep={use_sep} prompt_first={prompt_first} total={total} tsi={tsi}')

print('ALL OK')

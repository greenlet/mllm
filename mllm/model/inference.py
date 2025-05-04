from typing import Optional, Callable

import numpy as np
import torch


class Beam:
    max_len: int
    next_token_id: int
    last_token_id: int
    tokens_cur: list[int]
    log_prob: float
    append_next_token_id: bool
    tokens_inp: list[int]
    finished: bool

    def __init__(self, max_len: int, next_token_id: int, last_token_id: int, tokens: Optional[list[int]] = None,
                 log_prob: float = 0, append_next_token_id: bool = True):
        self.max_len = max_len
        self.next_token_id = next_token_id
        self.last_token_id = last_token_id
        self.log_prob = log_prob
        self.append_next_token_id = append_next_token_id
        if self.append_next_token_id:
            self.tokens_cur = [] if tokens is None else [*tokens]
            self.tokens_inp = [*self.tokens_cur, self.next_token_id]
        else:
            self.tokens_cur = [self.next_token_id] if not tokens else tokens
            self.tokens_inp = self.tokens_cur
        self.finished = self.tokens_cur and (self.tokens_cur[-1] == self.last_token_id or len(self.tokens_cur) == self.max_len)

    # token_ids: [num_beams]
    # probs: [num_beams]
    def next(self, token_ids: torch.Tensor, probs: torch.Tensor) -> list['Beam']:
        token_ids, probs = token_ids.squeeze().tolist(), probs.squeeze().tolist()
        res = []
        for token_id, prob in zip(token_ids, probs):
            beam = Beam(
                max_len=self.max_len,
                next_token_id=self.next_token_id,
                last_token_id=self.last_token_id,
                tokens=[*self.tokens_cur, token_id],
                log_prob=self.log_prob + np.log(prob),
                append_next_token_id=self.append_next_token_id,
            )
            res.append(beam)
        return res


class BeamSearch:
    num_beams: int
    max_len: int
    temperature: float
    next_token_id: int
    last_token_id: int
    device: torch.device
    append_next_token_id: bool
    active_beams: list[Beam]
    finished_beams: list[Beam]

    def __init__(self, num_beams: int, max_len: int, temperature: float, next_token_id: int, last_token_id: int,
                 device: torch.device, append_next_token_id: bool = True):
        self.num_beams = num_beams
        self.max_len = max_len
        self.temperature = temperature
        self.next_token_id = next_token_id
        self.last_token_id = last_token_id
        self.device = device
        self.append_next_token_id = append_next_token_id
        self.active_beams = [Beam(max_len=self.max_len, next_token_id=self.next_token_id, last_token_id=self.last_token_id, append_next_token_id=self.append_next_token_id)]
        self.finished_beams = []

    def run(self, run_inference: Callable[[torch.Tensor], torch.Tensor]) -> list[Beam]:
        while self.active_beams:
            inp_tokens = [beam.tokens_inp for beam in self.active_beams]
            # [n_active_beams, seq_len]
            inp_tokens_t = torch.tensor(inp_tokens, dtype=torch.long, device=self.device)
            # [n_active_beams, vocab_size]
            logits = run_inference(inp_tokens_t)
            # [n_active_beams, vocab_size]
            scaled_logits = logits / self.temperature
            # [n_active_beams, vocab_size]
            probs = torch.softmax(scaled_logits, dim=-1)
            # [n_active_beams, num_beams]
            top_probs, top_ids = probs.topk(self.num_beams, dim=-1)

            new_beams = []
            for ib, beam in enumerate(self.active_beams):
                next_token_ids, next_token_probs = top_ids[ib], top_probs[ib]
                # print(next_token_ids)
                # print(next_token_probs)
                new_beams += beam.next(next_token_ids, next_token_probs)

            new_beams.sort(key=lambda beam: -beam.log_prob)
            n_active_beams = self.num_beams - len(self.finished_beams)
            active_beams = []
            for i in range(n_active_beams):
                beam = new_beams[i]
                if beam.finished:
                    self.finished_beams.append(beam)
                else:
                    active_beams.append(beam)
            self.active_beams = active_beams

        self.finished_beams.sort(key = lambda beam: -beam.log_prob)
        return self.finished_beams

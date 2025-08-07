import torch


# t: [batch_size, seq_len, d_model]
def get_top_vects(t: torch.Tensor, n: int, calc_cos: bool) -> torch.Tensor:
    # [batch_size, seq_len, d_model]
    t1 = t
    if calc_cos:
        # [batch_size, seq_len, 1]
        tn = torch.linalg.norm(t1, dim=2, keepdim=True)
        # [batch_size, seq_len, d_model]
        t1 = t1 / tn
    # [batch_size, d_model, seq_len]
    t2 = t.transpose(1, 2)
    # [batch_size, seq_len, seq_len]
    cos_dists = torch.matmul(t1, t2)
    # [batch_size, seq_len, seq_len]
    probs = torch.softmax(cos_dists, dim=2)
    # [batch_size, seq_len]
    probs_sum = torch.sum(probs, dim=1)
    # [batch_size, n]
    _, inds = torch.topk(probs_sum, n, dim=1)
    batch = []
    for ib in range(t.shape[0]):
        batch.append(t[ib, inds[ib]])
    # [batch_size, n, d_model]
    res = torch.stack(batch, dim=0)
    return res



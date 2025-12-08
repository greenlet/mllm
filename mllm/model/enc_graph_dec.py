
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from mllm.config.model import EncGraphDecCfg


class EncGraphDec(nn.Module):

    def __init__(self, cfg: EncGraphDecCfg):
        super(EncGraphDec, self).__init__()
        self.cfg = cfg
        self.graph = GCNConv(cfg.enc_out_emb_dim, cfg.dec_inp_emb_dim)

    def forward(self, x, edge_index):
        encoded = self.encoder(x, edge_index)
        decoded = self.decoder(encoded, edge_index)
        return decoded


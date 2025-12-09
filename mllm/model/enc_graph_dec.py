
import torch
from torch import nn
from torch_geometric.nn import GCNConv

from mllm.config.model import GraphEncoderCfg, GraphDecoderCfg, GraphEncoderDecoderCfg
from mllm.model.bert.modeling_bert import BertModel



class GraphEncoder(nn.Module):
    def __init__(self, encoder_cfg: GraphEncoderCfg):
        super(GraphEncoder, self).__init__()
        self.bert = BertModel(encoder_cfg.bert_cfg)

    # inp: (batch_size, seq_len)
    def forward(self, inp: torch.Tensor):
        out = self.bert(inp)
        out = out.last_hidden_state[:, 0]
        return out


class GraphDecoder(nn.Module):
    def __init__(self, decoder_cfg: GraphDecoderCfg):
        super().__init__()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class GraphEncdec(nn.Module):

    def __init__(self, cfg: GraphEncoderDecoderCfg):
        super().__init__()
        self.cfg = cfg
        self.graph = GCNConv(cfg.enc_out_emb_dim, cfg.dec_inp_emb_dim)

    def forward(self, x, edge_index):
        encoded = self.encoder(x, edge_index)
        decoded = self.decoder(encoded, edge_index)
        return decoded


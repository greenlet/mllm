from torch import nn, Tensor

from mllm.config.model import EncoderConvCfg



class ConvLayer(nn.Module):
    cfg: EncoderConvCfg
    conv: nn.Conv1d
    glu: nn.GLU
    dropout: nn.Dropout

    def __init__(self, cfg: EncoderConvCfg):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Conv1d(in_channels=cfg.d_model, out_channels=cfg.d_model * 2, kernel_size=cfg.conv_kernel_size)
        self.glu = nn.GLU(dim=-1)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, inp: Tensor) -> Tensor:
        out = self.conv(inp)
        out = self.glu(out)
        out = self.dropout(out)
        out += inp
        return out


class EncoderConv(nn.Module):
    cfg: EncoderConvCfg
    layers: nn.ModuleList

    def __init__(self, cfg: EncoderConvCfg):
        super().__init__()
        self.cfg = cfg

        layers = []
        if cfg.share_layer_weights:
            n_levels = 1
        else:
            n_levels = cfg.n_levels
        for i_level in range(n_levels):
            for i_layer in range(cfg.n_layers_per_level):
                conv = ConvLayer(cfg)
                layers.append(conv)
            pool = nn.MaxPool1d(kernel_size=cfg.pool_kernel_size, stride=cfg.pool_stride)
            layers.append(pool)
        self.layers = nn.ModuleList(layers)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p, -0.1, 0.1)

    # [batch_size, seq_len, d_model]
    def forward(self, inp: Tensor) -> Tensor:
        out = inp
        n = self.cfg.n_levels if self.cfg.share_layer_weights else 1
        for i in range(n):
            out = self.layers(out)

        return out


'''
Contrastive Predictive Coding (CPC) model implementation.
'''
import torch
import torch.nn as nn


class CPCModel(nn.Module):
    def __init__(self, encoder: nn.Module, ar_model: nn.Module, n_future: int):
        super().__init__()
        self.encoder = encoder
        self.ar_model = ar_model
        self.n_future = n_future
        self.predictor = nn.ModuleList([
            nn.Linear(ar_model.output_dim, encoder.output_dim) for _ in range(n_future)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # Encode input
        c = self.ar_model(z)  # Autoregressive model
        predictions = [pred(c) for pred in self.predictor]  # Future predictions
        return torch.stack(predictions, dim=1)  # Shape: (batch_size, n_future, feature_dim)



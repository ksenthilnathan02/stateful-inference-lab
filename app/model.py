import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim = 32,d_model = 64, nhead = 4, num_layers = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        h = self.encoder(x)
        return h.mean(dim=1)  # pooled embedding

class PredictionHead(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, h):
        return torch.sigmoid(self.fc(h))
    
class Model:
    def __init__(self):
        self.encoder = TransformerEncoder()
        self.head = PredictionHead()
        self.encoder.eval()
        self.head.eval()

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def predict(self, h):
        with torch.no_grad():
            return self.head(h)
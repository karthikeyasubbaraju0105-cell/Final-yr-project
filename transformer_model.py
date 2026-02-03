import torch
import torch.nn as nn

class EdgeTransformer(nn.Module):
    """
    A small Transformer Encoder to predict next CPU load for an edge server.
    - Input: sequence of past CPU loads (float values)
    - Output: predicted next CPU load (float)
    """

    def __init__(self, seq_len=20, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Linear embedding for 1D CPU values → vector
        self.input_proj = nn.Linear(1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project final sequence output to a single prediction
        self.output_proj = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x: shape (batch, seq_len, 1)
        """
        x = self.input_proj(x)                # (batch, seq_len, embed_dim)
        x = self.transformer(x)               # (batch, seq_len, embed_dim)
        last = x[:, -1, :]                    # final element of sequence
        out = self.output_proj(last)          # (batch, 1)
        return out.squeeze(-1)

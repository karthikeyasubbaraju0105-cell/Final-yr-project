import torch
import numpy as np
from transformer_model import EdgeTransformer

class TransformerPredictor:
    def __init__(self, model_path="edge_predictor.pt", seq_len=20):
        self.seq_len = seq_len
        self.model = EdgeTransformer(seq_len=seq_len)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        print("Loaded Transformer model for prediction.")

    def predict_next_cpu(self, edge):
        """
        Uses last seq_len cpu usage points from edge.history.
        If history is too short, zero-pad.
        """
        hist = [h[1] for h in edge.history]

        if len(hist) == 0:
            return 0.0

        seq = hist[-self.seq_len:]
        if len(seq) < self.seq_len:
            seq = [0.0] * (self.seq_len - len(seq)) + seq

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = self.model(seq_tensor).item()
        return max(0.0, pred)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformer_model import EdgeTransformer

# Hyperparameters
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

class CPUDataset(Dataset):
    def __init__(self, cpu_history, seq_len):
        self.seq_len = seq_len
        self.data = []
        for hist in cpu_history:
            if len(hist) <= seq_len:
                continue
            for i in range(len(hist) - seq_len):
                seq = hist[i:i+seq_len]
                nxt = hist[i+seq_len]
                self.data.append((seq, nxt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, nxt = self.data[idx]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(-1)  # shape (seq_len, 1)
        nxt_tensor = torch.tensor(nxt, dtype=torch.float32)
        return seq_tensor, nxt_tensor

def load_cpu_history():
    print("Loading real CPU history from results/edge_history.csv ...")
    df = pd.read_csv("results/edge_history.csv")

    # CPU history grouped per edge_id
    cpu_history = {}
    grouped = df.groupby("edge_id")

    for eid, g in grouped:
        # Sort by time to ensure proper sequence
        seq = list(g.sort_values("t")["cpu"].values)
        cpu_history[eid] = seq

    print(f"Loaded CPU history for {len(cpu_history)} edges.")
    return cpu_history


def main():
    cpu_history = load_cpu_history()
    dataset = CPUDataset(list(cpu_history.values()), SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EdgeTransformer(seq_len=SEQ_LEN)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"Training Transformer on {len(dataset)} samples...")

    for epoch in range(EPOCHS):
        total_loss = 0
        for seq, nxt in loader:
            pred = model(seq)
            loss = loss_fn(pred, nxt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "edge_predictor.pt")
    print("Saved model to edge_predictor.pt")


if __name__ == "__main__":
    main()

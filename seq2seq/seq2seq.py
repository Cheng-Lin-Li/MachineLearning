"""
Sequence-to-sequence (encoder-decoder GRU) model for multi-step time series
prediction, written in PyTorch.

Dataset format (CSV header):  timestamp, input1, input2, output1

Task
----
Given the past `IN_LEN` steps of [input1, input2, output1], predict the next
`OUT_LEN` steps of output1.

Design choices (good talking points in an interview)
----------------------------------------------------
* Chronological train/val/test split: random splitting leaks the future.
* Scaler statistics are fit on the training rows only (no leakage).
* Encoder GRU compresses the history into a hidden state.
* Decoder GRU generates the forecast one step at a time, starting from the
  last observed output1. Training uses teacher forcing (with a probability),
  inference feeds back its own predictions.
* A "repeat last value" naive baseline is reported; a model that cannot beat
  it is not learning anything useful.

Usage
-----
    python seq2seq_timeseries.py --csv data.csv
    python seq2seq_timeseries.py            # runs on synthetic data
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FEATURES = ["input1", "input2", "output1"]  # encoder input columns
TARGET = "output1"                          # column to forecast


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_synthetic_csv(path: str, n: int = 3000) -> None:
    """Write a toy CSV in the expected format (output1 depends on the inputs)."""
    t = np.arange(n)
    in1 = np.sin(2 * np.pi * t / 50) + 0.1 * np.random.randn(n)
    in2 = np.cos(2 * np.pi * t / 80) + 0.1 * np.random.randn(n)
    out = 0.6 * in1 + 0.4 * in2 + 0.05 * np.random.randn(n)
    out = pd.Series(out).rolling(5, min_periods=1).mean().values  # smooth it
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "input1": in1, "input2": in2, "output1": out,
    })
    df.to_csv(path, index=False)


class WindowDataset(Dataset):
    """Turns a (T, 3) scaled array into sliding (history, future) windows.

    Each item returns:
        x : (in_len, n_features)  encoder input
        y : (out_len,)            future target values (scaled)
    """

    def __init__(self, data: np.ndarray, target_idx: int, in_len: int, out_len: int):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target_idx, self.in_len, self.out_len = target_idx, in_len, out_len

    def __len__(self) -> int:
        return len(self.data) - self.in_len - self.out_len + 1

    def __getitem__(self, i: int):
        x = self.data[i : i + self.in_len]                                   # history
        y = self.data[i + self.in_len : i + self.in_len + self.out_len,
                      self.target_idx]                                       # future
        return x, y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Seq2Seq(nn.Module):
    """GRU encoder-decoder for multi-step forecasting of a single target."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, layers, batch_first=True)
        self.decoder = nn.GRU(1, hidden, layers, batch_first=True)  # input: prev target
        self.head = nn.Linear(hidden, 1)                            # hidden -> next value

    def forward(self, x, out_len, y_true=None, teacher_forcing=0.0):
        """
        Args:
            x:        (B, in_len, n_features) encoder input.
            out_len:  number of future steps to produce.
            y_true:   (B, out_len) ground truth, only used for teacher forcing.
            teacher_forcing: probability of feeding the true previous value
                             (instead of the model's own prediction) per step.
        Returns:
            (B, out_len) predictions.
        """
        _, h = self.encoder(x)                  # h: (layers, B, hidden) = history summary
        prev = x[:, -1, -1:].unsqueeze(1)       # last observed target, (B, 1, 1)
        preds = []
        for t in range(out_len):
            out, h = self.decoder(prev, h)      # one decoding step
            step = self.head(out)               # (B, 1, 1)
            preds.append(step.squeeze(-1))      # (B, 1)
            if y_true is not None and random.random() < teacher_forcing:
                prev = y_true[:, t].view(-1, 1, 1)  # feed ground truth
            else:
                prev = step                         # feed own prediction
        return torch.cat(preds, dim=1)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, out_len, std, device):
    """Return (model_rmse, baseline_rmse, model_mae) in ORIGINAL units."""
    model.eval()
    se = ae = se_base = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, out_len)                         # no teacher forcing at eval
        base = x[:, -1, -1:].expand_as(y)                # naive: repeat last value
        se += ((pred - y) ** 2).sum().item()
        ae += (pred - y).abs().sum().item()
        se_base += ((base - y) ** 2).sum().item()
        n += y.numel()
    # errors were computed on scaled values; multiply by std to un-scale
    return (se / n) ** 0.5 * std, (se_base / n) ** 0.5 * std, ae / n * std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="path to CSV; omit for synthetic data")
    p.add_argument("--in_len", type=int, default=48)
    p.add_argument("--out_len", type=int, default=12)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tf", type=float, default=0.5, help="teacher forcing probability")
    args = p.parse_args()

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.csv is None:
        args.csv = "synthetic.csv"
        make_synthetic_csv(args.csv)
    df = pd.read_csv(args.csv, parse_dates=["timestamp"]).sort_values("timestamp")
    values = df[FEATURES].values.astype(np.float32)

    # Chronological 70/15/15 split; scale using TRAIN statistics only.
    n = len(values)
    i_tr, i_va = int(n * 0.7), int(n * 0.85)
    mean, std = values[:i_tr].mean(0), values[:i_tr].std(0) + 1e-8
    scaled = (values - mean) / std
    t_idx = FEATURES.index(TARGET)

    def loader(part, shuffle):
        ds = WindowDataset(part, t_idx, args.in_len, args.out_len)
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    train_dl = loader(scaled[:i_tr], True)
    val_dl = loader(scaled[i_tr:i_va], False)
    test_dl = loader(scaled[i_va:], False)

    model = Seq2Seq(len(FEATURES), args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    tgt_std = std[t_idx]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x, args.out_len, y, args.tf), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilise RNN training
            opt.step()
            total += loss.item() * len(x)
        val_rmse, val_base, _ = evaluate(model, val_dl, args.out_len, tgt_std, device)
        print(f"epoch {epoch:2d} | train mse(scaled) {total / len(train_dl.dataset):.4f} "
              f"| val RMSE {val_rmse:.4f} (naive {val_base:.4f})")

    rmse, base, mae = evaluate(model, test_dl, args.out_len, tgt_std, device)
    print(f"\nTEST  RMSE {rmse:.4f} | MAE {mae:.4f} | naive-baseline RMSE {base:.4f}")


if __name__ == "__main__":
    main()

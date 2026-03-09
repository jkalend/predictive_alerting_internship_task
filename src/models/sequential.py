"""LSTM and TCN models for sequential (raw-window) incident prediction.

These models consume raw W-step sequences (shape: batch × W × n_features)
instead of pre-computed rolling statistics. Requires PyTorch.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    raise ImportError("LSTM and TCN require PyTorch. Install with: pip install torch")


class LSTMClassifier(nn.Module):
    """Binary classifier on raw time-series windows using LSTM."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # last layer hidden
        return out.squeeze(-1)


class TCNBlock(nn.Module):
    """Single TCN residual block with dilated causal convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        y = torch.relu(self.conv1(x))
        y = y[:, :, : x.size(2)]  # causal: trim future
        y = torch.relu(self.conv2(y))
        y = y[:, :, : x.size(2)]
        return torch.relu(y + self.downsample(x))


class TCNClassifier(nn.Module):
    """Binary classifier on raw time-series windows using Temporal Convolutional Network."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        n_channels: int = 32,
        kernel_size: int = 3,
        n_levels: int = 4,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i in range(n_levels):
            out_ch = n_channels * (2**i)
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i)
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features) -> transpose to (batch, n_features, seq_len)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]  # last timestep
        return self.fc(x).squeeze(-1)


class SequentialWrapper:
    """sklearn-like wrapper for LSTM/TCN: fit(), predict_proba()."""

    def __init__(
        self,
        model_type: str,
        n_features: int,
        seq_len: int,
        scale_pos_weight: float = 1.0,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str | None = None,
        **model_kwargs,
    ):
        self.model_type = model_type
        self.n_features = n_features
        self.seq_len = seq_len
        self.scale_pos_weight = scale_pos_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = model_kwargs
        self.model_: nn.Module | None = None

    def _build_model(self) -> nn.Module:
        if self.model_type == "lstm":
            return LSTMClassifier(
                self.n_features, self.seq_len, **self.model_kwargs
            ).to(self.device)
        elif self.model_type == "tcn":
            return TCNClassifier(
                self.n_features, self.seq_len, **self.model_kwargs
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> SequentialWrapper:
        """Train on raw windows X of shape (n_samples, seq_len, n_features)."""
        X_t = torch.from_numpy(np.asarray(X).copy()).float()
        y_t = torch.from_numpy(np.asarray(y).copy()).float().unsqueeze(1)

        pos_weight = torch.tensor(
            [self.scale_pos_weight], dtype=torch.float32, device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model_ = self._build_model()
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb.squeeze(-1))
                loss.backward()
                opt.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities for class 1, shape (n_samples,)."""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")
        self.model_.eval()
        X_t = torch.from_numpy(X).float()
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        probs_list = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb)
                probs_list.append(torch.sigmoid(logits).cpu())
        return torch.cat(probs_list, dim=0).numpy()

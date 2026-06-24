#!/usr/bin/env python3
"""Deep classifiers (PyTorch) wrapped as sklearn estimators.

- ``TorchMLPClassifier`` (FNN): dense net over pooled embeddings (+tabular)
- ``TorchCNN1DClassifier``: 1D conv net over flattened one-hot / BLOSUM (L*C)

Both standardize X internally, mini-batch with Adam, early-stop on an internal
validation split, expose ``predict_proba`` / ``classes_`` (= 0..K-1), and implement
``get_params``/``set_params`` so ``sklearn.clone`` works inside CV.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    nn = None
    HAS_TORCH = False


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    return device


def _parse_hidden(hidden_sizes) -> Tuple[int, ...]:
    if isinstance(hidden_sizes, str):
        return tuple(int(p) for p in hidden_sizes.split("_") if p)
    return tuple(int(h) for h in hidden_sizes)


class _BaseTorchClassifier(ClassifierMixin, BaseEstimator):
    def _build_net(self, n_features: int, n_classes: int) -> "nn.Module":  # pragma: no cover
        raise NotImplementedError

    def _train(self, X: np.ndarray, y: np.ndarray):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for deep models. Install torch.")
        rng = np.random.RandomState(self.random_state)
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).ravel().astype(np.int64)

        self.classes_ = np.arange(int(self.n_classes))
        self._x_mean = X.mean(axis=0, keepdims=True)
        self._x_std = X.std(axis=0, keepdims=True) + 1e-8
        Xs = ((X - self._x_mean) / self._x_std).astype(np.float32)

        n = len(Xs)
        n_val = int(round(n * self.val_fraction))
        n_val = min(max(n_val, 0), max(0, n - 1))
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:] if n_val > 0 else perm

        device = _resolve_device(self.device)
        self._device_resolved = device
        net = self._build_net(Xs.shape[1], int(self.n_classes)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(Xs).to(device)
        yt = torch.from_numpy(y).to(device)

        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        best_val = float("inf")
        bad = 0
        batch_size = int(self.batch_size)
        max_epochs = int(self.max_epochs)
        if n > 50_000:
            max_epochs = min(max_epochs, 60)
            batch_size = max(batch_size, 1024)
        elif n > 10_000:
            max_epochs = min(max_epochs, 120)
            batch_size = max(batch_size, 256)

        for _epoch in range(max_epochs):
            net.train()
            order = rng.permutation(len(train_idx))
            tr = train_idx[order]
            for start in range(0, len(tr), batch_size):
                b = tr[start : start + batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                eval_idx = val_idx if n_val > 0 else train_idx
                vloss = float(loss_fn(net(Xt[eval_idx]), yt[eval_idx]).item())
            if vloss < best_val - 1e-6:
                best_val = vloss
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.n_features_in_ = Xs.shape[1]
        return self

    def fit(self, X, y):
        return self._train(X, y)

    def predict_proba(self, X):
        if not hasattr(self, "net_"):
            raise RuntimeError("Estimator not fitted")
        X = np.asarray(X, dtype=np.float32)
        Xs = ((X - self._x_mean) / self._x_std).astype(np.float32)
        device = self._device_resolved
        out = []
        with torch.no_grad():
            for start in range(0, len(Xs), 4096):
                chunk = torch.from_numpy(Xs[start : start + 4096]).to(device)
                logits = self.net_(chunk)
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, int(self.n_classes)), dtype=float)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TorchMLPClassifier(_BaseTorchClassifier):
    def __init__(
        self,
        n_classes: int = 2,
        hidden_sizes=(256, 128),
        dropout: float = 0.1,
        batchnorm: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 300,
        batch_size: int = 64,
        patience: int = 20,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.n_classes = n_classes
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _build_net(self, n_features: int, n_classes: int) -> "nn.Module":
        layers = []
        prev = n_features
        for h in _parse_hidden(self.hidden_sizes):
            layers.append(nn.Linear(prev, h))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        return nn.Sequential(*layers)


class TorchCNN1DClassifier(_BaseTorchClassifier):
    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 20,
        n_filters: int = 64,
        kernel_size: int = 5,
        n_conv_layers: int = 2,
        fc_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 300,
        batch_size: int = 64,
        patience: int = 20,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: str = "auto",
    ):
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_conv_layers = n_conv_layers
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _build_net(self, n_features: int, n_classes: int) -> "nn.Module":
        if n_features % self.n_channels != 0:
            raise ValueError(f"CNN expects features divisible by n_channels={self.n_channels}; got {n_features}")
        seq_len = n_features // self.n_channels
        channels = self.n_channels
        n_filters = self.n_filters
        kernel_size = self.kernel_size
        n_conv_layers = self.n_conv_layers
        fc_dim = self.fc_dim
        dropout = self.dropout

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                convs = []
                in_ch = channels
                for _ in range(n_conv_layers):
                    convs.append(nn.Conv1d(in_ch, n_filters, kernel_size, padding=kernel_size // 2))
                    convs.append(nn.ReLU())
                    in_ch = n_filters
                self.convs = nn.Sequential(*convs)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Sequential(
                    nn.Linear(n_filters, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, n_classes)
                )

            def forward(self, x):
                b = x.shape[0]
                x = x.view(b, seq_len, channels).transpose(1, 2)
                x = self.convs(x)
                x = self.pool(x).squeeze(-1)
                return self.head(x)

        return _CNN()


def build_torch_model(
    name: str,
    trial: "object",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
    n_classes: int,
):
    batch_size = trial.suggest_categorical(f"{name}_batch_size", [32, 64, 128, 256])
    batch_size = int(min(batch_size, max(8, n_samples_train // 2)))
    lr = trial.suggest_float(f"{name}_lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float(f"{name}_wd", 1e-7, 1e-3, log=True)
    dropout = trial.suggest_float(f"{name}_dropout", 0.0, 0.4)

    if name == "mlp_torch":
        hidden = trial.suggest_categorical("mlp_torch_hidden", ["128", "256", "256_128", "512_256", "512_256_128"])
        model = TorchMLPClassifier(
            n_classes=n_classes, hidden_sizes=hidden, dropout=dropout,
            batchnorm=trial.suggest_categorical("mlp_torch_bn", [True, False]),
            lr=lr, weight_decay=weight_decay, batch_size=batch_size, max_epochs=300, patience=20,
            random_state=random_state, device="auto",
        )
        return name, model
    if name == "cnn1d":
        model = TorchCNN1DClassifier(
            n_classes=n_classes, n_channels=20,
            n_filters=trial.suggest_categorical("cnn1d_filters", [32, 64, 128]),
            kernel_size=trial.suggest_categorical("cnn1d_kernel", [3, 5, 7]),
            n_conv_layers=trial.suggest_int("cnn1d_layers", 1, 3),
            fc_dim=trial.suggest_categorical("cnn1d_fc", [64, 128, 256]),
            dropout=dropout, lr=lr, weight_decay=weight_decay, batch_size=batch_size, max_epochs=300, patience=20,
            random_state=random_state, device="auto",
        )
        return name, model
    raise ValueError(f"Unknown torch classifier '{name}'")

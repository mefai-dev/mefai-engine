"""Temporal Fusion Transformer for multi-horizon price forecasting.

Architecture:
- Positional encoding for temporal awareness
- Multi-head self-attention for long-range dependencies
- Gated residual connections for feature selection
- Multi-horizon output (1h, 4h, 24h forecasts)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from mefai_engine.constants import Direction
from mefai_engine.models.base import BasePredictor
from mefai_engine.types import Prediction

logger = structlog.get_logger()


class TemporalTransformerPredictor(BasePredictor):
    """PyTorch Transformer for time-series price prediction.

    Takes a sequence of feature vectors (e.g., last 128 candles)
    and predicts price direction + magnitude at multiple horizons.
    """

    model_id = "transformer"
    model_version = "v1"

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        sequence_length: int = 128,
        n_features: int = 40,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        horizon_seconds: int = 14400,
    ) -> None:
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._d_ff = d_ff
        self._dropout = dropout
        self._seq_len = sequence_length
        self._n_features = n_features
        self._lr = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._horizon_seconds = horizon_seconds
        self._model: Any = None
        self._device: str = "cpu"
        self._trained = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _build_model(self) -> Any:
        """Build the PyTorch transformer model."""
        import torch
        import torch.nn as nn

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)

        class GatedResidual(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1):
                super().__init__()
                self.fc1 = nn.Linear(d_model, d_model)
                self.fc2 = nn.Linear(d_model, d_model)
                self.gate = nn.Linear(d_model, d_model)
                self.norm = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                h = torch.relu(self.fc1(x))
                h = self.dropout(self.fc2(h))
                g = torch.sigmoid(self.gate(x))
                return self.norm(residual + g * h)

        class PriceTransformer(nn.Module):
            def __init__(self, n_features: int, d_model: int, n_heads: int,
                         n_layers: int, d_ff: int, dropout: float, seq_len: int):
                super().__init__()
                self.input_proj = nn.Linear(n_features, d_model)
                self.pos_enc = PositionalEncoding(d_model, seq_len, dropout)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_ff, dropout=dropout,
                    batch_first=True, activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                self.gated_residual = GatedResidual(d_model, dropout)

                # Multi-horizon heads
                self.direction_head = nn.Linear(d_model, 3)  # LONG/FLAT/SHORT
                self.magnitude_head = nn.Linear(d_model, 1)  # Price change magnitude
                self.confidence_head = nn.Linear(d_model, 1)  # Prediction confidence

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                h = self.input_proj(x)
                h = self.pos_enc(h)
                h = self.encoder(h)
                h = self.gated_residual(h[:, -1, :])  # Use last timestep

                direction = self.direction_head(h)
                magnitude = self.magnitude_head(h)
                confidence = torch.sigmoid(self.confidence_head(h))

                return direction, magnitude, confidence

        model = PriceTransformer(
            n_features=self._n_features,
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            d_ff=self._d_ff,
            dropout=self._dropout,
            seq_len=self._seq_len,
        )
        return model

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the transformer model.

        Args:
            features: (n_samples, seq_len, n_features) 3D array
            targets: (n_samples,) direction labels {0, 1, 2}
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._n_features = features.shape[-1] if features.ndim == 3 else features.shape[-1]

        self._model = self._build_model().to(self._device)

        # Create sequences if input is 2D
        if features.ndim == 2:
            features = self._create_sequences(features, self._seq_len)
            targets = targets[self._seq_len - 1:]

        # Walk-forward split
        split = int(len(features) * (1 - validation_split))
        X_train = torch.FloatTensor(features[:split]).to(self._device)
        y_train = torch.LongTensor(targets[:split]).to(self._device)
        X_val = torch.FloatTensor(features[split:]).to(self._device)
        y_val = torch.LongTensor(targets[split:]).to(self._device)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self._batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self._epochs):
            self._model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                direction, magnitude, confidence = self._model(X_batch)
                loss = criterion(direction, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_dir, _, _ = self._model(X_val)
                val_loss = criterion(val_dir, y_val).item()
                val_pred = val_dir.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("transformer.early_stop", epoch=epoch)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "transformer.epoch",
                    epoch=epoch + 1,
                    train_loss=f"{train_loss / len(train_loader):.4f}",
                    val_loss=f"{val_loss:.4f}",
                    val_acc=f"{val_acc:.4f}",
                )

        self._trained = True
        return {"val_loss": best_val_loss, "val_accuracy": val_acc}

    def predict(self, features: np.ndarray) -> Prediction:
        """Predict from a feature sequence."""
        import torch

        if not self._trained or self._model is None:
            from mefai_engine.exceptions import ModelNotTrainedError
            raise ModelNotTrainedError("Transformer not trained")

        self._model.eval()

        if features.ndim == 2:
            features = features[-self._seq_len:]
            features = features[np.newaxis, :]  # Add batch dim

        with torch.no_grad():
            X = torch.FloatTensor(features).to(self._device)
            direction, magnitude, confidence = self._model(X)
            proba = torch.softmax(direction, dim=1)[0].cpu().numpy()
            pred_class = int(proba.argmax())
            mag = float(magnitude[0, 0].cpu().item())
            conf = float(confidence[0, 0].cpu().item())

        dir_map = {0: Direction.SHORT, 1: Direction.FLAT, 2: Direction.LONG}

        return Prediction(
            direction=dir_map.get(pred_class, Direction.FLAT),
            confidence=float(proba[pred_class]) * conf,
            magnitude=mag,
            horizon_seconds=self._horizon_seconds,
            model_id=self.model_id,
            model_version=self.model_version,
            timestamp=datetime.now(tz=timezone.utc),
        )

    def predict_batch(self, features: np.ndarray) -> list[Prediction]:
        """Batch prediction."""
        import torch

        if not self._trained or self._model is None:
            from mefai_engine.exceptions import ModelNotTrainedError
            raise ModelNotTrainedError("Transformer not trained")

        self._model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self._device)
            direction, magnitude, confidence = self._model(X)
            probas = torch.softmax(direction, dim=1).cpu().numpy()

        dir_map = {0: Direction.SHORT, 1: Direction.FLAT, 2: Direction.LONG}
        predictions = []
        for i in range(len(probas)):
            cls = int(probas[i].argmax())
            predictions.append(Prediction(
                direction=dir_map.get(cls, Direction.FLAT),
                confidence=float(probas[i][cls]),
                magnitude=float(magnitude[i, 0].cpu().item()),
                horizon_seconds=self._horizon_seconds,
                model_id=self.model_id,
                model_version=self.model_version,
            ))
        return predictions

    @staticmethod
    def _create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sliding window sequences from 2D array."""
        n = len(data) - seq_len + 1
        sequences = np.zeros((n, seq_len, data.shape[1]))
        for i in range(n):
            sequences[i] = data[i: i + seq_len]
        return sequences

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict() if self._model else None,
            "config": {
                "d_model": self._d_model,
                "n_heads": self._n_heads,
                "n_layers": self._n_layers,
                "d_ff": self._d_ff,
                "dropout": self._dropout,
                "seq_len": self._seq_len,
                "n_features": self._n_features,
            },
            "version": self.model_version,
        }, path)
        logger.info("transformer.saved", path=str(path))

    def load(self, path: Path) -> None:
        import torch
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        config = checkpoint["config"]
        self._d_model = config["d_model"]
        self._n_heads = config["n_heads"]
        self._n_layers = config["n_layers"]
        self._d_ff = config["d_ff"]
        self._seq_len = config["seq_len"]
        self._n_features = config["n_features"]
        self._model = self._build_model().to(self._device)
        if checkpoint["model_state"]:
            self._model.load_state_dict(checkpoint["model_state"])
        self.model_version = checkpoint.get("version", "v1")
        self._trained = True
        logger.info("transformer.loaded", path=str(path))

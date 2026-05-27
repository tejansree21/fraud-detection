"""
models/autoencoder.py
PyTorch Autoencoder for fraud anomaly detection.

Core idea: train ONLY on legitimate transactions. The model learns
to reconstruct normal patterns. Fraudulent transactions produce high
reconstruction error — that error becomes the fraud score.

Architecture:
  Input(15) → 32 → 16 → 8 → 16 → 32 → Output(15)
  Bottleneck of 8 forces learning of compact normal representations.
"""

from __future__ import annotations
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

logger = logging.getLogger("autoencoder")


# ── Model architecture ─────────────────────────────────────────────────────────

class FraudAutoencoder(nn.Module):
    """
    Symmetric autoencoder with BatchNorm and dropout for regularisation.
    Trained exclusively on legitimate transactions.
    """

    def __init__(self, input_dim: int = 15, bottleneck_dim: int = 8):
        super().__init__()
        self.input_dim      = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, bottleneck_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)


# ── Trainer ────────────────────────────────────────────────────────────────────

class AutoencoderTrainer:
    """
    Trains the autoencoder on legitimate transactions only.

    Key design choices:
    - Loss: MSE on reconstruction
    - Train on class=0 only — fraud cases are unseen during training
    - Threshold: 95th percentile of reconstruction errors on validation set
    """

    def __init__(
        self,
        input_dim:      int   = 15,
        bottleneck_dim: int   = 8,
        lr:             float = 1e-3,
        batch_size:     int   = 256,
        epochs:         int   = 30,
        device:         str   = "auto",
    ):
        self.batch_size = batch_size
        self.epochs     = epochs

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = FraudAutoencoder(input_dim, bottleneck_dim).to(self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, patience=3, factor=0.5, verbose=True
        )
        self.criterion  = nn.MSELoss()
        self.threshold  = None   # set after training
        self.train_losses: list[float] = []

        logger.info("AutoencoderTrainer | device=%s | epochs=%d", self.device, epochs)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
    ) -> dict:
        """
        Train on legitimate transactions only.
        Returns dict of training metrics for MLflow logging.
        """
        # Filter: train ONLY on legitimate transactions
        legit_mask = y_train == 0
        X_legit    = X_train[legit_mask]
        logger.info(
            "Training on %d legitimate transactions (excluded %d fraud)",
            len(X_legit), legit_mask.sum() == False
        )

        dataset    = TensorDataset(torch.FloatTensor(X_legit))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                self.optimiser.zero_grad()
                recon = self.model(batch)
                loss  = self.criterion(recon, batch)
                loss.backward()
                self.optimiser.step()
                epoch_loss += loss.item() * len(batch)

            epoch_loss /= len(X_legit)
            self.train_losses.append(epoch_loss)

            # Validation loss (on legitimate val transactions)
            val_legit_mask = y_val == 0
            val_loss = self._eval_loss(X_val[val_legit_mask])
            self.scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == self.epochs - 1:
                logger.info(
                    "Epoch %3d/%d | train_loss=%.6f | val_loss=%.6f",
                    epoch + 1, self.epochs, epoch_loss, val_loss
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_count = 0
                self._save_best()
            else:
                patience_count += 1
                if patience_count >= 7:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best weights
        self._load_best()

        # Set threshold at 95th percentile of legitimate val errors
        self.threshold = self._compute_threshold(X_val, y_val, percentile=95)
        logger.info("Reconstruction error threshold: %.6f", self.threshold)

        return {
            "final_train_loss": self.train_losses[-1],
            "best_val_loss":    best_val_loss,
            "threshold":        self.threshold,
            "epochs_trained":   len(self.train_losses),
        }

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return fraud scores in [0, 1] for each sample.
        Scores are normalised reconstruction errors.
        """
        self.model.eval()
        tensor = torch.FloatTensor(X).to(self.device)
        errors = self.model.reconstruction_error(tensor).cpu().numpy()

        # Normalise to [0, 1] using threshold as reference point
        if self.threshold is not None and self.threshold > 0:
            scores = errors / (self.threshold * 2)
            scores = np.clip(scores, 0, 1)
        else:
            # Fallback: min-max normalisation
            scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return scores.astype(np.float32)

    def save(self, path: Path) -> None:
        torch.save({
            "model_state":  self.model.state_dict(),
            "threshold":    self.threshold,
            "input_dim":    self.model.input_dim,
            "bottleneck":   self.model.bottleneck_dim,
            "train_losses": self.train_losses,
        }, path)
        logger.info("Autoencoder saved to %s", path)

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "AutoencoderTrainer":
        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        checkpoint = torch.load(path, map_location=device_obj)
        trainer    = cls(
            input_dim      = checkpoint["input_dim"],
            bottleneck_dim = checkpoint["bottleneck"],
            device         = str(device_obj),
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.model.eval()
        trainer.threshold    = checkpoint["threshold"]
        trainer.train_losses = checkpoint.get("train_losses", [])
        logger.info("Autoencoder loaded from %s | threshold=%.6f", path, trainer.threshold)
        return trainer

    # ── Private helpers ────────────────────────────────────────────────────────

    def _eval_loss(self, X: np.ndarray) -> float:
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            recon  = self.model(tensor)
            loss   = self.criterion(recon, tensor).item()
        self.model.train()
        return loss

    def _compute_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray, percentile: int = 95
    ) -> float:
        """95th percentile of reconstruction errors on legitimate val samples."""
        legit_X = X_val[y_val == 0]
        self.model.eval()
        tensor = torch.FloatTensor(legit_X).to(self.device)
        errors = self.model.reconstruction_error(tensor).cpu().numpy()
        return float(np.percentile(errors, percentile))

    def _save_best(self):
        self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def _load_best(self):
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

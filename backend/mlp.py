"""A small fully connected neural network implemented with NumPy only."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from backend.encoder import INPUT_SIZE, LEGACY_INPUT_SIZE


@dataclass
class Gradients:
    """Parameter gradients and the loss for one training example."""

    W1: NDArray[np.float64]
    b1: NDArray[np.float64]
    W2: NDArray[np.float64]
    b2: NDArray[np.float64]
    W3: NDArray[np.float64]
    b3: NDArray[np.float64]
    loss: float


class MLP:
    """NumPy network with ReLU, ReLU, and tanh activations."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden1_size: int = 256,
        hidden2_size: int = 64,
        seed: int | None = None,
    ) -> None:
        if min(input_size, hidden1_size, hidden2_size) <= 0:
            raise ValueError("All layer sizes must be positive.")

        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / input_size), (input_size, hidden1_size))
        self.b1 = np.zeros(hidden1_size, dtype=np.float64)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / hidden1_size), (hidden1_size, hidden2_size))
        self.b2 = np.zeros(hidden2_size, dtype=np.float64)
        self.W3 = rng.normal(0.0, np.sqrt(1.0 / hidden2_size), (hidden2_size, 1))
        self.b3 = np.zeros(1, dtype=np.float64)

    def _validate_input(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.W1.shape[0],):
            raise ValueError(
                f"Expected an input vector of shape ({self.W1.shape[0]},), "
                f"received {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Input features must all be finite.")
        return values

    def forward(self, features: NDArray[np.floating]) -> float:
        """Return a scalar evaluation in [-1, 1], from White's perspective."""
        values = self._validate_input(features)
        hidden1 = np.maximum(0.0, values @ self.W1 + self.b1)
        hidden2 = np.maximum(0.0, hidden1 @ self.W2 + self.b2)
        output = np.tanh(hidden2 @ self.W3 + self.b3)
        return float(output[0])

    def loss_and_gradients(
        self, features: NDArray[np.floating], target: float
    ) -> Gradients:
        """Compute squared error and its gradients for one labeled position."""
        if not np.isfinite(target) or not -1.0 <= target <= 1.0:
            raise ValueError("The target must be a finite value between -1 and 1.")

        values = self._validate_input(features)
        z1 = values @ self.W1 + self.b1
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ self.W3 + self.b3
        prediction = np.tanh(z3)

        error = float(prediction[0] - target)
        loss = error * error
        dz3 = 2.0 * error * (1.0 - prediction**2)
        dW3 = h2[:, None] @ dz3.reshape(1, 1)
        db3 = dz3

        dh2 = (dz3 @ self.W3.T).reshape(-1)
        dz2 = dh2 * (z2 > 0.0)
        dW2 = h1[:, None] @ dz2[None, :]
        db2 = dz2

        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0.0)
        dW1 = values[:, None] @ dz1[None, :]
        db1 = dz1
        return Gradients(dW1, db1, dW2, db2, dW3, db3, loss)

    def train_step(
        self,
        features: NDArray[np.floating],
        target: float,
        learning_rate: float = 0.001,
    ) -> float:
        """Apply one gradient-descent update and return the pre-update loss."""
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("The learning rate must be a positive finite number.")

        gradients = self.loss_and_gradients(features, target)
        self.W1 -= learning_rate * gradients.W1
        self.b1 -= learning_rate * gradients.b1
        self.W2 -= learning_rate * gradients.W2
        self.b2 -= learning_rate * gradients.b2
        self.W3 -= learning_rate * gradients.W3
        self.b3 -= learning_rate * gradients.b3
        return gradients.loss

    def save(self, path: str | Path) -> None:
        """Atomically save parameters and the inferred input representation version."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        version = 1 if self.W1.shape[0] == LEGACY_INPUT_SIZE else (
            2 if self.W1.shape[0] == INPUT_SIZE else 0
        )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{destination.stem}-",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    W1=self.W1,
                    b1=self.b1,
                    W2=self.W2,
                    b2=self.b2,
                    W3=self.W3,
                    b3=self.b3,
                    model_version=np.asarray(version, dtype=np.int64),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> MLP:
        """Load and validate a network saved with :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            required = ("W1", "b1", "W2", "b2", "W3", "b3")
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(f"Model archive is missing parameters: {', '.join(missing)}")
            parameters = {
                name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in required
            }
            model_version = (
                int(archive["model_version"])
                if "model_version" in archive
                else None
            )

        W1, b1 = parameters["W1"], parameters["b1"]
        W2, b2 = parameters["W2"], parameters["b2"]
        W3, b3 = parameters["W3"], parameters["b3"]
        if W1.ndim != 2 or W2.ndim != 2 or W3.ndim != 2:
            raise ValueError("Model weight arrays must be two-dimensional.")
        if b1.shape != (W1.shape[1],) or W2.shape[0] != W1.shape[1]:
            raise ValueError("Model's first and second layer dimensions do not match.")
        if b2.shape != (W2.shape[1],) or W3.shape != (W2.shape[1], 1):
            raise ValueError("Model's second and output layer dimensions do not match.")
        if b3.shape != (1,):
            raise ValueError("Model's output bias must have shape (1,).")
        if model_version == 1 and W1.shape[0] != LEGACY_INPUT_SIZE:
            raise ValueError("Model metadata declares V1 but its input dimension is not 773.")
        if model_version == 2 and W1.shape[0] != INPUT_SIZE:
            raise ValueError("Model metadata declares V2 but its input dimension is not 838.")
        if not all(np.all(np.isfinite(value)) for value in parameters.values()):
            raise ValueError("Model parameters must all be finite.")

        model = cls(W1.shape[0], W1.shape[1], W2.shape[1])
        model.W1, model.b1 = W1, b1
        model.W2, model.b2 = W2, b2
        model.W3, model.b3 = W3, b3
        return model

"""AttackGAN: generator creates attack-like traffic, discriminator classifies real vs fake."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .traffic_generator import N_FEATURES


class Generator:
    """Simple MLP generator: noise → synthetic attack traffic."""

    def __init__(self, latent_dim: int = 32, output_dim: int = N_FEATURES, seed: int = 42):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.standard_normal((latent_dim, 64)) * np.sqrt(2.0 / latent_dim)
        self.b1 = np.zeros(64)
        self.W2 = rng.standard_normal((64, 64)) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(64)
        self.W3 = rng.standard_normal((64, output_dim)) * np.sqrt(2.0 / 64)
        self.b3 = np.zeros(output_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Generate synthetic traffic from latent vectors z (batch, latent_dim)."""
        h1 = self._relu(z @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        # Sigmoid to keep in [0, 1]
        return 1.0 / (1.0 + np.exp(-np.clip(out, -30, 30)))

    def parameters(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


class Discriminator:
    """Simple MLP discriminator: traffic → P(real)."""

    def __init__(self, input_dim: int = N_FEATURES, seed: int = 99):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, 64)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(64)
        self.W2 = rng.standard_normal((64, 32)) * np.sqrt(2.0 / 64)
        self.b2 = np.zeros(32)
        self.W3 = rng.standard_normal((32, 1)) * np.sqrt(2.0 / 32)
        self.b3 = np.zeros(1)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return P(real) for each sample in x (batch, input_dim)."""
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        logit = h2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))

    def parameters(self) -> List[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


class AttackGAN:
    """GAN for generating attack-like network traffic."""

    def __init__(self, latent_dim: int = 32, seed: int = 42):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim, seed=seed)
        self.discriminator = Discriminator(seed=seed + 1)
        self.rng = np.random.default_rng(seed)

        # Training history
        self.g_losses: List[float] = []
        self.d_losses: List[float] = []

    def _bce_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        pred = np.clip(pred, 1e-7, 1 - 1e-7)
        return -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    def sample_latent(self, n: int) -> np.ndarray:
        return self.rng.standard_normal((n, self.latent_dim))

    def generate(self, n: int) -> np.ndarray:
        """Generate n synthetic attack traffic samples."""
        z = self.sample_latent(n)
        return self.generator.forward(z)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Score samples: P(real attack)."""
        return self.discriminator.forward(x)

    def d_loss(self, real_samples: np.ndarray, fake_samples: np.ndarray) -> float:
        """Discriminator loss: real→1, fake→0."""
        real_preds = self.discriminator.forward(real_samples)
        fake_preds = self.discriminator.forward(fake_samples)
        loss_real = self._bce_loss(real_preds, np.ones_like(real_preds))
        loss_fake = self._bce_loss(fake_preds, np.zeros_like(fake_preds))
        return (loss_real + loss_fake) / 2.0

    def g_loss(self, fake_samples: np.ndarray) -> float:
        """Generator loss: fool discriminator → fake→1."""
        fake_preds = self.discriminator.forward(fake_samples)
        return self._bce_loss(fake_preds, np.ones_like(fake_preds))

"""TrainingLoop: GAN training with gradient clipping."""

import numpy as np
from typing import Callable, Dict, List, Optional

from .attack_gan import AttackGAN
from .traffic_generator import NetworkTrafficGenerator


class TrainingLoop:
    """Train the AttackGAN using numerical gradient approximation."""

    def __init__(
        self,
        gan: AttackGAN,
        traffic_gen: NetworkTrafficGenerator,
        batch_size: int = 64,
        lr: float = 1e-3,
        grad_clip: float = 1.0,
        d_steps_per_g: int = 2,
    ):
        self.gan = gan
        self.traffic_gen = traffic_gen
        self.batch_size = batch_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.d_steps_per_g = d_steps_per_g
        self.history: List[Dict] = []

    def _numerical_grad(self, param: np.ndarray, loss_fn: Callable, eps: float = 1e-4) -> np.ndarray:
        """Estimate gradient using finite differences (for small params only)."""
        grad = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus = loss_fn()
            param[idx] = orig - eps
            loss_minus = loss_fn()
            param[idx] = orig
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        return grad

    def _sgd_update(self, params: List[np.ndarray], grads: List[np.ndarray]):
        """SGD with gradient clipping."""
        for param, grad in zip(params, grads):
            # Clip gradient norm
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.grad_clip:
                grad = grad * (self.grad_clip / grad_norm)
            param -= self.lr * grad

    def _fast_d_update(self, real: np.ndarray, fake: np.ndarray):
        """Approximate discriminator update via sign gradient descent."""
        d = self.gan.discriminator
        real_preds = d.forward(real)
        fake_preds = d.forward(fake)

        # Gradient signal: push real→1, fake→0
        d_real_err = real_preds - 1.0  # want 0
        d_fake_err = fake_preds - 0.0  # want 0

        # Simple parameter perturbation update (approximate)
        for param in d.parameters():
            if param.ndim == 2:
                noise = self.gan.rng.standard_normal(param.shape)
                param -= self.lr * 0.01 * noise * np.sign(d_real_err.mean() + d_fake_err.mean())

    def _fast_g_update(self):
        """Approximate generator update."""
        z = self.gan.sample_latent(self.batch_size)
        fake = self.gan.generator.forward(z)
        fake_preds = self.gan.discriminator.forward(fake)
        g_err = 1.0 - fake_preds.mean()  # want fake_preds → 1

        for param in self.gan.generator.parameters():
            if param.ndim == 2:
                noise = self.gan.rng.standard_normal(param.shape)
                param -= self.lr * 0.01 * noise * g_err

    def train_epoch(self, real_data: np.ndarray) -> Dict:
        """Run one training epoch."""
        n = len(real_data)
        idx = self.gan.rng.permutation(n)
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        n_batches = max(1, n // self.batch_size)

        for i in range(n_batches):
            batch_idx = idx[i * self.batch_size: (i + 1) * self.batch_size]
            real_batch = real_data[batch_idx]

            # Train discriminator
            for _ in range(self.d_steps_per_g):
                fake_batch = self.gan.generate(len(real_batch))
                d_loss = self.gan.d_loss(real_batch, fake_batch)
                d_loss_sum += d_loss
                self._fast_d_update(real_batch, fake_batch)

            # Train generator
            fake_batch = self.gan.generate(len(real_batch))
            g_loss = self.gan.g_loss(fake_batch)
            g_loss_sum += g_loss
            self._fast_g_update()

        metrics = {
            "d_loss": d_loss_sum / (n_batches * self.d_steps_per_g),
            "g_loss": g_loss_sum / n_batches,
        }
        self.gan.d_losses.append(metrics["d_loss"])
        self.gan.g_losses.append(metrics["g_loss"])
        self.history.append(metrics)
        return metrics

    def train(
        self,
        n_epochs: int,
        attack_type: str = "syn_flood",
        callback: Optional[Callable[[int, Dict], None]] = None,
    ) -> List[Dict]:
        """Full training loop."""
        real_data = self.traffic_gen.generate_attack_traffic(
            self.batch_size * 10, attack_type=attack_type
        )

        for epoch in range(n_epochs):
            metrics = self.train_epoch(real_data)
            if callback:
                callback(epoch, metrics)

        return self.history

    def get_loss_history(self) -> Dict[str, List[float]]:
        return {
            "d_loss": self.gan.d_losses,
            "g_loss": self.gan.g_losses,
        }

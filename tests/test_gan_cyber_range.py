"""Tests for GAN Cyber Range v2."""

import pytest
import numpy as np

from cyberrange.traffic_generator import NetworkTrafficGenerator, N_FEATURES
from cyberrange.attack_gan import AttackGAN, Generator, Discriminator
from cyberrange.cyber_range_env import CyberRangeEnv, SimpleDetector
from cyberrange.training_loop import TrainingLoop


# ── NetworkTrafficGenerator ────────────────────────────────────────────────────

class TestNetworkTrafficGenerator:
    def test_normal_traffic_shape(self):
        gen = NetworkTrafficGenerator()
        flows = gen.generate_normal_traffic(100)
        assert flows.shape == (100, N_FEATURES)

    def test_normal_traffic_in_range(self):
        gen = NetworkTrafficGenerator()
        flows = gen.generate_normal_traffic(200)
        assert np.all(flows >= 0.0)
        assert np.all(flows <= 1.0)

    def test_attack_traffic_syn_flood(self):
        gen = NetworkTrafficGenerator()
        flows = gen.generate_attack_traffic(50, attack_type="syn_flood")
        assert flows.shape == (50, N_FEATURES)
        assert np.all(flows >= 0)
        assert np.all(flows <= 1)

    def test_attack_traffic_port_scan(self):
        gen = NetworkTrafficGenerator()
        flows = gen.generate_attack_traffic(50, attack_type="port_scan")
        assert flows.shape == (50, N_FEATURES)

    def test_attack_traffic_data_exfil(self):
        gen = NetworkTrafficGenerator()
        flows = gen.generate_attack_traffic(50, attack_type="data_exfil")
        assert flows.shape == (50, N_FEATURES)

    def test_attack_vs_normal_are_different(self):
        gen = NetworkTrafficGenerator()
        normal = gen.generate_normal_traffic(100)
        attacks = gen.generate_attack_traffic(100, attack_type="syn_flood")
        # Means should differ
        assert not np.allclose(normal.mean(axis=0), attacks.mean(axis=0), atol=0.05)

    def test_n_features_property(self):
        gen = NetworkTrafficGenerator()
        assert gen.n_features == N_FEATURES


# ── Generator & Discriminator ──────────────────────────────────────────────────

class TestGANModels:
    def test_generator_output_shape(self):
        g = Generator(latent_dim=16, output_dim=10)
        z = np.random.randn(32, 16)
        out = g.forward(z)
        assert out.shape == (32, 10)

    def test_generator_output_in_range(self):
        g = Generator(latent_dim=16, output_dim=N_FEATURES)
        z = np.random.randn(50, 16)
        out = g.forward(z)
        assert np.all(out >= 0)
        assert np.all(out <= 1)

    def test_discriminator_output_shape(self):
        d = Discriminator(input_dim=N_FEATURES)
        x = np.random.rand(20, N_FEATURES)
        out = d.forward(x)
        assert out.shape == (20, 1)

    def test_discriminator_output_probability(self):
        d = Discriminator(input_dim=N_FEATURES)
        x = np.random.rand(20, N_FEATURES)
        out = d.forward(x)
        assert np.all(out >= 0)
        assert np.all(out <= 1)


# ── AttackGAN ──────────────────────────────────────────────────────────────────

class TestAttackGAN:
    def test_generate_shape(self):
        gan = AttackGAN(latent_dim=16)
        samples = gan.generate(50)
        assert samples.shape == (50, N_FEATURES)

    def test_generate_in_range(self):
        gan = AttackGAN()
        samples = gan.generate(100)
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_discriminate_shape(self):
        gan = AttackGAN()
        x = np.random.rand(30, N_FEATURES)
        preds = gan.discriminate(x)
        assert preds.shape == (30, 1)

    def test_d_loss_positive(self):
        gan = AttackGAN()
        gen_flow = NetworkTrafficGenerator()
        real = gen_flow.generate_attack_traffic(32)
        fake = gan.generate(32)
        loss = gan.d_loss(real, fake)
        assert loss >= 0

    def test_g_loss_positive(self):
        gan = AttackGAN()
        fake = gan.generate(32)
        loss = gan.g_loss(fake)
        assert loss >= 0


# ── CyberRangeEnv ──────────────────────────────────────────────────────────────

class TestCyberRangeEnv:
    def test_inject_returns_metrics(self):
        env = CyberRangeEnv(n_normal_flows=100)
        gen = NetworkTrafficGenerator()
        attacks = gen.generate_attack_traffic(50)
        metrics = env.inject_attacks(attacks)
        assert "detection_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "n_attacks_injected" in metrics

    def test_detection_rate_in_range(self):
        env = CyberRangeEnv(n_normal_flows=100)
        gen = NetworkTrafficGenerator()
        attacks = gen.generate_attack_traffic(50)
        metrics = env.inject_attacks(attacks)
        assert 0.0 <= metrics["detection_rate"] <= 1.0

    def test_log_grows(self):
        env = CyberRangeEnv(n_normal_flows=100)
        gen = NetworkTrafficGenerator()
        for _ in range(3):
            env.inject_attacks(gen.generate_attack_traffic(10))
        assert len(env.get_log()) == 3

    def test_reset_clears_log(self):
        env = CyberRangeEnv(n_normal_flows=100)
        gen = NetworkTrafficGenerator()
        env.inject_attacks(gen.generate_attack_traffic(10))
        env.reset()
        assert len(env.get_log()) == 0

    def test_detector_high_sensitivity(self):
        """High sensitivity should detect more attacks."""
        gen = NetworkTrafficGenerator()
        attacks = gen.generate_attack_traffic(100, attack_type="syn_flood")
        env_hi = CyberRangeEnv(n_normal_flows=100, detector_sensitivity=0.9)
        env_lo = CyberRangeEnv(n_normal_flows=100, detector_sensitivity=0.1)
        rate_hi = env_hi.inject_attacks(attacks)["detection_rate"]
        rate_lo = env_lo.inject_attacks(attacks)["detection_rate"]
        assert rate_hi >= rate_lo


# ── TrainingLoop ───────────────────────────────────────────────────────────────

class TestTrainingLoop:
    def test_train_runs(self):
        gan = AttackGAN(latent_dim=16)
        gen = NetworkTrafficGenerator()
        loop = TrainingLoop(gan, gen, batch_size=32, lr=1e-3)
        history = loop.train(n_epochs=3)
        assert len(history) == 3

    def test_losses_in_history(self):
        gan = AttackGAN(latent_dim=16)
        gen = NetworkTrafficGenerator()
        loop = TrainingLoop(gan, gen, batch_size=32)
        history = loop.train(n_epochs=2)
        for entry in history:
            assert "d_loss" in entry
            assert "g_loss" in entry

    def test_loss_history_shape(self):
        gan = AttackGAN(latent_dim=16)
        gen = NetworkTrafficGenerator()
        loop = TrainingLoop(gan, gen, batch_size=32)
        loop.train(n_epochs=4)
        lh = loop.get_loss_history()
        assert len(lh["d_loss"]) == 4
        assert len(lh["g_loss"]) == 4

    def test_callback_called(self):
        gan = AttackGAN(latent_dim=16)
        gen = NetworkTrafficGenerator()
        loop = TrainingLoop(gan, gen, batch_size=32)
        calls = []
        loop.train(n_epochs=3, callback=lambda e, m: calls.append(e))
        assert calls == [0, 1, 2]

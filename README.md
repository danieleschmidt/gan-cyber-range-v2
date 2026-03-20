# gan-cyber-range-v2

GAN-based attack generation for cyber range training. Generates realistic synthetic attack traffic for training network intrusion detection systems.

## Components

- **NetworkTrafficGenerator** – Synthetic TCP/UDP flow feature vectors (normal + attack types)
- **AttackGAN** – Generator + Discriminator MLP pair for realistic attack synthesis
- **CyberRangeEnv** – Inject generated attacks into simulated network, score detection rate
- **TrainingLoop** – GAN training with gradient clipping

## Usage

```python
from cyberrange.traffic_generator import NetworkTrafficGenerator
from cyberrange.attack_gan import AttackGAN
from cyberrange.cyber_range_env import CyberRangeEnv
from cyberrange.training_loop import TrainingLoop

# Train the GAN
gan = AttackGAN(latent_dim=32)
gen = NetworkTrafficGenerator()
loop = TrainingLoop(gan, gen, batch_size=64, lr=1e-3)
history = loop.train(n_epochs=20, attack_type="syn_flood")

# Generate attacks and test detection
env = CyberRangeEnv(n_normal_flows=1000, detector_sensitivity=0.5)
attacks = gan.generate(200)
metrics = env.inject_attacks(attacks)
print(f"Detection rate: {metrics['detection_rate']:.1%}")
print(f"False positive rate: {metrics['false_positive_rate']:.1%}")
```

## Attack Types

- `syn_flood` – High-rate SYN packet floods
- `port_scan` – Sequential port scanning
- `data_exfil` – Large data transfer to unusual destinations

## Development

```bash
pip install numpy
pytest tests/
```

"""NetworkTrafficGenerator: synthetic TCP/UDP flow features."""

import numpy as np
from typing import Dict, List, Optional, Tuple


# Feature names for network flow
FLOW_FEATURES = [
    "duration",           # flow duration in seconds
    "packet_count",       # total packets
    "byte_count",         # total bytes
    "packets_per_sec",    # packets / second
    "bytes_per_packet",   # bytes / packet
    "src_port",           # source port (normalized 0-1)
    "dst_port",           # destination port (normalized 0-1)
    "ttl",                # time-to-live (normalized)
    "tcp_flags",          # TCP flags bitfield (normalized)
    "inter_arrival_mean", # mean inter-arrival time
]

N_FEATURES = len(FLOW_FEATURES)


class NetworkTrafficGenerator:
    """Generate synthetic TCP/UDP network flow feature vectors."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.feature_names = FLOW_FEATURES

    def generate_normal_traffic(self, n: int) -> np.ndarray:
        """Generate n normal (benign) flow vectors, shape (n, N_FEATURES)."""
        samples = np.zeros((n, N_FEATURES))

        # duration: 0.1s to 30s, log-normal
        samples[:, 0] = np.clip(self.rng.lognormal(1.0, 1.2, n), 0.01, 100.0) / 100.0

        # packet_count: 5-1000
        samples[:, 1] = np.clip(self.rng.lognormal(3.0, 1.0, n), 1, 5000) / 5000.0

        # byte_count: 100-100000
        samples[:, 2] = np.clip(self.rng.lognormal(8.0, 1.5, n), 10, 1e6) / 1e6

        # packets_per_sec: 1-200
        samples[:, 3] = np.clip(self.rng.exponential(20, n), 0.1, 500) / 500.0

        # bytes_per_packet: 40-1500 (typical)
        samples[:, 4] = np.clip(self.rng.normal(500, 200, n), 40, 1500) / 1500.0

        # src_port: uniform over all ports
        samples[:, 5] = self.rng.uniform(0.3, 1.0, n)  # ephemeral ports

        # dst_port: common service ports
        dst_ports = self.rng.choice([80, 443, 22, 53, 8080, 3306], n)
        samples[:, 6] = dst_ports / 65535.0

        # ttl: 64 or 128
        samples[:, 7] = self.rng.choice([64, 128], n) / 255.0

        # tcp_flags: normal connection flags (SYN, SYN-ACK, ACK patterns)
        samples[:, 8] = self.rng.choice([0x02, 0x12, 0x10, 0x11], n) / 255.0

        # inter_arrival: normal 0-100ms
        samples[:, 9] = np.clip(self.rng.exponential(0.01, n), 0, 1.0)

        return np.clip(samples, 0.0, 1.0)

    def generate_attack_traffic(self, n: int, attack_type: str = "syn_flood") -> np.ndarray:
        """Generate n attack flow vectors."""
        if attack_type == "syn_flood":
            return self._syn_flood(n)
        elif attack_type == "port_scan":
            return self._port_scan(n)
        elif attack_type == "data_exfil":
            return self._data_exfil(n)
        else:
            return self._syn_flood(n)

    def _syn_flood(self, n: int) -> np.ndarray:
        """SYN flood: many short flows, high packet rate, SYN flags."""
        samples = np.zeros((n, N_FEATURES))
        samples[:, 0] = np.clip(self.rng.uniform(0.001, 0.01, n), 0, 1)  # very short
        samples[:, 1] = np.clip(self.rng.uniform(0.001, 0.01, n), 0, 1)  # few packets
        samples[:, 2] = np.clip(self.rng.uniform(0.0, 0.001, n), 0, 1)   # tiny bytes
        samples[:, 3] = np.clip(self.rng.uniform(0.8, 1.0, n), 0, 1)     # high rate
        samples[:, 4] = np.clip(self.rng.uniform(0.03, 0.1, n), 0, 1)    # small packets
        samples[:, 5] = self.rng.uniform(0.0, 1.0, n)
        samples[:, 6] = np.clip(self.rng.uniform(0.0, 0.1, n), 0, 1)     # victim port
        samples[:, 7] = self.rng.choice([64, 128], n) / 255.0
        samples[:, 8] = np.full(n, 0x02 / 255.0)                          # SYN only
        samples[:, 9] = np.clip(self.rng.uniform(0.0, 0.001, n), 0, 1)
        return np.clip(samples, 0.0, 1.0)

    def _port_scan(self, n: int) -> np.ndarray:
        """Port scan: many distinct dst ports, short flows."""
        samples = self.generate_normal_traffic(n)
        samples[:, 3] = np.clip(self.rng.uniform(0.7, 1.0, n), 0, 1)
        samples[:, 6] = self.rng.uniform(0.0, 1.0, n)  # random dst ports
        samples[:, 0] = np.clip(self.rng.uniform(0.0, 0.01, n), 0, 1)
        return np.clip(samples, 0.0, 1.0)

    def _data_exfil(self, n: int) -> np.ndarray:
        """Data exfiltration: large byte counts, unusual dst."""
        samples = self.generate_normal_traffic(n)
        samples[:, 2] = np.clip(self.rng.uniform(0.5, 1.0, n), 0, 1)   # large bytes
        samples[:, 4] = np.clip(self.rng.uniform(0.8, 1.0, n), 0, 1)   # large packets
        samples[:, 6] = self.rng.uniform(0.0, 0.3, n)                   # unusual port
        return np.clip(samples, 0.0, 1.0)

    @property
    def n_features(self) -> int:
        return N_FEATURES

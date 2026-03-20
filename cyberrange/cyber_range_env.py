"""CyberRangeEnv: inject generated attacks into simulated network, score detection rate."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .traffic_generator import NetworkTrafficGenerator, N_FEATURES


class SimpleDetector:
    """
    Rule-based detector with adjustable sensitivity.
    Uses feature thresholds to classify traffic as attack/normal.
    """

    def __init__(self, sensitivity: float = 0.5):
        """
        Args:
            sensitivity: detection threshold in [0,1].
                         Higher = more aggressive detection (more false positives too).
        """
        self.sensitivity = sensitivity

    def detect(self, flows: np.ndarray) -> np.ndarray:
        """
        Classify each flow as attack (1) or normal (0).
        
        Returns binary array of shape (n,).
        """
        scores = np.zeros(len(flows))

        # High packet rate (feature 3)
        scores += (flows[:, 3] > (1.0 - self.sensitivity * 0.7)).astype(float) * 0.3

        # Very short duration (feature 0)
        scores += (flows[:, 0] < self.sensitivity * 0.05).astype(float) * 0.2

        # SYN-only flags (feature 8 near 0.008)
        scores += (np.abs(flows[:, 8] - 0.008) < 0.005).astype(float) * 0.3

        # Unusual dst port (feature 6 very low or random)
        scores += (flows[:, 6] > 0.9).astype(float) * 0.2

        # Higher sensitivity = lower threshold = detect more (more true positives + false positives)
        threshold = 1.0 - self.sensitivity
        return (scores >= threshold).astype(int)


class CyberRangeEnv:
    """Simulated network environment for cyber range training."""

    def __init__(
        self,
        n_normal_flows: int = 1000,
        detector_sensitivity: float = 0.5,
        seed: int = 42,
    ):
        self.n_normal_flows = n_normal_flows
        self.detector = SimpleDetector(sensitivity=detector_sensitivity)
        self.traffic_gen = NetworkTrafficGenerator(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Pre-generate baseline normal traffic
        self.normal_traffic = self.traffic_gen.generate_normal_traffic(n_normal_flows)
        self._injection_log: List[Dict] = []

    def inject_attacks(self, attack_flows: np.ndarray) -> Dict:
        """
        Inject attack flows into the network simulation.
        
        Returns detection metrics.
        """
        n_attacks = len(attack_flows)

        # Detect attacks
        detections = self.detector.detect(attack_flows)
        detected = int(detections.sum())

        # False positive rate on normal traffic sample
        normal_sample = self.normal_traffic[
            self.rng.integers(0, self.n_normal_flows, min(200, n_attacks))
        ]
        fp_detections = self.detector.detect(normal_sample)
        fp_rate = float(fp_detections.mean())

        detection_rate = detected / max(n_attacks, 1)

        result = {
            "n_attacks_injected": n_attacks,
            "n_detected": detected,
            "detection_rate": detection_rate,
            "false_positive_rate": fp_rate,
            "f1_score": self._f1(detection_rate, fp_rate),
        }
        self._injection_log.append(result)
        return result

    def _f1(self, tpr: float, fpr: float) -> float:
        """Compute approximate F1 from TPR and FPR."""
        precision = tpr / max(tpr + fpr, 1e-6)
        recall = tpr
        if precision + recall < 1e-6:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def get_log(self) -> List[Dict]:
        return list(self._injection_log)

    def reset(self):
        self._injection_log.clear()
        self.normal_traffic = self.traffic_gen.generate_normal_traffic(self.n_normal_flows)

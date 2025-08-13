"""
GAN-based network attack pattern generation.

This module generates realistic network attack patterns including port scans,
DDoS attacks, lateral movement, and network reconnaissance activities.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import ipaddress
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class NetworkAttackPattern:
    """Represents a network attack pattern"""
    pattern_id: str
    attack_type: str
    source_ips: List[str]
    target_ips: List[str]
    ports: List[int]
    protocols: List[str]
    packet_rate: float
    duration: int  # seconds
    stealth_score: float
    sophistication_level: str


@dataclass
class NetworkFlow:
    """Represents a network flow in an attack"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    byte_count: int
    flow_duration: float
    flags: List[str]


class NetworkAttackGenerator(nn.Module):
    """GAN generator for network attack patterns"""
    
    def __init__(
        self,
        noise_dim: int = 64,
        output_dim: int = 256,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
            
        self.noise_dim = noise_dim
        
        # Generator architecture
        layers = []
        input_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
            
        # Output layer
        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        ])
        
        self.main = nn.Sequential(*layers)
        
        # Separate heads for different attack components
        self.ip_head = nn.Linear(output_dim, 64)      # IP address features
        self.port_head = nn.Linear(output_dim, 32)    # Port features  
        self.timing_head = nn.Linear(output_dim, 16)  # Timing features
        self.behavior_head = nn.Linear(output_dim, 32) # Behavior features
        
    def forward(self, noise: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate network attack features"""
        
        # Main feature extraction
        features = self.main(noise)
        
        # Generate specialized features
        outputs = {
            'ip_features': torch.sigmoid(self.ip_head(features)),
            'port_features': torch.sigmoid(self.port_head(features)),
            'timing_features': torch.sigmoid(self.timing_head(features)),
            'behavior_features': torch.tanh(self.behavior_head(features))
        }
        
        return outputs


class NetworkAttackDiscriminator(nn.Module):
    """Discriminator for network attack GAN"""
    
    def __init__(self, input_dim: int = 144):  # Sum of feature dimensions
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Classify network attack patterns as real or synthetic"""
        
        # Concatenate all features
        combined_features = torch.cat([
            features['ip_features'],
            features['port_features'], 
            features['timing_features'],
            features['behavior_features']
        ], dim=1)
        
        return self.main(combined_features)


class NetworkAttackGAN:
    """Complete GAN system for network attack generation"""
    
    def __init__(
        self,
        noise_dim: int = 64,
        device: str = "auto"
    ):
        self.noise_dim = noise_dim
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize networks
        self.generator = NetworkAttackGenerator(noise_dim=noise_dim).to(self.device)
        self.discriminator = NetworkAttackDiscriminator().to(self.device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0001,
            betas=(0.5, 0.999)
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        # Training history
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'attack_diversity': []
        }
        
        logger.info(f"Initialized NetworkAttackGAN on device: {self.device}")
        
    def train(
        self,
        real_attack_data: Dict[str, torch.Tensor],
        epochs: int = 1000,
        batch_size: int = 64
    ) -> Dict[str, List[float]]:
        """Train the network attack GAN"""
        
        logger.info(f"Starting network attack GAN training for {epochs} epochs")
        
        # Create data loader from feature tensors
        combined_real = torch.cat([
            real_attack_data['ip_features'],
            real_attack_data['port_features'],
            real_attack_data['timing_features'],
            real_attack_data['behavior_features']
        ], dim=1)
        
        dataset = torch.utils.data.TensorDataset(combined_real)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        for epoch in range(epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                
                # Split back into feature components
                real_features = self._split_features(real_batch)
                
                # Train discriminator
                d_loss = self._train_discriminator(real_features)
                epoch_d_losses.append(d_loss)
                
                # Train generator
                g_loss = self._train_generator(real_batch.size(0))
                epoch_g_losses.append(g_loss)
                
            # Record epoch metrics
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['d_loss'].append(avg_d_loss)
            
            # Log progress
            if epoch % 100 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"G Loss: {avg_g_loss:.4f}, "
                    f"D Loss: {avg_d_loss:.4f}"
                )
                
        logger.info("Network attack GAN training completed")
        return self.training_history
        
    def generate_attack_patterns(
        self,
        num_patterns: int = 100,
        attack_types: List[str] = None
    ) -> List[NetworkAttackPattern]:
        """Generate synthetic network attack patterns"""
        
        if attack_types is None:
            attack_types = ['port_scan', 'ddos', 'lateral_movement', 'reconnaissance']
            
        logger.info(f"Generating {num_patterns} network attack patterns")
        
        self.generator.eval()
        patterns = []
        
        with torch.no_grad():
            batch_size = 32
            for i in range(0, num_patterns, batch_size):
                current_batch_size = min(batch_size, num_patterns - i)
                
                # Generate noise
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                
                # Generate features
                features = self.generator(noise)
                
                # Convert to attack patterns
                for j in range(current_batch_size):
                    pattern = self._create_attack_pattern(
                        {key: features[key][j] for key in features},
                        attack_types
                    )
                    patterns.append(pattern)
                    
        logger.info(f"Generated {len(patterns)} network attack patterns")
        return patterns
        
    def generate_network_flows(
        self,
        attack_pattern: NetworkAttackPattern,
        num_flows: int = 1000
    ) -> List[NetworkFlow]:
        """Generate detailed network flows for an attack pattern"""
        
        logger.info(f"Generating {num_flows} network flows for {attack_pattern.attack_type}")
        
        flows = []
        
        for i in range(num_flows):
            flow = self._create_network_flow(attack_pattern, i)
            flows.append(flow)
            
        return flows
        
    def _train_discriminator(self, real_features: Dict[str, torch.Tensor]) -> float:
        """Train discriminator for one step"""
        
        self.d_optimizer.zero_grad()
        batch_size = real_features['ip_features'].size(0)
        
        # Train on real data
        real_output = self.discriminator(real_features)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_loss = nn.BCELoss()(real_output, real_labels)
        
        # Train on fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_features = self.generator(noise)
        fake_output = self.discriminator(fake_features)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_loss = nn.BCELoss()(fake_output, fake_labels)
        
        # Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
        
    def _train_generator(self, batch_size: int) -> float:
        """Train generator for one step"""
        
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_features = self.generator(noise)
        fake_output = self.discriminator(fake_features)
        
        # Generator wants discriminator to think fake is real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = nn.BCELoss()(fake_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
        
    def _split_features(self, combined_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split combined feature tensor back into components"""
        
        ip_features = combined_features[:, :64]
        port_features = combined_features[:, 64:96]
        timing_features = combined_features[:, 96:112]
        behavior_features = combined_features[:, 112:144]
        
        return {
            'ip_features': ip_features,
            'port_features': port_features,
            'timing_features': timing_features,
            'behavior_features': behavior_features
        }
        
    def _create_attack_pattern(
        self,
        features: Dict[str, torch.Tensor],
        attack_types: List[str]
    ) -> NetworkAttackPattern:
        """Create NetworkAttackPattern from generated features"""
        
        # Convert features to CPU numpy
        ip_feat = features['ip_features'].cpu().numpy()
        port_feat = features['port_features'].cpu().numpy()
        timing_feat = features['timing_features'].cpu().numpy()
        behavior_feat = features['behavior_features'].cpu().numpy()
        
        # Determine attack type
        attack_type = random.choice(attack_types)
        
        # Generate IPs based on features
        source_ips = self._generate_ips_from_features(ip_feat[:32], 'source')
        target_ips = self._generate_ips_from_features(ip_feat[32:], 'target')
        
        # Generate ports based on features and attack type
        ports = self._generate_ports_from_features(port_feat, attack_type)
        
        # Generate protocols
        protocols = self._determine_protocols(attack_type, behavior_feat)
        
        # Extract timing and behavioral characteristics
        packet_rate = float(timing_feat[0] * 10000)  # Scale to realistic range
        duration = int(timing_feat[1] * 3600)  # Up to 1 hour
        stealth_score = float(behavior_feat[0])
        
        # Determine sophistication
        sophistication_level = self._determine_sophistication(behavior_feat)
        
        return NetworkAttackPattern(
            pattern_id=f"netattack_{random.randint(100000, 999999)}",
            attack_type=attack_type,
            source_ips=source_ips,
            target_ips=target_ips,
            ports=ports,
            protocols=protocols,
            packet_rate=packet_rate,
            duration=duration,
            stealth_score=stealth_score,
            sophistication_level=sophistication_level
        )
        
    def _generate_ips_from_features(self, features: np.ndarray, ip_type: str) -> List[str]:
        """Generate IP addresses from features"""
        
        ips = []
        
        # Generate 1-5 IPs based on features
        num_ips = min(5, max(1, int(features[0] * 5)))
        
        for i in range(num_ips):
            if ip_type == 'source':
                # External/attacker IPs
                if features[i + 1] > 0.7:
                    # High feature value -> external IP
                    ip = f"{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                else:
                    # Internal compromised IP
                    ip = f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
            else:
                # Target IPs - usually internal
                ip = f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
                
            ips.append(ip)
            
        return ips
        
    def _generate_ports_from_features(self, features: np.ndarray, attack_type: str) -> List[int]:
        """Generate port numbers from features"""
        
        # Common ports by attack type
        common_ports = {
            'port_scan': [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995],
            'ddos': [80, 443, 53, 25],
            'lateral_movement': [22, 135, 139, 445, 3389, 5985],
            'reconnaissance': [21, 22, 23, 25, 53, 80, 443]
        }
        
        base_ports = common_ports.get(attack_type, [80, 443, 22])
        
        # Select ports based on features
        num_ports = min(10, max(1, int(features[0] * 10)))
        selected_ports = random.sample(base_ports, min(num_ports, len(base_ports)))
        
        # Add random high ports if features suggest it
        if features[1] > 0.6:
            for _ in range(min(3, num_ports - len(selected_ports))):
                selected_ports.append(random.randint(1024, 65535))
                
        return selected_ports
        
    def _determine_protocols(self, attack_type: str, behavior_features: np.ndarray) -> List[str]:
        """Determine protocols used in attack"""
        
        protocol_prefs = {
            'port_scan': ['tcp'],
            'ddos': ['tcp', 'udp', 'icmp'],
            'lateral_movement': ['tcp'],
            'reconnaissance': ['tcp', 'udp']
        }
        
        base_protocols = protocol_prefs.get(attack_type, ['tcp'])
        
        # Add protocols based on behavior features
        if behavior_features[1] > 0.5:
            base_protocols.append('udp')
        if behavior_features[2] > 0.7:
            base_protocols.append('icmp')
            
        return list(set(base_protocols))
        
    def _determine_sophistication(self, behavior_features: np.ndarray) -> str:
        """Determine attack sophistication level"""
        
        avg_behavior = np.mean(np.abs(behavior_features))
        
        if avg_behavior > 0.8:
            return 'advanced'
        elif avg_behavior > 0.5:
            return 'intermediate'
        else:
            return 'basic'
            
    def _create_network_flow(self, pattern: NetworkAttackPattern, flow_index: int) -> NetworkFlow:
        """Create a detailed network flow for an attack pattern"""
        
        # Select random source and target
        src_ip = random.choice(pattern.source_ips)
        dst_ip = random.choice(pattern.target_ips)
        
        # Select ports
        dst_port = random.choice(pattern.ports)
        src_port = random.randint(1024, 65535)
        
        # Select protocol
        protocol = random.choice(pattern.protocols)
        
        # Generate flow characteristics based on attack type
        if pattern.attack_type == 'port_scan':
            packet_count = random.randint(1, 5)
            byte_count = packet_count * random.randint(64, 128)
            flow_duration = random.uniform(0.1, 1.0)
            flags = ['SYN'] if protocol == 'tcp' else []
            
        elif pattern.attack_type == 'ddos':
            packet_count = random.randint(100, 10000)
            byte_count = packet_count * random.randint(64, 1500)
            flow_duration = random.uniform(1.0, 60.0)
            flags = ['SYN', 'ACK'] if protocol == 'tcp' else []
            
        elif pattern.attack_type == 'lateral_movement':
            packet_count = random.randint(10, 100)
            byte_count = packet_count * random.randint(128, 1024)
            flow_duration = random.uniform(5.0, 300.0)
            flags = ['SYN', 'ACK', 'PSH', 'FIN'] if protocol == 'tcp' else []
            
        else:  # reconnaissance
            packet_count = random.randint(5, 50)
            byte_count = packet_count * random.randint(64, 512)
            flow_duration = random.uniform(1.0, 30.0)
            flags = ['SYN', 'ACK'] if protocol == 'tcp' else []
            
        return NetworkFlow(
            flow_id=f"flow_{pattern.pattern_id}_{flow_index}",
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            packet_count=packet_count,
            byte_count=byte_count,
            flow_duration=flow_duration,
            flags=flags
        )
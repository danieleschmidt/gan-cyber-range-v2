"""
Core GAN architecture for generating synthetic cyberattacks.

This module implements the foundational GAN architecture that generates realistic
attack patterns across multiple vectors for defensive training purposes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AttackVector:
    """Represents a synthetic attack vector with metadata"""
    attack_type: str
    payload: Union[str, bytes, Dict]
    techniques: List[str]
    severity: float
    stealth_level: float
    target_systems: List[str]
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None


class Generator(nn.Module):
    """GAN Generator for creating synthetic attack patterns"""
    
    def __init__(self, noise_dim: int = 100, output_dim: int = 512, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]
            
        layers = []
        input_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
            
        layers.extend([
            nn.Linear(input_dim, output_dim),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.model(noise)


class Discriminator(nn.Module):
    """GAN Discriminator for distinguishing real vs synthetic attacks"""
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]
            
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
            
        layers.extend([
            nn.Linear(current_dim, 1),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AttackVectorizer:
    """Converts attack patterns to/from vector representations"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 512):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.reverse_vocab = {}
        
    def fit(self, attack_data: List[str]) -> None:
        """Build vocabulary from attack data"""
        all_tokens = []
        for attack in attack_data:
            tokens = self._tokenize(attack)
            all_tokens.extend(tokens)
            
        unique_tokens = list(set(all_tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens[:self.vocab_size])}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
    def transform(self, attacks: Union[str, List[str]]) -> torch.Tensor:
        """Convert attacks to vector representation"""
        if isinstance(attacks, str):
            attacks = [attacks]
            
        vectors = []
        for attack in attacks:
            tokens = self._tokenize(attack)
            vector = torch.zeros(self.embedding_dim)
            
            for i, token in enumerate(tokens[:self.embedding_dim]):
                if token in self.vocab:
                    vector[i] = self.vocab[token] / self.vocab_size
                    
            vectors.append(vector)
            
        return torch.stack(vectors) if len(vectors) > 1 else vectors[0].unsqueeze(0)
    
    def inverse_transform(self, vectors: torch.Tensor) -> List[str]:
        """Convert vectors back to attack strings"""
        attacks = []
        
        for vector in vectors:
            tokens = []
            for val in vector:
                idx = int(val.item() * self.vocab_size)
                if idx in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[idx])
                    
            attacks.append(" ".join(tokens))
            
        return attacks
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for attack patterns"""
        return text.lower().replace('.', ' ').split()


class AttackGAN:
    """Main GAN class for generating synthetic cyberattacks"""
    
    def __init__(
        self,
        architecture: str = "wasserstein",
        attack_types: List[str] = None,
        noise_dim: int = 100,
        training_mode: str = "standard",
        device: str = "auto"
    ):
        self.architecture = architecture
        self.attack_types = attack_types or ["malware", "network", "web", "social_engineering"]
        self.noise_dim = noise_dim
        self.training_mode = training_mode
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initialized AttackGAN on device: {self.device}")
        
        # Initialize networks
        self.generator = Generator(noise_dim=noise_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
        
        # Initialize vectorizer
        self.vectorizer = AttackVectorizer()
        
        # Training history
        self.training_history = {
            "g_loss": [],
            "d_loss": [],
            "diversity_scores": []
        }
        
    def train(
        self,
        real_attacks: Union[str, List[str], Path],
        epochs: int = 1000,
        batch_size: int = 64,
        privacy_budget: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """Train the GAN on real attack data"""
        
        logger.info(f"Starting GAN training for {epochs} epochs")
        
        # Load and prepare data
        if isinstance(real_attacks, (str, Path)):
            attack_data = self._load_attack_data(real_attacks)
        else:
            attack_data = real_attacks
            
        # Fit vectorizer and convert to tensors
        self.vectorizer.fit(attack_data)
        attack_vectors = self.vectorizer.transform(attack_data)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(attack_vectors)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_batch,) in enumerate(dataloader):
                real_batch = real_batch.to(self.device)
                
                # Train Discriminator
                d_loss = self._train_discriminator(real_batch)
                d_losses.append(d_loss)
                
                # Train Generator
                g_loss = self._train_generator(real_batch.size(0))
                g_losses.append(g_loss)
                
            # Record epoch statistics
            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            
            self.training_history["g_loss"].append(avg_g_loss)
            self.training_history["d_loss"].append(avg_d_loss)
            
            # Log progress
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
                
        logger.info("Training completed successfully")
        return self.training_history
    
    def generate(
        self,
        num_samples: int = 1000,
        diversity_threshold: float = 0.8,
        filter_detectable: bool = True
    ) -> List[AttackVector]:
        """Generate synthetic attack vectors"""
        
        logger.info(f"Generating {num_samples} synthetic attacks")
        
        self.generator.eval()
        generated_attacks = []
        
        with torch.no_grad():
            # Generate in batches
            batch_size = 64
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                # Generate noise
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                
                # Generate vectors
                fake_vectors = self.generator(noise)
                
                # Convert to attack strings
                attack_strings = self.vectorizer.inverse_transform(fake_vectors.cpu())
                
                # Create AttackVector objects
                for attack_str in attack_strings:
                    attack_vector = self._create_attack_vector(attack_str)
                    if self._is_valid_attack(attack_vector, filter_detectable):
                        generated_attacks.append(attack_vector)
                        
        logger.info(f"Generated {len(generated_attacks)} valid attacks")
        return generated_attacks[:num_samples]
    
    def diversity_score(self, attacks: List[AttackVector]) -> float:
        """Calculate diversity score for generated attacks"""
        if len(attacks) < 2:
            return 0.0
            
        # Convert attacks to vectors for comparison
        attack_strings = [attack.payload if isinstance(attack.payload, str) else str(attack.payload) 
                         for attack in attacks]
        vectors = self.vectorizer.transform(attack_strings)
        
        # Calculate pairwise cosine similarities
        similarities = torch.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=2)
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(len(attacks), dtype=bool)
        similarities = similarities[mask]
        
        # Diversity is 1 - average similarity
        diversity = 1.0 - similarities.mean().item()
        return max(0.0, min(1.0, diversity))
    
    def save_model(self, path: Union[str, Path]) -> None:
        """Save trained model"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'training_history': self.training_history,
            'vectorizer_vocab': self.vectorizer.vocab,
            'config': {
                'architecture': self.architecture,
                'attack_types': self.attack_types,
                'noise_dim': self.noise_dim,
                'training_mode': self.training_mode
            }
        }, path / 'attack_gan_model.pth')
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """Load trained model"""
        path = Path(path) / 'attack_gan_model.pth'
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.vectorizer.vocab = checkpoint['vectorizer_vocab']
        
        logger.info(f"Model loaded from {path}")
    
    def _train_discriminator(self, real_batch: torch.Tensor) -> float:
        """Train discriminator for one step"""
        self.d_optimizer.zero_grad()
        
        batch_size = real_batch.size(0)
        
        # Train on real data
        real_output = self.discriminator(real_batch)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_loss = nn.BCELoss()(real_output, real_labels)
        
        # Train on fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_batch = self.generator(noise).detach()
        fake_output = self.discriminator(fake_batch)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_loss = nn.BCELoss()(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def _train_generator(self, batch_size: int) -> float:
        """Train generator for one step"""
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_batch = self.generator(noise)
        fake_output = self.discriminator(fake_batch)
        
        # Generator wants discriminator to think fake data is real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = nn.BCELoss()(fake_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def _load_attack_data(self, path: Union[str, Path]) -> List[str]:
        """Load attack data from file or directory"""
        path = Path(path)
        attack_data = []
        
        if path.is_file():
            with open(path, 'r') as f:
                attack_data = f.readlines()
        elif path.is_dir():
            for file_path in path.glob("*.txt"):
                with open(file_path, 'r') as f:
                    attack_data.extend(f.readlines())
                    
        # Clean data
        attack_data = [line.strip() for line in attack_data if line.strip()]
        
        logger.info(f"Loaded {len(attack_data)} attack samples")
        return attack_data
    
    def _create_attack_vector(self, attack_str: str) -> AttackVector:
        """Create AttackVector from generated string"""
        # Simple heuristic-based parsing
        attack_type = self._classify_attack_type(attack_str)
        techniques = self._extract_techniques(attack_str)
        severity = np.random.uniform(0.1, 1.0)  # Placeholder
        stealth_level = np.random.uniform(0.1, 1.0)  # Placeholder
        target_systems = ["generic"]  # Placeholder
        
        return AttackVector(
            attack_type=attack_type,
            payload=attack_str,
            techniques=techniques,
            severity=severity,
            stealth_level=stealth_level,
            target_systems=target_systems
        )
    
    def _classify_attack_type(self, attack_str: str) -> str:
        """Classify attack type from string"""
        attack_str_lower = attack_str.lower()
        
        if any(keyword in attack_str_lower for keyword in ['malware', 'virus', 'trojan', 'ransomware']):
            return 'malware'
        elif any(keyword in attack_str_lower for keyword in ['sql', 'xss', 'csrf', 'injection']):
            return 'web'
        elif any(keyword in attack_str_lower for keyword in ['phishing', 'social', 'email']):
            return 'social_engineering'
        elif any(keyword in attack_str_lower for keyword in ['network', 'port', 'scan', 'ddos']):
            return 'network'
        else:
            return np.random.choice(self.attack_types)
    
    def _extract_techniques(self, attack_str: str) -> List[str]:
        """Extract MITRE ATT&CK techniques from attack string"""
        # Placeholder implementation
        techniques = []
        if 'scan' in attack_str.lower():
            techniques.append('T1046')  # Network Service Scanning
        if 'credential' in attack_str.lower():
            techniques.append('T1110')  # Brute Force
        if 'lateral' in attack_str.lower():
            techniques.append('T1021')  # Remote Services
            
        return techniques or ['T1001']  # Default technique
    
    def _is_valid_attack(self, attack: AttackVector, filter_detectable: bool = True) -> bool:
        """Validate generated attack"""
        # Basic validation
        if not attack.payload or len(str(attack.payload)) < 10:
            return False
            
        # Filter easily detectable attacks if requested
        if filter_detectable and attack.stealth_level < 0.3:
            return False
            
        return True
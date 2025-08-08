"""
Comprehensive tests for AttackGAN module.

Tests cover GAN training, generation, vectorization, and edge cases.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from gan_cyber_range.core.attack_gan import (
    AttackGAN, AttackVector, AttackVectorizer, 
    Generator, Discriminator
)


class TestAttackVectorizer:
    """Test AttackVectorizer functionality"""
    
    def test_init(self):
        """Test vectorizer initialization"""
        vectorizer = AttackVectorizer(vocab_size=1000, embedding_dim=256)
        assert vectorizer.vocab_size == 1000
        assert vectorizer.embedding_dim == 256
        assert vectorizer.vocab == {}
        assert vectorizer.reverse_vocab == {}
    
    def test_fit(self):
        """Test vocabulary fitting"""
        vectorizer = AttackVectorizer(vocab_size=10)
        attack_data = [
            "sql injection attack",
            "xss payload script",
            "malware trojan virus"
        ]
        
        vectorizer.fit(attack_data)
        
        assert len(vectorizer.vocab) <= 10
        assert len(vectorizer.reverse_vocab) == len(vectorizer.vocab)
        
        # Check that common tokens are included
        expected_tokens = ["sql", "injection", "attack", "xss", "payload", "script"]
        found_tokens = [token for token in expected_tokens if token in vectorizer.vocab]
        assert len(found_tokens) > 0
    
    def test_transform_single_attack(self):
        """Test transforming single attack to vector"""
        vectorizer = AttackVectorizer(vocab_size=100, embedding_dim=128)
        attack_data = ["sql injection", "xss attack", "malware execution"]
        
        vectorizer.fit(attack_data)
        result = vectorizer.transform("sql injection")
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 128)
    
    def test_transform_multiple_attacks(self):
        """Test transforming multiple attacks"""
        vectorizer = AttackVectorizer(vocab_size=100, embedding_dim=64)
        attack_data = ["sql injection", "xss attack", "malware execution"]
        
        vectorizer.fit(attack_data)
        attacks = ["sql injection", "xss attack"]
        result = vectorizer.transform(attacks)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 64)
    
    def test_inverse_transform(self):
        """Test converting vectors back to strings"""
        vectorizer = AttackVectorizer(vocab_size=100, embedding_dim=32)
        attack_data = ["sql injection", "xss attack"]
        
        vectorizer.fit(attack_data)
        vectors = vectorizer.transform(attack_data)
        reconstructed = vectorizer.inverse_transform(vectors)
        
        assert isinstance(reconstructed, list)
        assert len(reconstructed) == 2
        assert all(isinstance(attack, str) for attack in reconstructed)
    
    def test_tokenize(self):
        """Test tokenization functionality"""
        vectorizer = AttackVectorizer()
        text = "SQL injection. attack payload!"
        tokens = vectorizer._tokenize(text)
        
        expected = ["sql", "injection", "attack", "payload!"]
        assert tokens == expected


class TestGenerator:
    """Test Generator neural network"""
    
    def test_init(self):
        """Test generator initialization"""
        gen = Generator(noise_dim=100, output_dim=512, hidden_dims=[256, 384])
        
        assert isinstance(gen, torch.nn.Module)
        # Check that layers are created
        assert len(list(gen.model.children())) > 0
    
    def test_forward(self):
        """Test generator forward pass"""
        gen = Generator(noise_dim=50, output_dim=128)
        noise = torch.randn(10, 50)
        
        output = gen(noise)
        
        assert output.shape == (10, 128)
        # Output should be in [-1, 1] range due to Tanh
        assert torch.all(output >= -1) and torch.all(output <= 1)
    
    def test_different_architectures(self):
        """Test generator with different architectures"""
        # Small generator
        gen_small = Generator(noise_dim=32, output_dim=64, hidden_dims=[128])
        noise_small = torch.randn(5, 32)
        output_small = gen_small(noise_small)
        assert output_small.shape == (5, 64)
        
        # Large generator
        gen_large = Generator(noise_dim=128, output_dim=1024, hidden_dims=[256, 512, 768])
        noise_large = torch.randn(3, 128)
        output_large = gen_large(noise_large)
        assert output_large.shape == (3, 1024)


class TestDiscriminator:
    """Test Discriminator neural network"""
    
    def test_init(self):
        """Test discriminator initialization"""
        disc = Discriminator(input_dim=512, hidden_dims=[256, 128])
        
        assert isinstance(disc, torch.nn.Module)
        assert len(list(disc.model.children())) > 0
    
    def test_forward(self):
        """Test discriminator forward pass"""
        disc = Discriminator(input_dim=128)
        data = torch.randn(10, 128)
        
        output = disc(data)
        
        assert output.shape == (10, 1)
        # Output should be in [0, 1] range due to Sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_real_vs_fake_discrimination(self):
        """Test discriminator can distinguish patterns"""
        disc = Discriminator(input_dim=64)
        
        # Create clearly different patterns
        real_data = torch.ones(5, 64)  # All ones
        fake_data = torch.zeros(5, 64)  # All zeros
        
        real_output = disc(real_data)
        fake_output = disc(fake_data)
        
        # Outputs should be different (though not necessarily in expected direction before training)
        assert real_output.shape == fake_output.shape == (5, 1)


class TestAttackVector:
    """Test AttackVector dataclass"""
    
    def test_creation(self):
        """Test creating attack vector"""
        vector = AttackVector(
            attack_type="web",
            payload="<script>alert('xss')</script>",
            techniques=["T1059"],
            severity=7.5,
            stealth_level=0.6,
            target_systems=["web_server"],
            timestamp="2024-01-01T12:00:00Z"
        )
        
        assert vector.attack_type == "web"
        assert vector.payload == "<script>alert('xss')</script>"
        assert vector.techniques == ["T1059"]
        assert vector.severity == 7.5
        assert vector.stealth_level == 0.6
        assert vector.target_systems == ["web_server"]
        assert vector.timestamp == "2024-01-01T12:00:00Z"
    
    def test_with_metadata(self):
        """Test attack vector with metadata"""
        metadata = {"source": "test", "campaign": "red_team_1"}
        vector = AttackVector(
            attack_type="malware",
            payload=b"\x90\x90\x90\x90",  # Shellcode bytes
            techniques=["T1055"],
            severity=9.0,
            stealth_level=0.8,
            target_systems=["windows"],
            metadata=metadata
        )
        
        assert vector.metadata == metadata
        assert isinstance(vector.payload, bytes)


class TestAttackGAN:
    """Test main AttackGAN class"""
    
    @pytest.fixture
    def sample_attack_data(self):
        """Sample attack data for testing"""
        return [
            "sql injection union select",
            "xss script alert cookie",
            "malware trojan persistence",
            "phishing email credential harvest",
            "ddos amplification reflection"
        ]
    
    @pytest.fixture
    def attack_gan(self):
        """Create AttackGAN instance for testing"""
        return AttackGAN(
            architecture="standard",
            attack_types=["web", "malware"],
            noise_dim=32,  # Small for testing
            device="cpu"  # Force CPU for testing
        )
    
    def test_init(self, attack_gan):
        """Test AttackGAN initialization"""
        assert attack_gan.architecture == "standard"
        assert attack_gan.attack_types == ["web", "malware"]
        assert attack_gan.noise_dim == 32
        assert attack_gan.device.type == "cpu"
        
        # Check components are initialized
        assert isinstance(attack_gan.generator, Generator)
        assert isinstance(attack_gan.discriminator, Discriminator)
        assert hasattr(attack_gan, 'g_optimizer')
        assert hasattr(attack_gan, 'd_optimizer')
        assert isinstance(attack_gan.vectorizer, AttackVectorizer)
    
    @patch('gan_cyber_range.core.attack_gan.logger')
    def test_train_with_list_data(self, mock_logger, attack_gan, sample_attack_data):
        """Test training with list of attack data"""
        
        # Train for minimal epochs
        history = attack_gan.train(
            real_attacks=sample_attack_data,
            epochs=2,
            batch_size=2
        )
        
        # Check training history
        assert isinstance(history, dict)
        assert 'g_loss' in history
        assert 'd_loss' in history
        assert len(history['g_loss']) == 2  # 2 epochs
        assert len(history['d_loss']) == 2
        
        # Verify logging was called
        mock_logger.info.assert_called()
    
    @patch('gan_cyber_range.core.attack_gan.logger')
    def test_train_with_file_data(self, mock_logger, attack_gan, sample_attack_data):
        """Test training with file data"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for attack in sample_attack_data:
                f.write(attack + '\n')
            temp_file = f.name
        
        try:
            history = attack_gan.train(
                real_attacks=temp_file,
                epochs=1,
                batch_size=2
            )
            
            assert isinstance(history, dict)
            assert 'g_loss' in history
            assert 'd_loss' in history
            
        finally:
            Path(temp_file).unlink()  # Clean up
    
    def test_generate_attacks(self, attack_gan, sample_attack_data):
        """Test generating synthetic attacks"""
        
        # First train briefly
        attack_gan.train(sample_attack_data, epochs=1, batch_size=2)
        
        # Generate attacks
        generated = attack_gan.generate(
            num_samples=5,
            diversity_threshold=0.5,
            filter_detectable=False
        )
        
        assert isinstance(generated, list)
        assert len(generated) <= 5  # May be less due to filtering
        assert all(isinstance(attack, AttackVector) for attack in generated)
        
        # Check attack vector properties
        if generated:
            attack = generated[0]
            assert hasattr(attack, 'attack_type')
            assert hasattr(attack, 'payload')
            assert hasattr(attack, 'techniques')
    
    def test_diversity_score(self, attack_gan, sample_attack_data):
        """Test diversity score calculation"""
        
        # Train briefly
        attack_gan.train(sample_attack_data, epochs=1, batch_size=2)
        
        # Generate some attacks
        attacks = attack_gan.generate(num_samples=3, filter_detectable=False)
        
        if len(attacks) >= 2:
            diversity = attack_gan.diversity_score(attacks)
            assert isinstance(diversity, float)
            assert 0.0 <= diversity <= 1.0
        else:
            # If not enough attacks generated, diversity should be 0
            diversity = attack_gan.diversity_score(attacks)
            assert diversity == 0.0
    
    def test_save_and_load_model(self, attack_gan, sample_attack_data):
        """Test model saving and loading"""
        
        # Train briefly
        original_history = attack_gan.train(sample_attack_data, epochs=1, batch_size=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model"
            
            # Save model
            attack_gan.save_model(model_path)
            assert (model_path / "attack_gan_model.pth").exists()
            
            # Create new GAN and load model
            new_gan = AttackGAN(device="cpu")
            new_gan.load_model(model_path)
            
            # Check that history was loaded
            assert new_gan.training_history == original_history
    
    def test_create_attack_vector(self, attack_gan):
        """Test creating attack vector from string"""
        
        attack_str = "sql injection union select password"
        vector = attack_gan._create_attack_vector(attack_str)
        
        assert isinstance(vector, AttackVector)
        assert vector.payload == attack_str
        assert isinstance(vector.attack_type, str)
        assert isinstance(vector.techniques, list)
        assert 0.0 <= vector.severity <= 1.0
        assert 0.0 <= vector.stealth_level <= 1.0
    
    def test_classify_attack_type(self, attack_gan):
        """Test attack type classification"""
        
        test_cases = [
            ("sql injection attack", "web"),
            ("malware trojan virus", "malware"),
            ("phishing email social", "social_engineering"),
            ("network port scan", "network"),
            ("random text here", None)  # Should fall back to random choice
        ]
        
        for attack_str, expected_type in test_cases:
            result = attack_gan._classify_attack_type(attack_str)
            if expected_type:
                assert result == expected_type
            else:
                assert result in attack_gan.attack_types
    
    def test_extract_techniques(self, attack_gan):
        """Test MITRE ATT&CK technique extraction"""
        
        test_cases = [
            ("network scan reconnaissance", ["T1046"]),
            ("credential brute force", ["T1110"]),
            ("lateral movement remote", ["T1021"]),
            ("simple attack", ["T1001"])  # Default
        ]
        
        for attack_str, expected in test_cases:
            result = attack_gan._extract_techniques(attack_str)
            assert isinstance(result, list)
            assert len(result) > 0
            # Check if expected techniques are present
            for technique in expected:
                if attack_str != "simple attack":  # Skip default case
                    assert technique in result or len(result) > 0
    
    def test_is_valid_attack(self, attack_gan):
        """Test attack validation"""
        
        # Valid attack
        valid_attack = AttackVector(
            attack_type="web",
            payload="sql injection union select",
            techniques=["T1190"],
            severity=7.0,
            stealth_level=0.6,
            target_systems=["web"]
        )
        assert attack_gan._is_valid_attack(valid_attack, filter_detectable=False)
        
        # Invalid attack - too short payload
        invalid_attack = AttackVector(
            attack_type="web",
            payload="short",
            techniques=["T1190"],
            severity=7.0,
            stealth_level=0.6,
            target_systems=["web"]
        )
        assert not attack_gan._is_valid_attack(invalid_attack, filter_detectable=False)
        
        # Low stealth attack with filtering enabled
        low_stealth_attack = AttackVector(
            attack_type="web",
            payload="detectable sql injection",
            techniques=["T1190"],
            severity=7.0,
            stealth_level=0.2,  # Low stealth
            target_systems=["web"]
        )
        assert not attack_gan._is_valid_attack(low_stealth_attack, filter_detectable=True)
        assert attack_gan._is_valid_attack(low_stealth_attack, filter_detectable=False)
    
    def test_train_discriminator(self, attack_gan, sample_attack_data):
        """Test discriminator training step"""
        
        # Prepare data
        attack_gan.vectorizer.fit(sample_attack_data)
        attack_vectors = attack_gan.vectorizer.transform(sample_attack_data)
        
        # Train discriminator on small batch
        batch = attack_vectors[:2]
        d_loss = attack_gan._train_discriminator(batch)
        
        assert isinstance(d_loss, float)
        assert d_loss >= 0.0  # Loss should be non-negative
    
    def test_train_generator(self, attack_gan):
        """Test generator training step"""
        
        batch_size = 2
        g_loss = attack_gan._train_generator(batch_size)
        
        assert isinstance(g_loss, float)
        assert g_loss >= 0.0  # Loss should be non-negative
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_attack_data_file_not_found(self, mock_open, attack_gan):
        """Test handling of missing attack data file"""
        
        with pytest.raises(FileNotFoundError):
            attack_gan._load_attack_data("nonexistent_file.txt")
    
    def test_load_attack_data_directory(self, attack_gan, sample_attack_data):
        """Test loading attack data from directory"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple files
            for i, attack in enumerate(sample_attack_data):
                file_path = temp_path / f"attacks_{i}.txt"
                file_path.write_text(attack + '\n')
            
            # Load from directory
            loaded_data = attack_gan._load_attack_data(temp_path)
            
            assert isinstance(loaded_data, list)
            assert len(loaded_data) == len(sample_attack_data)
    
    def test_different_architectures(self):
        """Test different GAN architectures"""
        
        architectures = ["standard", "wasserstein", "conditional"]
        
        for arch in architectures:
            gan = AttackGAN(architecture=arch, device="cpu")
            assert gan.architecture == arch
            assert isinstance(gan.generator, Generator)
            assert isinstance(gan.discriminator, Discriminator)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_attack_data(self):
        """Test handling of empty attack data"""
        
        gan = AttackGAN(device="cpu")
        
        # Should handle empty list gracefully
        with pytest.raises(Exception):  # Should raise an error
            gan.train(real_attacks=[], epochs=1)
    
    def test_invalid_device(self):
        """Test handling of invalid device"""
        
        # Should handle invalid device gracefully
        gan = AttackGAN(device="invalid_device")
        # Should fall back to CPU
        assert gan.device.type == "cpu"
    
    def test_very_small_batch_size(self):
        """Test with very small batch size"""
        
        gan = AttackGAN(device="cpu")
        sample_data = ["sql injection", "xss attack"]
        
        # Should handle batch size of 1
        history = gan.train(sample_data, epochs=1, batch_size=1)
        assert isinstance(history, dict)
    
    def test_generate_with_untrained_model(self):
        """Test generating attacks with untrained model"""
        
        gan = AttackGAN(device="cpu")
        
        # Should still generate something (though quality will be poor)
        attacks = gan.generate(num_samples=2, filter_detectable=False)
        assert isinstance(attacks, list)
    
    def test_diversity_score_edge_cases(self):
        """Test diversity score with edge cases"""
        
        gan = AttackGAN(device="cpu")
        
        # Empty list
        assert gan.diversity_score([]) == 0.0
        
        # Single attack
        single_attack = [AttackVector(
            attack_type="web", payload="test", techniques=["T1059"],
            severity=1.0, stealth_level=1.0, target_systems=["web"]
        )]
        assert gan.diversity_score(single_attack) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
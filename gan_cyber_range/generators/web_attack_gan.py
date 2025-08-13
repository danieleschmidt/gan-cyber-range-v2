"""
GAN-based web attack generation for security training.

This module generates realistic web attack patterns including SQL injection,
XSS, CSRF, and other web application vulnerabilities for defensive training.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import string
import urllib.parse
import base64

logger = logging.getLogger(__name__)


@dataclass
class WebAttackPayload:
    """Represents a web attack payload"""
    payload_id: str
    attack_type: str
    payload_data: str
    target_parameter: str
    http_method: str
    success_probability: float
    evasion_techniques: List[str]
    encoding_type: str


@dataclass
class WebAttackSession:
    """Represents a complete web attack session"""
    session_id: str
    attack_type: str
    payloads: List[WebAttackPayload]
    target_url: str
    user_agent: str
    attack_vector: str
    sophistication_level: str
    duration: int  # seconds


class WebAttackGenerator(nn.Module):
    """GAN generator for web attack payloads"""
    
    def __init__(
        self,
        noise_dim: int = 128,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        max_length: int = 200
    ):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Embedding layer for payload tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequential payload generation
        self.lstm = nn.LSTM(
            input_size=embedding_dim + noise_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
        # Attack type classifier head
        self.attack_type_head = nn.Linear(hidden_dim, 8)  # Number of attack types
        
        # Sophistication head
        self.sophistication_head = nn.Linear(hidden_dim, 16)
        
    def forward(
        self, 
        noise: torch.Tensor,
        sequence_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate web attack payload sequences"""
        
        if sequence_length is None:
            sequence_length = self.max_length
            
        batch_size = noise.size(0)
        device = noise.device
        
        # Initialize hidden state
        h0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        
        # Start token (assume index 1 is start token)
        current_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(sequence_length):
            # Embed current token
            token_embed = self.embedding(current_token)
            
            # Expand noise for this timestep
            noise_expanded = noise.unsqueeze(1).expand(-1, 1, -1)
            
            # Concatenate token embedding with noise
            lstm_input = torch.cat([token_embed, noise_expanded], dim=2)
            
            # LSTM forward pass
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            # Project to vocabulary
            logits = self.output_proj(lstm_out.squeeze(1))
            outputs.append(logits)
            
            # Sample next token (during training) or use argmax (during inference)
            if self.training:
                probs = torch.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, 1)
            else:
                current_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Stack outputs
        payload_logits = torch.stack(outputs, dim=1)
        
        # Generate attack metadata from final hidden state
        attack_type_logits = self.attack_type_head(hidden[0][-1])
        sophistication_features = self.sophistication_head(hidden[0][-1])
        
        return {
            'payload_logits': payload_logits,
            'attack_type_logits': attack_type_logits,
            'sophistication_features': sophistication_features
        }


class WebAttackDiscriminator(nn.Module):
    """Discriminator for web attack GAN"""
    
    def __init__(
        self,
        vocab_size: int = 5000,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM for sequence classification
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, payload_sequences: torch.Tensor) -> torch.Tensor:
        """Classify payload sequences as real or synthetic"""
        
        # Embed sequences
        embedded = self.embedding(payload_sequences)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use final hidden state for classification
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classify
        output = self.classifier(final_hidden)
        
        return output


class WebAttackGAN:
    """Complete GAN system for web attack generation"""
    
    def __init__(
        self,
        vocab_size: int = 5000,
        noise_dim: int = 128,
        device: str = "auto"
    ):
        self.vocab_size = vocab_size
        self.noise_dim = noise_dim
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize vocabulary
        self.vocab = self._build_web_attack_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Initialize networks
        self.generator = WebAttackGenerator(
            noise_dim=noise_dim,
            vocab_size=len(self.vocab)
        ).to(self.device)
        
        self.discriminator = WebAttackDiscriminator(
            vocab_size=len(self.vocab)
        ).to(self.device)
        
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
            'payload_diversity': []
        }
        
        logger.info(f"Initialized WebAttackGAN on device: {self.device}")
        
    def train(
        self,
        real_payloads: List[str],
        epochs: int = 1000,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """Train the web attack GAN"""
        
        logger.info(f"Starting web attack GAN training for {epochs} epochs")
        
        # Tokenize and encode real payloads
        encoded_payloads = [self._encode_payload(payload) for payload in real_payloads]
        
        # Pad sequences
        max_len = max(len(seq) for seq in encoded_payloads)
        padded_payloads = []
        
        for seq in encoded_payloads:
            if len(seq) < max_len:
                seq.extend([0] * (max_len - len(seq)))  # Pad with 0
            padded_payloads.append(seq[:max_len])  # Truncate if too long
            
        payload_tensor = torch.tensor(padded_payloads, dtype=torch.long)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(payload_tensor)
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
                
                # Train discriminator
                d_loss = self._train_discriminator(real_batch)
                epoch_d_losses.append(d_loss)
                
                # Train generator
                g_loss = self._train_generator(real_batch.size(0), real_batch.size(1))
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
                
        logger.info("Web attack GAN training completed")
        return self.training_history
        
    def generate_web_attacks(
        self,
        num_attacks: int = 100,
        attack_types: List[str] = None
    ) -> List[WebAttackSession]:
        """Generate synthetic web attack sessions"""
        
        if attack_types is None:
            attack_types = ['sql_injection', 'xss', 'csrf', 'lfi', 'rfi', 'xxe', 'ssrf', 'cmd_injection']
            
        logger.info(f"Generating {num_attacks} web attack sessions")
        
        self.generator.eval()
        attack_sessions = []
        
        with torch.no_grad():
            batch_size = 16
            for i in range(0, num_attacks, batch_size):
                current_batch_size = min(batch_size, num_attacks - i)
                
                # Generate noise
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                
                # Generate attack features
                outputs = self.generator(noise, sequence_length=150)
                
                # Convert to attack sessions
                for j in range(current_batch_size):
                    session = self._create_attack_session(
                        outputs['payload_logits'][j],
                        outputs['attack_type_logits'][j],
                        outputs['sophistication_features'][j],
                        attack_types
                    )
                    attack_sessions.append(session)
                    
        logger.info(f"Generated {len(attack_sessions)} web attack sessions")
        return attack_sessions
        
    def _train_discriminator(self, real_batch: torch.Tensor) -> float:
        """Train discriminator for one step"""
        
        self.d_optimizer.zero_grad()
        batch_size = real_batch.size(0)
        sequence_length = real_batch.size(1)
        
        # Train on real data
        real_output = self.discriminator(real_batch)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        real_loss = nn.BCELoss()(real_output, real_labels)
        
        # Train on fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_outputs = self.generator(noise, sequence_length)
        fake_sequences = torch.argmax(fake_outputs['payload_logits'], dim=-1)
        fake_output = self.discriminator(fake_sequences.detach())
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        fake_loss = nn.BCELoss()(fake_output, fake_labels)
        
        # Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
        
    def _train_generator(self, batch_size: int, sequence_length: int) -> float:
        """Train generator for one step"""
        
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_outputs = self.generator(noise, sequence_length)
        fake_sequences = torch.argmax(fake_outputs['payload_logits'], dim=-1)
        fake_output = self.discriminator(fake_sequences)
        
        # Generator wants discriminator to think fake is real
        real_labels = torch.ones(batch_size, 1, device=self.device)
        g_loss = nn.BCELoss()(fake_output, real_labels)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
        
    def _build_web_attack_vocab(self) -> List[str]:
        """Build vocabulary for web attack payloads"""
        
        vocab = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # SQL injection tokens
        sql_tokens = [
            'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'UNION', 'INSERT', 'UPDATE', 'DELETE',
            'DROP', 'CREATE', 'ALTER', 'TABLE', 'DATABASE', 'NULL', 'TRUE', 'FALSE',
            '--', '/*', '*/', "'", '"', ';', '=', '!=', '<', '>', '<=', '>=',
            'admin', 'password', 'user', 'login', 'users', 'information_schema'
        ]
        
        # XSS tokens
        xss_tokens = [
            '<script>', '</script>', '<img>', '<iframe>', '<object>', '<embed>',
            'alert', 'prompt', 'confirm', 'document', 'cookie', 'window', 'location',
            'onload', 'onerror', 'onclick', 'onmouseover', 'javascript:', 'eval',
            'innerHTML', 'outerHTML', 'appendChild', 'removeChild'
        ]
        
        # General web tokens
        web_tokens = [
            'http://', 'https://', 'file://', 'ftp://', '../', './', '..\\', '.\\',
            'admin', 'administrator', 'root', 'test', 'guest', 'anonymous',
            'passwd', 'shadow', 'etc', 'var', 'tmp', 'home', 'usr', 'bin',
            'GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'TRACE',
            'Content-Type', 'User-Agent', 'Cookie', 'Authorization', 'X-Forwarded-For'
        ]
        
        # Command injection tokens
        cmd_tokens = [
            '|', '&', '&&', '||', ';', '\n', '\r', '`', '$', '${', '}',
            'cat', 'ls', 'dir', 'type', 'echo', 'ping', 'wget', 'curl',
            'nc', 'netcat', 'bash', 'sh', 'cmd', 'powershell', 'python'
        ]
        
        # Encoding and evasion tokens
        evasion_tokens = [
            '%20', '%27', '%22', '%3C', '%3E', '%2F', '%5C', '%00', '%0A', '%0D',
            'char', 'concat', 'substring', 'ascii', 'hex', 'base64', 'url',
            'unicode', 'utf-8', 'utf-16', 'iso-8859-1'
        ]
        
        # Combine all tokens
        vocab.extend(sql_tokens)
        vocab.extend(xss_tokens)
        vocab.extend(web_tokens)
        vocab.extend(cmd_tokens)
        vocab.extend(evasion_tokens)
        
        # Add numbers and common characters
        vocab.extend([str(i) for i in range(10)])
        vocab.extend(list(string.ascii_lowercase))
        vocab.extend(list(string.ascii_uppercase))
        vocab.extend(['(', ')', '[', ']', '{', '}', '!', '@', '#', '$', '%', '^', '&', '*'])
        
        return list(set(vocab))  # Remove duplicates
        
    def _encode_payload(self, payload: str) -> List[int]:
        """Encode payload string to token IDs"""
        
        # Simple tokenization (in practice would use more sophisticated tokenizer)
        tokens = self._tokenize_payload(payload)
        
        encoded = [self.token_to_id.get('<START>', 1)]
        
        for token in tokens:
            token_id = self.token_to_id.get(token, self.token_to_id.get('<UNK>', 3))
            encoded.append(token_id)
            
        encoded.append(self.token_to_id.get('<END>', 2))
        
        return encoded
        
    def _tokenize_payload(self, payload: str) -> List[str]:
        """Tokenize payload string"""
        
        # Handle URL encoding
        payload = urllib.parse.unquote(payload)
        
        tokens = []
        current_token = ""
        
        for char in payload:
            if char in " \t\n\r":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in "()[]{}|&;<>='\"":
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
                
        if current_token:
            tokens.append(current_token)
            
        return tokens
        
    def _decode_payload(self, token_ids: List[int]) -> str:
        """Decode token IDs back to payload string"""
        
        tokens = []
        for token_id in token_ids:
            if token_id in [0, 1, 2]:  # PAD, START, END
                continue
            token = self.id_to_token.get(token_id, '<UNK>')
            if token != '<UNK>':
                tokens.append(token)
                
        return ' '.join(tokens)
        
    def _create_attack_session(
        self,
        payload_logits: torch.Tensor,
        attack_type_logits: torch.Tensor,
        sophistication_features: torch.Tensor,
        attack_types: List[str]
    ) -> WebAttackSession:
        """Create WebAttackSession from generated features"""
        
        # Decode payload
        payload_ids = torch.argmax(payload_logits, dim=-1).cpu().tolist()
        payload_string = self._decode_payload(payload_ids)
        
        # Determine attack type
        attack_type_idx = torch.argmax(attack_type_logits).item()
        attack_type = attack_types[attack_type_idx % len(attack_types)]
        
        # Determine sophistication
        sophistication_score = torch.mean(torch.abs(sophistication_features)).item()
        if sophistication_score > 0.7:
            sophistication_level = 'advanced'
        elif sophistication_score > 0.4:
            sophistication_level = 'intermediate'
        else:
            sophistication_level = 'basic'
            
        # Create payload object
        payload = WebAttackPayload(
            payload_id=f"payload_{random.randint(100000, 999999)}",
            attack_type=attack_type,
            payload_data=payload_string,
            target_parameter=self._select_target_parameter(attack_type),
            http_method=self._select_http_method(attack_type),
            success_probability=random.uniform(0.3, 0.9),
            evasion_techniques=self._select_evasion_techniques(sophistication_level),
            encoding_type=self._select_encoding_type(sophistication_level)
        )
        
        # Create session
        session = WebAttackSession(
            session_id=f"session_{random.randint(100000, 999999)}",
            attack_type=attack_type,
            payloads=[payload],
            target_url=self._generate_target_url(attack_type),
            user_agent=self._generate_user_agent(sophistication_level),
            attack_vector=self._determine_attack_vector(attack_type),
            sophistication_level=sophistication_level,
            duration=random.randint(30, 3600)
        )
        
        return session
        
    def _select_target_parameter(self, attack_type: str) -> str:
        """Select appropriate target parameter for attack type"""
        
        parameters = {
            'sql_injection': ['id', 'username', 'search', 'category', 'page'],
            'xss': ['comment', 'message', 'name', 'search', 'feedback'],
            'csrf': ['token', 'action', 'submit', 'form_id'],
            'lfi': ['file', 'page', 'include', 'template', 'lang'],
            'cmd_injection': ['cmd', 'command', 'exec', 'system', 'ping']
        }
        
        param_list = parameters.get(attack_type, ['param'])
        return random.choice(param_list)
        
    def _select_http_method(self, attack_type: str) -> str:
        """Select HTTP method for attack type"""
        
        method_prefs = {
            'sql_injection': 'GET',
            'xss': 'POST',
            'csrf': 'POST',
            'lfi': 'GET',
            'cmd_injection': 'POST'
        }
        
        return method_prefs.get(attack_type, 'GET')
        
    def _select_evasion_techniques(self, sophistication_level: str) -> List[str]:
        """Select evasion techniques based on sophistication"""
        
        techniques = {
            'basic': ['url_encoding'],
            'intermediate': ['url_encoding', 'case_variation', 'comment_insertion'],
            'advanced': ['url_encoding', 'case_variation', 'comment_insertion', 'unicode_encoding', 'double_encoding']
        }
        
        return techniques.get(sophistication_level, ['url_encoding'])
        
    def _select_encoding_type(self, sophistication_level: str) -> str:
        """Select encoding type based on sophistication"""
        
        encodings = {
            'basic': 'none',
            'intermediate': 'url',
            'advanced': random.choice(['url', 'base64', 'unicode', 'hex'])
        }
        
        return encodings.get(sophistication_level, 'none')
        
    def _generate_target_url(self, attack_type: str) -> str:
        """Generate realistic target URL"""
        
        domains = ['example.com', 'testsite.org', 'vulnerable-app.net', 'demo-server.io']
        domain = random.choice(domains)
        
        paths = {
            'sql_injection': ['/search.php', '/products.php', '/user.php', '/admin/login.php'],
            'xss': ['/comment.php', '/forum.php', '/feedback.php', '/contact.php'],
            'csrf': ['/transfer.php', '/account.php', '/settings.php', '/admin.php'],
            'lfi': ['/include.php', '/page.php', '/view.php', '/download.php'],
            'cmd_injection': ['/ping.php', '/admin/system.php', '/tools/network.php']
        }
        
        path_list = paths.get(attack_type, ['/index.php'])
        path = random.choice(path_list)
        
        return f"http://{domain}{path}"
        
    def _generate_user_agent(self, sophistication_level: str) -> str:
        """Generate user agent based on sophistication"""
        
        user_agents = {
            'basic': 'Mozilla/5.0 (compatible; TestBot/1.0)',
            'intermediate': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'advanced': random.choice([
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            ])
        }
        
        return user_agents.get(sophistication_level, 'Mozilla/5.0 (compatible; AttackBot/1.0)')
        
    def _determine_attack_vector(self, attack_type: str) -> str:
        """Determine attack vector"""
        
        vectors = {
            'sql_injection': 'parameter_injection',
            'xss': 'form_injection',
            'csrf': 'form_submission',
            'lfi': 'file_inclusion',
            'cmd_injection': 'command_execution'
        }
        
        return vectors.get(attack_type, 'unknown')
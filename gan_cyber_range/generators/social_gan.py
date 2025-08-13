"""
GAN-based social engineering attack generation.

This module generates realistic social engineering campaigns including phishing emails,
social media manipulation, and human-targeted attack scenarios for security awareness training.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import string
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class SocialEngineeringProfile:
    """Profile for social engineering target"""
    target_id: str
    name: str
    role: str
    department: str
    seniority_level: str
    access_level: str
    susceptibility_score: float
    interests: List[str]
    social_media_presence: str
    communication_style: str


@dataclass
class PhishingEmail:
    """Represents a phishing email"""
    email_id: str
    subject: str
    sender: str
    sender_name: str
    body: str
    attachments: List[str]
    urgency_level: str
    pretext: str
    target_emotions: List[str]
    social_proof_elements: List[str]


@dataclass
class SocialEngineeringCampaign:
    """Complete social engineering campaign"""
    campaign_id: str
    campaign_type: str
    target_profiles: List[SocialEngineeringProfile]
    attack_vectors: List[str]
    emails: List[PhishingEmail]
    timeline: Dict[str, str]
    success_metrics: List[str]
    sophistication_level: str


class SocialEngineeringGenerator(nn.Module):
    """GAN generator for social engineering content"""
    
    def __init__(
        self,
        noise_dim: int = 128,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Text generation components
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Subject line generator
        self.subject_lstm = nn.LSTM(
            input_size=embedding_dim + noise_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Email body generator  
        self.body_lstm = nn.LSTM(
            input_size=embedding_dim + noise_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        # Output projections
        self.subject_output = nn.Linear(hidden_dim, vocab_size)
        self.body_output = nn.Linear(hidden_dim, vocab_size)
        
        # Metadata generators
        self.urgency_head = nn.Linear(hidden_dim, 4)  # low, medium, high, critical
        self.emotion_head = nn.Linear(hidden_dim, 8)  # fear, greed, curiosity, etc.
        self.pretext_head = nn.Linear(hidden_dim, 10) # IT support, CEO, etc.
        
    def forward(
        self,
        noise: torch.Tensor,
        subject_length: int = 10,
        body_length: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Generate social engineering email components"""
        
        batch_size = noise.size(0)
        device = noise.device
        
        # Generate subject line
        subject_tokens = self._generate_sequence(
            noise, self.subject_lstm, self.subject_output, subject_length
        )
        
        # Generate email body
        body_tokens = self._generate_sequence(
            noise, self.body_lstm, self.body_output, body_length
        )
        
        # Generate metadata from final hidden states
        _, (subject_hidden, _) = self.subject_lstm(
            torch.cat([
                self.embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=device)),
                noise.unsqueeze(1).expand(-1, 1, -1)
            ], dim=2)
        )
        
        urgency_logits = self.urgency_head(subject_hidden[-1])
        emotion_logits = self.emotion_head(subject_hidden[-1])
        pretext_logits = self.pretext_head(subject_hidden[-1])
        
        return {
            'subject_tokens': subject_tokens,
            'body_tokens': body_tokens,
            'urgency_logits': urgency_logits,
            'emotion_logits': emotion_logits,
            'pretext_logits': pretext_logits
        }
        
    def _generate_sequence(
        self,
        noise: torch.Tensor,
        lstm: nn.LSTM,
        output_layer: nn.Linear,
        sequence_length: int
    ) -> torch.Tensor:
        """Generate token sequence using LSTM"""
        
        batch_size = noise.size(0)
        device = noise.device
        
        # Initialize hidden state
        num_layers = lstm.num_layers
        hidden_size = lstm.hidden_size
        h0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        
        # Start token
        current_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(sequence_length):
            # Embed current token
            token_embed = self.embedding(current_token)
            
            # Add noise
            noise_expanded = noise.unsqueeze(1).expand(-1, 1, -1)
            lstm_input = torch.cat([token_embed, noise_expanded], dim=2)
            
            # LSTM forward
            lstm_out, hidden = lstm(lstm_input, hidden)
            
            # Project to vocabulary
            logits = output_layer(lstm_out.squeeze(1))
            outputs.append(logits)
            
            # Sample next token
            if self.training:
                probs = torch.softmax(logits, dim=-1)
                current_token = torch.multinomial(probs, 1)
            else:
                current_token = torch.argmax(logits, dim=-1, keepdim=True)
                
        return torch.stack(outputs, dim=1)


class SocialEngineeringDiscriminator(nn.Module):
    """Discriminator for social engineering GAN"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Separate encoders for subject and body
        self.subject_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.body_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # Combined subject + body features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        subject_tokens: torch.Tensor,
        body_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Classify social engineering content as real or synthetic"""
        
        # Embed sequences
        subject_embedded = self.embedding(subject_tokens)
        body_embedded = self.embedding(body_tokens)
        
        # Encode sequences
        _, (subject_hidden, _) = self.subject_encoder(subject_embedded)
        _, (body_hidden, _) = self.body_encoder(body_embedded)
        
        # Combine hidden states
        subject_features = torch.cat([subject_hidden[-2], subject_hidden[-1]], dim=1)
        body_features = torch.cat([body_hidden[-2], body_hidden[-1]], dim=1)
        
        combined_features = torch.cat([subject_features, body_features], dim=1)
        
        # Classify
        output = self.classifier(combined_features)
        
        return output


class SocialEngineeringGAN:
    """Complete GAN system for social engineering generation"""
    
    def __init__(
        self,
        vocab_size: int = 10000,
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
            
        # Build vocabulary
        self.vocab = self._build_social_engineering_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        
        # Initialize networks
        self.generator = SocialEngineeringGenerator(
            noise_dim=noise_dim,
            vocab_size=len(self.vocab)
        ).to(self.device)
        
        self.discriminator = SocialEngineeringDiscriminator(
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
            'campaign_diversity': []
        }
        
        logger.info(f"Initialized SocialEngineeringGAN on device: {self.device}")
        
    def generate_phishing_campaigns(
        self,
        num_campaigns: int = 50,
        target_profiles: List[SocialEngineeringProfile] = None
    ) -> List[SocialEngineeringCampaign]:
        """Generate synthetic phishing campaigns"""
        
        if target_profiles is None:
            target_profiles = self._generate_default_targets()
            
        logger.info(f"Generating {num_campaigns} phishing campaigns")
        
        self.generator.eval()
        campaigns = []
        
        with torch.no_grad():
            batch_size = 16
            for i in range(0, num_campaigns, batch_size):
                current_batch_size = min(batch_size, num_campaigns - i)
                
                # Generate noise
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                
                # Generate email components
                outputs = self.generator(noise, subject_length=15, body_length=150)
                
                # Create campaigns
                for j in range(current_batch_size):
                    campaign = self._create_phishing_campaign(
                        outputs['subject_tokens'][j],
                        outputs['body_tokens'][j],
                        outputs['urgency_logits'][j],
                        outputs['emotion_logits'][j],
                        outputs['pretext_logits'][j],
                        target_profiles
                    )
                    campaigns.append(campaign)
                    
        logger.info(f"Generated {len(campaigns)} phishing campaigns")
        return campaigns
        
    def _create_phishing_campaign(
        self,
        subject_tokens: torch.Tensor,
        body_tokens: torch.Tensor,
        urgency_logits: torch.Tensor,
        emotion_logits: torch.Tensor,
        pretext_logits: torch.Tensor,
        target_profiles: List[SocialEngineeringProfile]
    ) -> SocialEngineeringCampaign:
        """Create phishing campaign from generated components"""
        
        # Decode text
        subject = self._decode_tokens(subject_tokens)
        body = self._decode_tokens(body_tokens)
        
        # Decode metadata
        urgency_levels = ['low', 'medium', 'high', 'critical']
        urgency = urgency_levels[torch.argmax(urgency_logits).item()]
        
        emotions = ['fear', 'greed', 'curiosity', 'authority', 'urgency', 'trust', 'reciprocity', 'scarcity']
        target_emotion = emotions[torch.argmax(emotion_logits).item() % len(emotions)]
        
        pretexts = ['it_support', 'ceo_request', 'hr_notice', 'security_alert', 'invoice', 
                   'delivery', 'bank_notice', 'social_media', 'job_offer', 'survey']
        pretext = pretexts[torch.argmax(pretext_logits).item()]
        
        # Generate sender based on pretext
        sender_info = self._generate_sender_info(pretext)
        
        # Create phishing email
        email = PhishingEmail(
            email_id=f"phish_{random.randint(100000, 999999)}",
            subject=subject,
            sender=sender_info['email'],
            sender_name=sender_info['name'],
            body=body,
            attachments=self._generate_attachments(pretext),
            urgency_level=urgency,
            pretext=pretext,
            target_emotions=[target_emotion],
            social_proof_elements=self._generate_social_proof(pretext)
        )
        
        # Select target profiles
        selected_targets = random.sample(
            target_profiles, 
            min(random.randint(1, 10), len(target_profiles))
        )
        
        # Create campaign
        campaign = SocialEngineeringCampaign(
            campaign_id=f"campaign_{random.randint(100000, 999999)}",
            campaign_type='spear_phishing',
            target_profiles=selected_targets,
            attack_vectors=['email'],
            emails=[email],
            timeline=self._generate_campaign_timeline(),
            success_metrics=['click_rate', 'credential_harvest', 'attachment_execution'],
            sophistication_level=self._determine_sophistication(urgency, pretext, target_emotion)
        )
        
        return campaign
        
    def _build_social_engineering_vocab(self) -> List[str]:
        """Build vocabulary for social engineering content"""
        
        vocab = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # Common words in phishing emails
        phishing_words = [
            'urgent', 'immediate', 'action', 'required', 'verify', 'account', 'suspended',
            'click', 'here', 'login', 'password', 'security', 'update', 'confirm',
            'expired', 'invoice', 'payment', 'overdue', 'refund', 'winner', 'congratulations',
            'limited', 'time', 'offer', 'exclusive', 'free', 'prize', 'claim'
        ]
        
        # Business/professional terms
        business_words = [
            'dear', 'regards', 'sincerely', 'team', 'department', 'manager', 'director',
            'ceo', 'admin', 'support', 'service', 'customer', 'client', 'user',
            'meeting', 'conference', 'document', 'report', 'policy', 'procedure',
            'compliance', 'audit', 'review', 'approval', 'authorization'
        ]
        
        # Technical terms
        tech_words = [
            'system', 'server', 'database', 'backup', 'maintenance', 'upgrade',
            'software', 'application', 'platform', 'network', 'connection',
            'firewall', 'antivirus', 'malware', 'virus', 'spam', 'phishing'
        ]
        
        # Emotional triggers
        emotion_words = [
            'important', 'critical', 'serious', 'warning', 'alert', 'notice',
            'problem', 'issue', 'error', 'failure', 'breach', 'violation',
            'deadline', 'expires', 'final', 'last', 'chance', 'opportunity'
        ]
        
        # Common English words
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
            'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up'
        ]
        
        # Combine all vocabularies
        vocab.extend(phishing_words)
        vocab.extend(business_words)
        vocab.extend(tech_words)
        vocab.extend(emotion_words)
        vocab.extend(common_words)
        
        # Add punctuation and special characters
        vocab.extend(['.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']'])
        
        return list(set(vocab))  # Remove duplicates
        
    def _decode_tokens(self, token_tensor: torch.Tensor) -> str:
        """Decode token tensor to string"""
        
        token_ids = torch.argmax(token_tensor, dim=-1).cpu().tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in [0, 1, 2]:  # PAD, START, END
                continue
            token = self.id_to_token.get(token_id, '<UNK>')
            if token != '<UNK>':
                tokens.append(token)
                
        return ' '.join(tokens)
        
    def _generate_sender_info(self, pretext: str) -> Dict[str, str]:
        """Generate sender information based on pretext"""
        
        sender_templates = {
            'it_support': {
                'name': random.choice(['IT Support', 'Tech Team', 'System Admin']),
                'email': random.choice(['support@company.com', 'it-help@company.com', 'admin@company.com'])
            },
            'ceo_request': {
                'name': random.choice(['John Smith', 'Sarah Johnson', 'Michael Brown']),
                'email': random.choice(['ceo@company.com', 'executive@company.com'])
            },
            'hr_notice': {
                'name': random.choice(['HR Department', 'Human Resources', 'HR Team']),
                'email': random.choice(['hr@company.com', 'humanresources@company.com'])
            },
            'security_alert': {
                'name': random.choice(['Security Team', 'InfoSec', 'Cyber Security']),
                'email': random.choice(['security@company.com', 'infosec@company.com'])
            },
            'bank_notice': {
                'name': random.choice(['Customer Service', 'Account Services', 'Security Team']),
                'email': random.choice(['noreply@bank.com', 'security@bank.com'])
            }
        }
        
        return sender_templates.get(pretext, {
            'name': 'Customer Service',
            'email': 'noreply@company.com'
        })
        
    def _generate_attachments(self, pretext: str) -> List[str]:
        """Generate relevant attachments based on pretext"""
        
        attachment_templates = {
            'invoice': ['invoice.pdf', 'bill.pdf', 'statement.xlsx'],
            'hr_notice': ['policy.pdf', 'handbook.doc', 'form.pdf'],
            'it_support': ['patch.exe', 'update.zip', 'instructions.pdf'],
            'security_alert': ['report.pdf', 'scan_results.txt', 'action_required.doc'],
            'job_offer': ['contract.pdf', 'job_description.doc', 'benefits.pdf']
        }
        
        attachments = attachment_templates.get(pretext, [])
        
        if attachments and random.random() < 0.6:  # 60% chance of attachment
            return [random.choice(attachments)]
        
        return []
        
    def _generate_social_proof(self, pretext: str) -> List[str]:
        """Generate social proof elements"""
        
        social_proof = {
            'ceo_request': ['executive_signature', 'company_logo'],
            'bank_notice': ['bank_logo', 'security_badges', 'contact_info'],
            'it_support': ['company_branding', 'helpdesk_info'],
            'security_alert': ['security_badges', 'incident_number'],
            'hr_notice': ['hr_signature', 'company_policy_reference']
        }
        
        return social_proof.get(pretext, ['generic_branding'])
        
    def _generate_campaign_timeline(self) -> Dict[str, str]:
        """Generate campaign timeline"""
        
        return {
            'preparation': f"{random.randint(1, 7)} days",
            'execution': f"{random.randint(1, 3)} days",
            'follow_up': f"{random.randint(1, 14)} days",
            'analysis': f"{random.randint(1, 3)} days"
        }
        
    def _determine_sophistication(self, urgency: str, pretext: str, emotion: str) -> str:
        """Determine campaign sophistication level"""
        
        sophistication_score = 0
        
        # Urgency scoring
        if urgency in ['high', 'critical']:
            sophistication_score += 1
            
        # Pretext scoring
        if pretext in ['ceo_request', 'security_alert']:
            sophistication_score += 2
        elif pretext in ['hr_notice', 'it_support']:
            sophistication_score += 1
            
        # Emotion scoring
        if emotion in ['authority', 'trust']:
            sophistication_score += 1
            
        if sophistication_score >= 3:
            return 'advanced'
        elif sophistication_score >= 2:
            return 'intermediate'
        else:
            return 'basic'
            
    def _generate_default_targets(self) -> List[SocialEngineeringProfile]:
        """Generate default target profiles"""
        
        roles = ['analyst', 'manager', 'director', 'admin', 'developer', 'intern', 'executive']
        departments = ['finance', 'hr', 'it', 'sales', 'marketing', 'operations', 'legal']
        seniority_levels = ['junior', 'mid', 'senior', 'executive']
        
        targets = []
        
        for i in range(20):
            target = SocialEngineeringProfile(
                target_id=f"target_{i:03d}",
                name=f"Employee {i+1}",
                role=random.choice(roles),
                department=random.choice(departments),
                seniority_level=random.choice(seniority_levels),
                access_level=random.choice(['standard', 'elevated', 'admin']),
                susceptibility_score=random.uniform(0.1, 0.9),
                interests=random.sample(['sports', 'technology', 'travel', 'food', 'music'], 2),
                social_media_presence=random.choice(['low', 'medium', 'high']),
                communication_style=random.choice(['formal', 'casual', 'brief', 'detailed'])
            )
            targets.append(target)
            
        return targets
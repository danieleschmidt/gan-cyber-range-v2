# Contributing to GAN-Cyber-Range-v2

Thank you for your interest in contributing to GAN-Cyber-Range-v2! This document provides guidelines for contributing to this cybersecurity research platform.

## Code of Conduct

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Getting Started

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/gan-cyber-range-v2.git
   cd gan-cyber-range-v2
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Setup**
   ```bash
   pytest tests/ -v
   python -c "import gan_cyber_range; print('Setup successful')"
   ```

### Project Structure

```
gan-cyber-range-v2/
├── gan_cyber_range/          # Main package
│   ├── core/                 # Core components (GAN, CyberRange)
│   ├── generators/           # Attack generators
│   ├── red_team/            # Red team LLM components
│   ├── blue_team/           # Blue team evaluation
│   ├── simulation/          # Network simulation
│   ├── analysis/            # Analysis and reporting
│   └── utils/               # Utilities (logging, caching, etc.)
├── tests/                   # Test suite
├── docs/                    # Documentation
├── examples/                # Usage examples
├── scripts/                 # Utility scripts
├── data/                    # Data files
└── configs/                 # Configuration files
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports** - Help us identify and fix issues
2. **Feature Requests** - Suggest new functionality
3. **Code Contributions** - Implement features or fix bugs
4. **Documentation** - Improve docs, examples, or tutorials
5. **Research Contributions** - Academic papers, datasets, benchmarks
6. **Security Reviews** - Help improve security posture

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Bug Report Template:**
```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.9.2]
 - Package Version: [e.g. 2.0.0]
 - GPU: [e.g. NVIDIA RTX 3080]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

**Feature Request Template:**
```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
A clear description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.

**Implementation Ideas**
If you have ideas about how this could be implemented, please share them.
```

### Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. **Make Changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new attack generation algorithm"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

### Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(gan): add Wasserstein GAN implementation
fix(range): resolve container deployment issue
docs(api): update API documentation for AttackGAN
test(utils): add comprehensive caching tests
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

1. **Line Length**: 100 characters (not 79)
2. **Imports**: Use absolute imports, group by standard/third-party/local
3. **Type Hints**: Required for all public functions and class methods
4. **Docstrings**: Google-style docstrings for all public APIs

**Example:**
```python
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AttackGenerator:
    """Generates synthetic cyber attacks for training purposes.
    
    This class implements various algorithms for creating realistic
    attack patterns that can be used to train defensive systems.
    
    Args:
        model_type: Type of generative model to use
        config: Configuration parameters for the generator
        
    Example:
        >>> generator = AttackGenerator("gan", config)
        >>> attacks = generator.generate(num_samples=100)
    """
    
    def __init__(self, model_type: str, config: Dict[str, any]) -> None:
        self.model_type = model_type
        self.config = config
        self.model = self._initialize_model()
    
    def generate(self, num_samples: int, **kwargs) -> List[Attack]:
        """Generate synthetic attacks.
        
        Args:
            num_samples: Number of attacks to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated Attack objects
            
        Raises:
            ValueError: If num_samples is not positive
            ModelNotTrainedError: If model hasn't been trained
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
            
        logger.info(f"Generating {num_samples} attacks using {self.model_type}")
        
        # Implementation here
        return attacks
    
    def _initialize_model(self) -> Model:
        """Initialize the generative model (private method)."""
        # Implementation here
        pass
```

### Code Quality Tools

We use several tools to maintain code quality:

1. **Black** - Code formatting
2. **isort** - Import sorting  
3. **flake8** - Linting
4. **mypy** - Type checking
5. **pytest** - Testing

**Configuration files:**
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pyproject.toml` - Tool configurations
- `pytest.ini` - Test configuration

### Testing Requirements

All contributions must include appropriate tests:

1. **Unit Tests** - Test individual functions/methods
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows
4. **Security Tests** - Test security aspects

**Test Structure:**
```python
import pytest
from unittest.mock import Mock, patch
from gan_cyber_range.core import AttackGAN


class TestAttackGAN:
    """Test suite for AttackGAN class."""
    
    @pytest.fixture
    def attack_gan(self):
        """Create AttackGAN instance for testing."""
        return AttackGAN(device="cpu")  # Force CPU for tests
    
    def test_initialization(self, attack_gan):
        """Test proper initialization."""
        assert attack_gan.device.type == "cpu"
        assert attack_gan.noise_dim == 100
    
    def test_train_with_valid_data(self, attack_gan):
        """Test training with valid attack data."""
        sample_data = ["sql injection", "xss attack", "malware"]
        
        history = attack_gan.train(sample_data, epochs=2, batch_size=2)
        
        assert isinstance(history, dict)
        assert 'g_loss' in history
        assert 'd_loss' in history
    
    @patch('gan_cyber_range.core.attack_gan.torch.cuda.is_available')
    def test_gpu_detection(self, mock_cuda_available, attack_gan):
        """Test GPU detection and fallback."""
        mock_cuda_available.return_value = False
        
        gan = AttackGAN(device="auto")
        assert gan.device.type == "cpu"
    
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with pytest.raises(ValueError):
            AttackGAN(noise_dim=-1)
```

**Test Coverage Requirements:**
- Minimum 85% code coverage
- All public APIs must be tested
- Critical paths must have multiple test cases
- Error conditions must be tested

### Security Considerations

Given the security-focused nature of this project, special attention must be paid to security:

1. **No Real Malware** - Never include actual malicious code
2. **Sandboxing** - All attack simulations must be properly contained
3. **Input Validation** - Validate all user inputs
4. **Audit Logging** - Log security-relevant actions
5. **Access Control** - Implement proper authorization
6. **Data Protection** - Protect sensitive training data

**Security Review Checklist:**
- [ ] No hardcoded secrets or credentials
- [ ] Proper input validation and sanitization
- [ ] Network isolation for attack simulations
- [ ] Audit logging for security events
- [ ] Safe handling of attack data and payloads
- [ ] Protection against directory traversal
- [ ] Rate limiting for API endpoints

## Documentation Standards

### API Documentation

Use Google-style docstrings for all public APIs:

```python
def train_gan(data: List[str], epochs: int = 1000, batch_size: int = 64) -> Dict[str, List[float]]:
    """Train the GAN on attack data.
    
    Trains a generative adversarial network to learn patterns from real
    attack data and generate synthetic attacks for defensive training.
    
    Args:
        data: List of attack strings or path to data file
        epochs: Number of training epochs (default: 1000)
        batch_size: Training batch size, must be power of 2 (default: 64)
        
    Returns:
        Dictionary containing training history with 'g_loss' and 'd_loss' keys,
        each mapping to a list of loss values per epoch.
        
    Raises:
        ValueError: If epochs <= 0 or batch_size is not a power of 2
        FileNotFoundError: If data path doesn't exist
        
    Example:
        >>> attack_data = ["sql injection", "xss payload", "malware sample"]
        >>> history = train_gan(attack_data, epochs=500, batch_size=32)
        >>> print(f"Final generator loss: {history['g_loss'][-1]:.4f}")
        Final generator loss: 0.1234
        
    Note:
        Training time scales with dataset size and number of epochs.
        GPU acceleration is recommended for large datasets.
    """
```

### User Guides

Create comprehensive user guides for new features:

1. **Overview** - What the feature does
2. **Installation** - How to set it up
3. **Basic Usage** - Simple examples
4. **Advanced Usage** - Complex scenarios
5. **Troubleshooting** - Common issues
6. **API Reference** - Complete API docs

### Research Documentation

For research contributions, provide:

1. **Methodology** - Research approach and design
2. **Datasets** - Data sources and preprocessing
3. **Experiments** - Experimental setup and results
4. **Reproducibility** - Complete reproduction instructions
5. **Citation** - How to cite the work

## Research Contributions

### Academic Standards

Research contributions should meet academic standards:

1. **Novelty** - Present new ideas or improvements
2. **Rigor** - Use proper experimental methodology
3. **Reproducibility** - Provide complete reproduction packages
4. **Evaluation** - Compare against established baselines
5. **Ethics** - Follow ethical research guidelines

### Datasets and Benchmarks

When contributing datasets or benchmarks:

1. **Documentation** - Comprehensive metadata and descriptions
2. **Licensing** - Clear license for usage and redistribution
3. **Validation** - Quality checks and validation procedures
4. **Baselines** - Reference implementations and results
5. **Maintenance** - Ongoing maintenance and updates

### Publication Process

For significant research contributions:

1. **Pre-submission Review** - Internal review before external submission
2. **Community Feedback** - Seek feedback from the community
3. **Peer Review** - Submit to appropriate venues
4. **Open Access** - Make publications openly available when possible

## Community

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General discussion and questions
- **Discord Server** - Real-time chat and collaboration
- **Mailing List** - Announcements and important updates

### Mentorship Program

We offer mentorship for new contributors:

- **Getting Started** - Help with initial setup and first contributions
- **Code Review** - Detailed feedback on code quality and style
- **Research Guidance** - Support for research contributions
- **Career Development** - Advice on cybersecurity research careers

### Recognition

We recognize valuable contributions through:

- **Contributor List** - Recognition in project documentation
- **Research Citations** - Co-authorship on academic papers
- **Conference Presentations** - Opportunities to present work
- **Recommendation Letters** - Support for academic and career goals

## Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

### Release Checklist

Before creating a release:

1. [ ] All tests pass
2. [ ] Documentation is updated
3. [ ] CHANGELOG.md is updated
4. [ ] Version numbers are bumped
5. [ ] Security review is completed
6. [ ] Performance benchmarks are run
7. [ ] Backwards compatibility is verified

### Beta Testing

Major releases go through beta testing:

1. **Alpha Release** - Internal testing by core team
2. **Beta Release** - External testing by community
3. **Release Candidate** - Final testing before stable release
4. **Stable Release** - General availability

## Legal and Licensing

### License Agreement

By contributing, you agree that your contributions will be licensed under the MIT License.

### Copyright

- Retain copyright for substantial contributions
- Sign Contributor License Agreement (CLA) for significant changes
- Ensure all dependencies are compatible with MIT license

### Export Controls

This project may be subject to export controls. Contributors should be aware of:

- **ITAR** - International Traffic in Arms Regulations
- **EAR** - Export Administration Regulations
- **Country Restrictions** - Embargoed countries and entities

### Responsible Disclosure

For security vulnerabilities:

1. **Do NOT** create public issues for security vulnerabilities
2. **Email** security@terragonlabs.com with details
3. **Wait** for response before public disclosure
4. **Follow** coordinated disclosure timeline

## Getting Help

If you need help with contributing:

1. **Read Documentation** - Check existing docs first
2. **Search Issues** - Look for similar questions
3. **Ask on Discord** - Real-time help from community
4. **Create Discussion** - For general questions
5. **Create Issue** - For specific problems

## Thank You!

Thank you for taking the time to contribute to GAN-Cyber-Range-v2! Your contributions help advance cybersecurity research and education for everyone.

---

*This document is adapted from open source contribution guidelines and follows best practices for cybersecurity projects.*
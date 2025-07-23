# Contributing to Alpaca Improved

Thank you for your interest in contributing to Alpaca Improved! This document provides guidelines and information for contributors to ensure a smooth collaboration process.

## 🎯 Project Vision

Alpaca Improved aims to be the premier options trading platform built on Alpaca's ecosystem, providing institutional-grade tools for strategy development, backtesting, and deployment. We value:

- **Code Quality**: Clean, maintainable, and well-documented code
- **Testing**: Comprehensive test coverage and validation
- **Performance**: Efficient algorithms and optimized implementations
- **Security**: Safe trading practices and secure code
- **Collaboration**: Open communication and knowledge sharing

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Security Guidelines](#security-guidelines)
- [Release Process](#release-process)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (3.9+ recommended)
- Poetry for dependency management
- Git for version control
- Alpaca paper trading account (for testing)

### Setup Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/alpaca_improved.git
   cd alpaca_improved
   ```

2. **Install Dependencies**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install project dependencies
   poetry install --with dev
   
   # Activate virtual environment
   poetry shell
   ```

3. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your Alpaca paper trading credentials
   # NEVER use live trading credentials in development!
   ```

4. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   pytest tests/unit/ -v
   
   # Run linting
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/
   ```

## 🔄 Development Workflow

### Branching Strategy

We follow the **GitFlow** branching model:

- `main`: Production-ready code, protected branch
- `develop`: Integration branch for features
- `feature/*`: New features and enhancements
- `hotfix/*`: Critical fixes for production
- `release/*`: Release preparation branches

### Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards (see below)
   - Write tests for new functionality
   - Update documentation as needed
   - Commit frequently with clear messages

3. **Test Your Changes**
   ```bash
   # Run full test suite
   pytest
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   
   # Check code quality
   make lint
   make format
   ```

4. **Submit Pull Request**
   - Push your feature branch
   - Create pull request against `develop`
   - Fill out PR template completely
   - Request reviews from maintainers

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(strategies): add momentum trading strategy with RSI signals

fix(data): resolve issue with options chain data caching

docs(api): update backtesting engine documentation

test(strategies): add comprehensive tests for base strategy class
```

## 📏 Coding Standards

### Python Code Style

We follow **PEP 8** with these specific guidelines:

1. **Formatting**
   - Use **Black** for code formatting (88 character line length)
   - Use **isort** for import sorting
   - Follow existing code patterns

2. **Type Hints**
   - All functions must include type hints
   - Use `from typing import` for complex types
   - Example:
     ```python
     def calculate_sharpe_ratio(
         returns: pd.Series,
         risk_free_rate: float = 0.02
     ) -> float:
         """Calculate Sharpe ratio for return series."""
         pass
     ```

3. **Documentation**
   - Use **Google-style docstrings**
   - Document all public functions, classes, and modules
   - Include examples for complex functions
   - Example:
     ```python
     def backtest_strategy(
         strategy: BaseStrategy,
         start_date: datetime,
         end_date: datetime
     ) -> BacktestResults:
         """
         Run comprehensive backtest for a trading strategy.
         
         Args:
             strategy: Strategy instance to backtest
             start_date: Start date for backtesting
             end_date: End date for backtesting
             
         Returns:
             BacktestResults object with performance metrics
             
         Raises:
             ValueError: If start_date is after end_date
             
         Example:
             >>> strategy = MomentumStrategy(config)
             >>> results = backtest_strategy(
             ...     strategy,
             ...     datetime(2023, 1, 1),
             ...     datetime(2023, 12, 31)
             ... )
             >>> print(f"Sharpe ratio: {results.sharpe_ratio:.2f}")
         """
         pass
     ```

4. **Error Handling**
   - Use specific exception types
   - Log errors appropriately
   - Provide meaningful error messages
   - Clean up resources in finally blocks

5. **Imports**
   - Group imports: standard library, third-party, local
   - Use absolute imports for project modules
   - Avoid wildcard imports (`from module import *`)

### Code Organization

1. **Module Structure**
   ```
   src/alpaca_improved/
   ├── __init__.py          # Package initialization
   ├── config.py            # Configuration management
   ├── strategies/          # Trading strategies
   ├── backtesting/         # Backtesting engines
   ├── data/                # Data management
   ├── trading/             # Live trading
   ├── utils/               # Utility functions
   └── bots/                # Trading bots
   ```

2. **Class Design**
   - Follow SOLID principles
   - Use dependency injection
   - Implement abstract base classes for extensibility
   - Prefer composition over inheritance

3. **Configuration**
   - Use environment variables for sensitive data
   - Centralize configuration in `config.py`
   - Validate configuration on startup

## 🧪 Testing Requirements

### Test Coverage Standards

- **Minimum 80% code coverage** for all modules
- **90%+ coverage** for critical trading and risk management code
- All public functions must have tests
- Edge cases and error conditions must be tested

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes in isolation
   - Use mocks for external dependencies
   - Fast execution (< 1 second per test)
   - Example:
     ```python
     def test_calculate_sharpe_ratio():
         returns = pd.Series([0.01, -0.02, 0.03, 0.01])
         result = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
         assert isinstance(result, float)
         assert result > 0  # Assuming positive Sharpe ratio
     ```

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Use test databases and mock APIs
   - Verify data flows and transformations

3. **End-to-End Tests** (`tests/e2e/`)
   - Test complete workflows
   - Use paper trading accounts
   - Run in CI/CD pipeline only

4. **Performance Tests** (`tests/performance/`)
   - Benchmark critical functions
   - Memory usage validation
   - Scalability testing

### Testing Best Practices

1. **Arrange-Act-Assert Pattern**
   ```python
   def test_strategy_initialization():
       # Arrange
       config = StrategyConfig(name="test", max_positions=5)
       
       # Act
       strategy = MomentumStrategy(config)
       
       # Assert
       assert strategy.config.name == "test"
       assert strategy.state == StrategyState.INITIALIZED
   ```

2. **Descriptive Test Names**
   ```python
   def test_backtest_returns_error_when_start_date_after_end_date():
       pass
   
   def test_options_strategy_validates_expiration_date_range():
       pass
   ```

3. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   def sample_market_data():
       return pd.DataFrame({
           'open': [100, 101, 102],
           'high': [101, 103, 104],
           'low': [99, 100, 101],
           'close': [100.5, 102, 103],
           'volume': [1000, 1200, 1100]
       })
   ```

4. **Mock External Dependencies**
   ```python
   @patch('alpaca_improved.data.AlpacaDataClient')
   def test_data_extraction_handles_api_errors(mock_client):
       mock_client.get_bars.side_effect = APIError("Rate limit exceeded")
       # Test error handling
   ```

## 📚 Documentation

### Documentation Standards

1. **Code Documentation**
   - Comprehensive docstrings for all public APIs
   - Inline comments for complex logic
   - Type hints for all function signatures

2. **API Documentation**
   - Auto-generated from docstrings using Sphinx
   - Include examples and use cases
   - Document error conditions and exceptions

3. **User Guides**
   - Step-by-step tutorials
   - Best practices and common patterns
   - Troubleshooting guides

4. **Architecture Documentation**
   - High-level system design
   - Component interactions
   - Design decisions and rationale

### Documentation Updates

- Update docs when adding new features
- Review and update existing docs during refactoring
- Include code examples that are tested
- Keep README and installation guides current

## 🔀 Pull Request Process

### Before Submitting

1. **Code Quality Checklist**
   - [ ] Code follows style guidelines
   - [ ] All tests pass
   - [ ] Coverage targets met
   - [ ] Documentation updated
   - [ ] No security vulnerabilities

2. **Functional Testing**
   - [ ] Feature works as intended
   - [ ] Edge cases handled
   - [ ] Error conditions tested
   - [ ] Performance acceptable

### PR Template

When creating a pull request, include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline must pass
   - Code quality gates satisfied
   - Security scans clear

2. **Peer Review**
   - At least 2 approvals required
   - Domain expert review for complex changes
   - Address all feedback before merge

3. **Final Validation**
   - Manual testing in staging environment
   - Performance impact assessment
   - Documentation review

## 🐛 Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, package versions
- **Steps to Reproduce**: Detailed steps to trigger the bug
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full error logs and stack traces
- **Additional Context**: Screenshots, logs, configuration

### Feature Requests

Use the feature request template and include:

- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: Detailed description of the feature
- **Alternatives Considered**: Other approaches evaluated
- **Additional Context**: Use cases, examples, mockups

### Security Issues

**Do not report security issues publicly!**

Send security-related issues to: security@alpacaimproved.com

Include:
- Detailed vulnerability description
- Steps to reproduce
- Potential impact assessment
- Suggested fixes (if any)

## 🔒 Security Guidelines

### Development Security

1. **Never Commit Secrets**
   - Use environment variables
   - Add sensitive files to `.gitignore`
   - Scan commits for secrets

2. **API Key Management**
   - Use paper trading accounts only
   - Rotate keys regularly
   - Limit API key permissions

3. **Code Security**
   - Validate all inputs
   - Use parameterized queries
   - Follow OWASP guidelines

4. **Dependency Security**
   - Keep dependencies updated
   - Use `safety` to check for vulnerabilities
   - Pin dependency versions

### Trading Security

1. **Risk Management**
   - Implement position limits
   - Use stop-loss orders
   - Monitor portfolio exposure

2. **Data Validation**
   - Validate market data
   - Check order parameters
   - Implement circuit breakers

## 📦 Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow

1. **Pre-Release Testing**
   - Full test suite execution
   - Performance benchmarks
   - Security scans
   - Manual testing

2. **Release Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b release/v1.2.0
   ```

3. **Release Preparation**
   - Update version numbers
   - Update CHANGELOG.md
   - Final documentation review
   - Create release notes

4. **Release Execution**
   - Merge to main branch
   - Create Git tag
   - Deploy to PyPI
   - Update documentation site

5. **Post-Release**
   - Merge back to develop
   - Close GitHub milestone
   - Announce release

## 🤝 Community Guidelines

### Communication

- **Be Respectful**: Treat all community members with respect
- **Be Constructive**: Provide helpful, actionable feedback
- **Be Patient**: Understand that maintainers are volunteers
- **Be Clear**: Communicate clearly and concisely

### Code of Conduct

We adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). All community members are expected to follow these guidelines.

### Getting Help

- **Documentation**: Check docs first
- **GitHub Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community Discord server

## 📝 License

By contributing to Alpaca Improved, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

We appreciate all contributions, big and small! Contributors will be:

- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to contributor events
- Recognized in project documentation

---

**Thank you for contributing to Alpaca Improved!** 

Your efforts help make options trading more accessible and powerful for everyone. If you have questions about contributing, please don't hesitate to reach out to the maintainers.

For more information, visit:
- **Documentation**: https://alpaca-improved.readthedocs.io/
- **GitHub**: https://github.com/yourusername/alpaca_improved
- **Discord**: [Community Server](https://discord.gg/alpaca-improved) 
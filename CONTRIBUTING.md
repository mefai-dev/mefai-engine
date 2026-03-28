# Contributing to MEFAI Engine

Thank you for your interest in contributing to MEFAI Engine. This document explains
how to get started and what we expect from contributions.

## Getting Started

### Prerequisites

- Python 3.11 or 3.12
- Git

### Development Setup

1. Clone the repository:

```bash
git clone https://github.com/mefai-io/mefai-engine.git
cd mefai-engine
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install in development mode with all dependencies:

```bash
pip install -e ".[all]"
```

4. Verify everything works:

```bash
python -m pytest tests/unit/ -v
ruff check src/ tests/
mypy src/mefai_engine/
```

## Code Style

We use **ruff** for linting and formatting and **mypy** for type checking.

### Ruff

- Target version: Python 3.11
- Line length: 100 characters
- Enabled rule sets: E F W I N UP B A SIM TCH

Run the linter:

```bash
ruff check src/ tests/
```

### Mypy

- Strict mode enabled
- Pydantic plugin enabled

Run type checks:

```bash
mypy src/mefai_engine/
```

### General Style Rules

- Use type annotations on all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and under 50 lines where possible
- Prefer pure functions over stateful classes when practical
- Use numpy vectorized operations instead of Python loops for numeric code

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Write your code following the style guidelines above
3. Add or update tests for any changed behavior
4. Ensure all tests pass locally before pushing
5. Open a pull request against `main`
6. Fill in the PR template with a clear description of what changed and why
7. Wait for at least one maintainer review before merging

### PR Requirements

- All CI checks must pass (lint + tests + type checks)
- New features must include unit tests
- Breaking changes must be documented in the changelog
- Keep PRs focused on a single concern when possible

## Testing

### Running Tests

```bash
# Unit tests only (fast and no external deps)
python -m pytest tests/unit/ -v

# All tests with coverage
python -m pytest tests/ -v --cov=mefai_engine --cov-report=term-missing
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place end to end tests in `tests/e2e/`
- Use descriptive test names that explain the expected behavior
- Each test should verify one thing

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Include your Python version and OS

## Code of Conduct

Be respectful and constructive. We are building something together and everyone
should feel welcome to participate.

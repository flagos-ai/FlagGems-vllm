---
title: Contribution
weight: 60
bookCollapseSection: true
---

# Contributing to FlagGems-vllm

Welcome! Contributions to FlagGems-vllm are highly appreciated.

## Ways to Contribute

- **Add new operators** for vLLM scenarios
- **Optimize existing operators** (improve performance, add autotune configs)
- **Add backend support** (non-NVIDIA hardware)
- **Improve tests** (expand coverage, add edge cases)
- **Write documentation** (guides, examples)
- **Report bugs** (issues, performance regressions)

## Getting Started

1. Read the [operator development protocol](workflow/)
2. Review the [contribution overview](overview/)
3. Set up your development environment
4. Make your changes following the protocol
5. Submit a pull request

## Operator Development Protocol

**MANDATORY**: Read `workflow.md` before adding or modifying operators.

Key requirements:
- **NO_TORCH_COMPUTE_FALLBACK**: Production code must not use torch compute ops
- **Pre-coding tables**: Project truth, torch contract, implementation paths
- **Autotune required**: NVIDIA backend must have tuned configs
- **Standard code drops**: `src/flaggems_vllm/ops/<op>.py`, `tests/test_<op>.py`, `benchmark/test_<op>.py`
- **Acceptance criteria**: SpeedUp ≥ 0.9 on core shapes

The protocol defines hard gates (G0–G7) that must be passed.

## Code Style

Run pre-commit before submitting:

```bash
pip install pre-commit
pre-commit run --all-files
```

Configured tools:
- **black**: Line length 88
- **isort**: `--profile black`
- **flake8**: Max line length 120

## Testing Requirements

All new operators must include:
- Functional tests in `tests/test_<op>.py`
- Performance benchmarks in `benchmark/test_<op>.py`
- Test marker: `@pytest.mark.<op>`

Tests must pass with `--quick` and full modes.

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes following the protocol
4. Run tests and linting
5. Submit a pull request with:
   - Clear description of changes
   - Link to related issue (if any)
   - Test results (functional + performance)

## Communication

- **GitHub Issues**: Bug reports, feature requests
- **Email**: flaggems@baai.ac.cn
- **WeChat**: Join the FlagGems community group

## Topics

- [Overview](overview/) - Contribution guidelines and conventions
- [Workflow Protocol](workflow/) - Operator development protocol
- [Backend Development](backend/) - Adding new hardware backends

## Next Steps

- [Contribution overview](overview/)
- [Operator development protocol](workflow/)

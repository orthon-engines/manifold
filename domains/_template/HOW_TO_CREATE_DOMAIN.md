# How to Create a PRISM Domain

Domains extend PRISM with specialized physics calculations for specific engineering fields.

## Quick Start

1. Create a directory: `domains/your_domain/`
2. Add required files (see below)
3. Validate against known data
4. Submit PR

## Required Files

```
domains/your_domain/
├── DOMAIN.md           # Documentation & requirements
├── __init__.py         # Engine registry
├── engine1.py          # First calculation
├── engine2.py          # Second calculation
└── ...
```

## DOMAIN.md Template

```markdown
# Your Domain Name

Brief description of the physics domain.

## Required Signals

| Pattern | Unit Category | Description |
|---------|---------------|-------------|
| `signal_name` | unit_type | What it measures |

## Required Constants

| Name | Unit | Default | Description |
|------|------|---------|-------------|
| `constant` | unit | value | What it means |

## Capabilities Provided

| Capability | Engine | Output | Description |
|------------|--------|--------|-------------|
| `CAPABILITY` | engine.py | output | What it computes |

## Equations

Document the physics equations used.

## Validated Against

| Test Case | Expected | Notes |
|-----------|----------|-------|
| Known test | Result | Reference |

## References

1. Citation for physics equations
2. Benchmark data source
```

## Engine Template

```python
"""
Engine Name — What It Computes

Physical meaning and equations.

References:
    - Source for equations
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class EngineResult:
    """Result container."""
    value: float
    confidence: float
    warnings: List[str]


def compute(
    input1: np.ndarray,
    input2: np.ndarray,
    **kwargs
) -> EngineResult:
    """
    Compute the thing.

    Args:
        input1: Description
        input2: Description

    Returns:
        EngineResult with computed value
    """
    warnings = []
    confidence = 1.0

    # Validate inputs
    if len(input1) != len(input2):
        warnings.append("Input length mismatch")
        confidence *= 0.5

    # Do the calculation
    result = ...  # Your physics here

    return EngineResult(
        value=result,
        confidence=confidence,
        warnings=warnings,
    )


# Engine metadata for registry
ENGINE_META = {
    'name': 'engine_name',
    'capability': 'CAPABILITY_NAME',
    'description': 'What it computes',
    'requires_signals': ['signal1', 'signal2'],
    'requires_constants': ['constant1'],
    'output_unit': 'unit',
}


# Self-test
if __name__ == "__main__":
    print("Engine Name — Self Test")
    print("=" * 40)

    # Test with known values
    test_input1 = np.array([...])
    test_input2 = np.array([...])
    expected = ...

    result = compute(test_input1, test_input2)

    print(f"Result: {result.value}")
    print(f"Expected: {expected}")
    print(f"Error: {abs(result.value - expected):.6f}")
    print(f"Confidence: {result.confidence:.0%}")
```

## __init__.py Template

```python
"""
Your Domain — Brief Description

Usage:
    from domains.your_domain import engine1, engine2

    result = engine1.compute(data)
"""

from . import engine1
from . import engine2

ENGINES = {
    'engine1': engine1.compute,
    'engine2': engine2.compute,
}

CAPABILITIES = {
    'CAPABILITY1': engine1,
    'CAPABILITY2': engine2,
}

__all__ = [
    'engine1',
    'engine2',
    'ENGINES',
    'CAPABILITIES',
]
```

## Validation Requirements

Your domain must include:

1. **Known-answer tests**: Compare against published benchmark data
2. **Unit tests**: Edge cases, error handling
3. **Self-test in each engine**: `if __name__ == "__main__"`

## Example Domains

See existing domains for reference:

- `domains/turbomachinery/` - Gas turbine physics
- `domains/fluid/` - Incompressible flow analysis

## Checklist Before PR

- [ ] DOMAIN.md with equations and references
- [ ] All engines have docstrings
- [ ] All engines have `ENGINE_META`
- [ ] All engines have self-tests
- [ ] Validated against benchmark data
- [ ] __init__.py exports ENGINES dict
- [ ] No interpretation (labels, classifications) - just numbers

# Berezinian Abliteration Framework 🧬

[![PyPI version](https://badge.fury.io/py/berezinian-abliterator.svg)](https://badge.fury.io/py/berezinian-abliterator)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction 🌟

The Berezinian Abliteration Framework combines three powerful techniques for modifying pre-trained language models:
- Traditional abliteration (behavior suppression)
- Reverse abliteration (behavior enhancement)
- Berezinian (superdeterminant) transformations

This framework enables precise, mathematically-grounded modifications to model behavior while preserving the underlying structure through supermatrix operations.

## Core Mathematical Concepts 📐

### Berezinian Operations

The framework uses the Berezinian (superdeterminant) to handle transformations in the model's weight space:

1. For even supermatrices [A B; C D]:
   ```
   Ber(X) = det(A - BD⁻¹C)det(D)⁻¹
   ```

2. For odd supermatrices, we first apply the J-transformation:
   ```
   J = [0 I; -I 0]
   ```

### Combined Modification Approaches

- **Abliteration**: Suppresses behaviors by subtracting Berezinian-weighted projections
- **Reverse Abliteration**: Enhances behaviors by adding Berezinian-weighted projections
- **Hybrid Operations**: Combines both operations with superdeterminant structure preservation

## Installation 💾

```bash
pip install berezinian-abliterator
```

## Basic Usage 🚀

```python
from berezinian_abliterator import BerezinianAblator

# Initialize the ablator
ablator = BerezinianAblator(
    model="your-model-path",
    dataset=(harmful_data, harmless_data),
    device="cuda"
)

# Cache activations
ablator.cache_activations()

# Perform Berezinian abliteration
ablator.apply_berezinian_ablation(
    directions=ablator.refusal_dirs().values(),
    is_even=True  # Set to False for odd supermatrix structure
)

# Perform reverse abliteration with Berezinian weighting
ablator.apply_berezinian_enhancement(
    directions=ablator.enhancement_dirs().values(),
    strength=0.1
)

# Evaluate results
scores = ablator.berezinian_scores()
```

## Advanced Features 🔧

### 1. Supermatrix Structure Preservation

```python
# Configure even/odd supermatrix handling
ablator.get_super_projection(
    layer=5,
    direction=direction_tensor,
    is_even=True
)
```

### 2. Hybrid Operations

```python
# Combined abliteration and enhancement
with ablator:
    ablator.apply_berezinian_ablation(refusal_dirs)
    ablator.apply_berezinian_enhancement(enhancement_dirs)
    ablator.test_enhancement()
```

### 3. Layer-Specific Modifications

```python
# Target specific layers
ablator.apply_berezinian_ablation(
    directions=directions,
    layers=[3, 4, 5],
    W_O=True,
    mlp=True
)
```

## Key Components 🔍

### BerezinianAblator Class

Core class that implements:
- Berezinian calculations
- Supermatrix projections
- Combined abliteration operations
- Enhancement direction computation
- Activation caching and analysis

### Key Methods

1. `calculate_berezinian()`: Computes superdeterminant
2. `get_super_projection()`: Creates Berezinian-weighted projections
3. `apply_berezinian_ablation()`: Performs suppression with structure preservation
4. `apply_berezinian_enhancement()`: Performs enhancement with structure preservation
5. `berezinian_scores()`: Evaluates modifications using superdeterminant metrics

## Applications 🎯

1. **Enhanced Safety Control**
   - More precise behavior suppression
   - Better preservation of model capabilities
   - Mathematically grounded modifications

2. **Targeted Enhancement**
   - Structure-preserving behavior amplification
   - Controlled capability boosting
   - Stable feature enhancement

3. **Research Applications**
   - Study of model geometry through supermatrix lens
   - Analysis of behavior modification stability
   - Investigation of structure preservation techniques

## Advantages Over Traditional Methods 💪

1. **Mathematical Rigor**
   - Based on superdeterminant theory
   - Preserves supermatrix structure
   - More stable modifications

2. **Precision**
   - Finer control over modifications
   - Better preservation of model properties
   - Reduced unintended side effects

3. **Flexibility**
   - Combined suppression and enhancement
   - Layer-specific modifications
   - Adjustable operation strength

## Limitations and Considerations ⚠️

1. **Computational Overhead**
   - Berezinian calculations add complexity
   - May require more resources than simple abliteration

2. **Mathematical Prerequisites**
   - Understanding of supermatrix operations
   - Knowledge of superdeterminant theory

3. **Stability Considerations**
   - Need for careful balance in hybrid operations
   - Importance of proper structure preservation

## Best Practices 📚

1. **Initialization**
   - Start with small modification strengths
   - Test on subset of layers first
   - Verify structure preservation

2. **Operation Selection**
   - Choose appropriate supermatrix type (even/odd)
   - Balance abliteration and enhancement
   - Monitor stability metrics

3. **Validation**
   - Test modifications extensively
   - Monitor for unintended effects
   - Validate structure preservation

## Contributing 🤝

Contributions welcome! Areas of particular interest:
- Optimization of Berezinian calculations
- New structure preservation techniques
- Enhanced stability measures
- Additional application examples

## Citation 📄

If you use this framework in your research, please cite:

```bibtex
@software{berezinian_abliterator,
  title = {Berezinian Abliteration Framework},
  year = {2024},
  author = {[Your Name]},
  url = {https://github.com/yourusername/berezinian-abliterator}
}
```

## License 📜

Apache License 2.0

## Acknowledgments 🙏

- Based on work from the original Abliterator framework
- Incorporates research on superdeterminant theory
- Built on transformer-lens library


# AbliterationNet 🧬

[![PyPI version](https://badge.fury.io/py/reverseAbliterator.svg)](https://badge.fury.io/py/reverseAbliterator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AbliterationNet is a powerful web application for applying abliteration techniques to transformer models. Built with Next.js, shadcn/ui, and FastAPI, it provides an intuitive interface for model modification using geometric and contour-based abliteration approaches.

## 🌟 Features

- **Interactive Model Loading**: Load any Hugging Face transformer model directly into the application
- **Multiple Abliteration Techniques**:
  - Base Abliteration: Remove undesired model behaviors
  - Reverse Abliteration: Enhance specific capabilities
  - Geometric Abliteration: Topology-preserving modifications
  - Contour Abliteration: Path-based integration methods
  - Berezinian Abliteration: Supermanifold-based approaches
- **Real-time Visualization**: Monitor abliteration effects through interactive dashboards
- **Direct HuggingFace Integration**: Push modified models back to your HuggingFace account

## 🧮 Mathematical Foundations

### Base Abliteration

The foundational abliteration process operates on the model parameter space M where θ ∈ M represents the model parameters. For a given layer l:

```
W'_l = W_l - αP_r(W_l)
```

where:
- P_r(W) is the projection onto the refusal direction r
- α is the ablation strength parameter
- W_l represents the layer weights

### Geometric Abliteration

Geometric abliteration leverages differential geometry to preserve topological structures:

```
T(W) = P_γ ∘ F_t ∘ H(W)
```

where:
- P_γ is the parallel transport operator
- F_t is the geometric flow operator
- H is the harmonic projection operator

### Contour Abliteration

Utilizes vector field integration along eigenvector-defined paths:

```
∫_γ F·dr = ∫₀¹ F(γ(t))·γ'(t)dt
```

where:
- γ(t) represents the integration path
- F is the vector field in activation space

### Reverse Abliteration

Implements capability enhancement through:

```
W'_l = W_l + βP_e(W_l)
```

where:
- P_e(W) is the projection onto enhancement direction e
- β is the enhancement strength parameter

## 🛠 Technical Architecture

### Frontend (Next.js + shadcn/ui)

```typescript
src/
  ├── app/
  │   ├── page.tsx              # Main application page
  │   └── layout.tsx            # Root layout
  ├── components/
  │   ├── ModelLoader.tsx       # HuggingFace model loading interface
  │   ├── AbliterationPanel.tsx # Abliteration controls and visualization
  │   └── MetricsDisplay.tsx    # Real-time metrics visualization
  └── lib/
      └── abliteration/         # Frontend abliteration utilities
```

### Backend (FastAPI)

```python
app/
  ├── main.py                   # FastAPI application
  ├── abliterator/
  │   ├── base.py              # Base abliteration implementation
  │   ├── geometric.py         # Geometric abliteration methods
  │   ├── contour.py          # Contour integration methods
  │   └── reverse.py          # Reverse abliteration implementation
  └── utils/
      └── model_handling.py    # Model loading and saving utilities
```

## 🚀 Getting Started

1. Install Dependencies:
```bash
# Frontend
npm install

# Backend
pip install reverseAbliterator fastapi uvicorn
```

2. Configure Environment:
```bash
# .env
HUGGINGFACE_TOKEN=your_token_here
```

3. Run the Application:
```bash
# Frontend
npm run dev

# Backend
uvicorn app.main:app --reload
```

## 📚 Usage Example

```python
from reverseAbliterator import ReverseAbliterator

# Initialize abliterator
abliterator = ReverseAbliterator(
    model="bert-base-uncased",
    dataset=([target_data, baseline_data]),
)

# Cache activations
abliterator.cache_activations(N=100)

# Apply enhancement
abliterator.enhance_model(strength=0.1)

# Save modified model
abliterator.save_modified_model("enhanced_model")
```

## 🎯 Key Algorithms

### Berezinian Ablation

For supermatrix modifications:

```python
def calculate_berezinian(matrix):
    """
    Calculate superdeterminant for even/odd cases
    """
    n = matrix.shape[0] // 2
    A = matrix[:n, :n]
    B = matrix[:n, n:]
    C = matrix[n:, :n]
    D = matrix[n:, n:]
    
    return det(A - B @ D^(-1) @ C) / det(D)
```

### Geometric Flow Integration

```python
def geometric_flow(initial, target, steps):
    """
    Apply Ricci flow for geometric modification
    """
    flow = [initial]
    for t in range(steps):
        ricci = compute_ricci_curvature(flow[-1])
        flow.append(flow[-1] - ricci @ (flow[-1] - target))
    return flow
```

## 📊 Performance Metrics

The system tracks several key metrics:

1. **Direction Quality**:
```
cos(d,μ_target) > cos(d,μ_other) + δ
```

2. **Modification Bounds**:
```
‖W'‖_F ≤ (1 + γ)‖W‖_F
```

3. **Convergence Rate**:
```
‖T(W) - W*‖ ≤ Ce^{-λt}
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [ReverseAbliterator](https://github.com/Leoleojames1/reverseAbliterator)
- [Transformer Lens](https://github.com/neelnanda-io/TransformerLens)

## 📚 Citations

If you use AbliterationNet in your research, please cite:

```bibtex
@software{abliterationnet2024,
  author = {Leo James},
  title = {AbliterationNet: A Framework for Model Behavior Modification},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Leoleojames1/abliterationNet}
}
```
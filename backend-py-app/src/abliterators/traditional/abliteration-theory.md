# Mathematical Framework for Abliterator Systems

## 1. Common Foundations

### 1.1 Model Space Definition
Let M be the space of transformer model parameters where:
- θ ∈ M represents model parameters
- H ⊂ ℝᵈ represents hidden state space with dimension d
- f_θ: X → Y represents the model function

### 1.2 Activation Space
For layer l:
- A_l ⊂ ℝᵈ represents activation space
- h_l(x) ∈ A_l represents activations for input x
- W_l ∈ ℝᵈˣᵈ represents weight matrices

## 2. Base Abliterator Framework

### 2.1 Harmful vs. Harmless Directions
For harmful activations H and harmless activations B:

```
ΣH = 1/n Σᵢ hᵢhᵢᵀ    (harmful covariance)
ΣB = 1/n Σᵢ bᵢbᵢᵀ    (harmless covariance)
```

### 2.2 Refusal Direction Calculation
The refusal direction r is computed as:

```
r = μH - μB
r̂ = r/‖r‖₂       (normalized refusal direction)
```

where:
- μH = 1/n Σᵢ hᵢ (harmful mean)
- μB = 1/n Σᵢ bᵢ (harmless mean)

### 2.3 Layer Modification
For each layer l:

```
W'_l = W_l - αP_r(W_l)
```

where:
- P_r(W) = (Wr̂)r̂ᵀ is the projection onto r
- α is the ablation strength

### 2.4 Scoring Function
For token probabilities p:

```
s(x) = max(p(t|x): t ∈ T_neg)
```

where T_neg is the set of negative tokens.

## 3. Reverse Abliterator Framework

### 3.1 Target vs. Baseline Directions
For target activations T and baseline activations B:

```
ΣT = 1/n Σᵢ tᵢtᵢᵀ    (target covariance)
ΣB = 1/n Σᵢ bᵢbᵢᵀ    (baseline covariance)
```

### 3.2 Enhancement Direction Calculation
The enhancement direction e is:

```
e = μT - μB
ê = e/‖e‖₂       (normalized enhancement direction)
```

where:
- μT = 1/n Σᵢ tᵢ (target mean)
- μB = 1/n Σᵢ bᵢ (baseline mean)

### 3.3 Layer Enhancement
For each layer l:

```
W'_l = W_l + βP_e(W_l)
```

where:
- P_e(W) = (Wê)êᵀ is the projection onto e
- β is the enhancement strength

### 3.4 Enhancement Score
For token probabilities p:

```
s(x) = max(p(t|x): t ∈ T_pos)
```

where T_pos is the set of target tokens.

## 4. Key Differences

### 4.1 Objective Functions

Base Abliterator:
```
min_θ 𝔼ₓ∼X[max p(t|x) for t ∈ T_neg]
```

Reverse Abliterator:
```
max_θ 𝔼ₓ∼X[max p(t|x) for t ∈ T_pos]
```

### 4.2 Modification Direction
Base: Subtractive modification
```
ΔW = -αP_r(W)
```

Reverse: Additive modification
```
ΔW = +βP_e(W)
```

## 5. Optimization Theory

### 5.1 Gradient Perspective
For both systems, the effective gradient is:

```
∇W = ±γP_d(W)
```

where:
- γ is the learning rate (α or β)
- d is the direction (r or e)
- ± depends on ablation vs. enhancement

### 5.2 Convergence Conditions
For stable convergence:

```
‖W'_l - W_l‖_F ≤ ε‖W_l‖_F
```

where:
- ‖·‖_F is the Frobenius norm
- ε is the stability threshold

## 6. Performance Metrics

### 6.1 Directional Alignment
For direction d and activations a:

```
alignment(d,a) = |d^⊤a|/(‖d‖₂‖a‖₂)
```

### 6.2 Layer-wise Effect
For layer l:

```
effect(l) = ‖P_d(W_l)‖_F/‖W_l‖_F
```

## 7. Implementation Considerations

### 7.1 Numerical Stability
Add regularization to covariance matrices:

```
Σ' = Σ + λI
```

where λ is a small positive constant.

### 7.2 Batch Processing
For batch size b:

```
μ_batch = 1/b Σᵢ₌₁ᵇ aᵢ
```

## 8. Theoretical Guarantees

### 8.1 Direction Quality
For good direction vectors:

```
cos(d,μ_target) > cos(d,μ_other) + δ
```

where δ is the separation margin.

### 8.2 Modification Bounds
For stable modifications:

```
‖W'‖_F ≤ (1 + γ)‖W‖_F
```

where γ is the strength parameter.

## 9. Complexity Analysis

Space complexity: O(d² + nd)
Time complexity: O(d³ + ndk)

where:
- d is hidden state dimension
- n is number of samples
- k is number of top directions

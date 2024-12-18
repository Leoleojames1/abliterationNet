import torch
import torch.nn.functional as F
import numpy as np
import einops
from typing import Dict, List, Optional, Tuple, Union, Callable
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, utils
from dataclasses import dataclass

@dataclass
class UnifiedConfig:
    contour_resolution: int = 100
    super_structure: str = "even"
    integration_method: str = "trapezoidal"
    n_eigenvectors: int = 3
    adaptive_resolution: bool = True
    preserve_threshold: float = 0.1
    cache_computations: bool = True
    
class UnifiedBerezinianContourAbliterator:
    def __init__(
        self,
        model: str,
        dataset: Tuple[List[str], List[str]],
        device: str = 'cuda',
        config: UnifiedConfig = None,
        activation_layers: List[str] = ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
    ):
        self.model = HookedTransformer.from_pretrained_no_processing(
            model,
            device=device,
            dtype=torch.bfloat16
        )
        self.config = config or UnifiedConfig()
        self.dataset = dataset
        self.device = device
        self.activation_layers = activation_layers
        self.cache = {}
        self._setup_model()

    def _setup_model(self):
        """Initialize model settings"""
        self.hidden_size = self.model.cfg.d_model
        self.n_layers = self.model.cfg.n_layers
        self.model.requires_grad_(False)
        self.original_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
    def calculate_berezinian(
        self,
        matrix: Float[Tensor, "d_model d_model"],
        is_even: bool = True
    ) -> Float[Tensor, "scalar"]:
        """Calculate the Berezinian (superdeterminant) of a matrix."""
        n = matrix.shape[0] // 2
        A = matrix[:n, :n]
        B = matrix[:n, n:]
        C = matrix[n:, :n]
        D = matrix[n:, n:]
        
        try:
            if is_even:
                D_inv = torch.linalg.inv(D)
                schur = A - B @ D_inv @ C
                return torch.linalg.det(schur) / torch.linalg.det(D)
            else:
                J = torch.zeros_like(matrix)
                J[:n, n:] = torch.eye(n, device=matrix.device)
                J[n:, :n] = -torch.eye(n, device=matrix.device)
                transformed = J @ matrix
                return self.calculate_berezinian(transformed, is_even=True)
        except torch.linalg.LinAlgError:
            return torch.tensor(0.0, device=matrix.device)

    def generate_contour_path(
        self,
        activation: Float[Tensor, "batch d_model"],
        n_points: Optional[int] = None
    ) -> Tuple[Float[Tensor, "points d_model"], Float[Tensor, "points d_model"]]:
        """Generate contour path using eigenvectors and Berezinian weighting."""
        n_points = n_points or self.config.contour_resolution
        
        # Compute activation covariance and its eigenvectors
        centered = activation - torch.mean(activation, dim=0, keepdim=True)
        cov = torch.mm(centered.T, centered) / (activation.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Select top eigenvectors
        top_eigenvectors = eigenvectors[:, -self.config.n_eigenvectors:]
        
        # Generate base contour path
        t = torch.linspace(0, 2*np.pi, n_points, device=activation.device)
        contour_path = torch.zeros(n_points, activation.shape[1], device=activation.device)
        tangent_vectors = torch.zeros_like(contour_path)
        
        # Create path using eigenvectors
        for i in range(self.config.n_eigenvectors):
            angle = 2 * np.pi * i / self.config.n_eigenvectors
            weight = torch.sqrt(eigenvalues[-i-1])  # Weight by eigenvalue importance
            
            contour_path += weight * torch.outer(torch.cos(t + angle), top_eigenvectors[:, i])
            tangent_vectors += weight * torch.outer(-torch.sin(t + angle), top_eigenvectors[:, i])
            
        return contour_path, tangent_vectors

    def compute_berezinian_weights(
        self,
        path: Float[Tensor, "points d_model"]
    ) -> Float[Tensor, "points"]:
        """Compute Berezinian weights along the contour path."""
        weights = torch.zeros(path.shape[0], device=path.device)
        
        for i in range(path.shape[0] - 1):
            # Create local transformation matrix
            delta = path[i+1] - path[i]
            transform = torch.outer(path[i], delta)
            
            # Calculate Berezinian
            ber = self.calculate_berezinian(
                transform,
                is_even=(self.config.super_structure == "even")
            )
            weights[i] = torch.abs(ber)
            
        # Normalize weights
        weights = F.softmax(weights, dim=0)
        return weights

    def weighted_contour_integral(
        self,
        vector_field: Float[Tensor, "batch d_model"],
        path: Float[Tensor, "points d_model"],
        weights: Float[Tensor, "points"],
        tangent_vectors: Optional[Float[Tensor, "points d_model"]] = None
    ) -> Float[Tensor, "batch"]:
        """Compute weighted contour integral using Berezinian weights."""
        if tangent_vectors is None:
            tangent_vectors = path[1:] - path[:-1]
            tangent_vectors = F.pad(tangent_vectors, (0, 0, 0, 1))
            
        # Normalize tangent vectors
        tangent_vectors = tangent_vectors / (torch.norm(tangent_vectors, dim=-1, keepdim=True) + 1e-8)
        
        # Interpolate vector field along path
        field_values = einops.einsum(
            vector_field, path,
            "batch d_model, points d_model -> batch points"
        )
        
        # Apply Berezinian weights
        weighted_values = field_values * weights.unsqueeze(0)
        
        # Compute dot product with tangent vectors
        integrand = einops.einsum(
            weighted_values, tangent_vectors,
            "batch points, points d_model -> batch points"
        )
        
        # Perform numerical integration
        if self.config.integration_method == "trapezoidal":
            integral = torch.trapz(integrand, dim=-1)
        elif self.config.integration_method == "simpson":
            integral = torch.simpson(integrand, dim=-1)
        else:
            integral = torch.sum(integrand, dim=-1) / path.shape[0]
            
        return integral

    def unified_transform(
        self,
        activation: Float[Tensor, "batch d_model"],
        strength: float = 1.0
    ) -> Float[Tensor, "batch d_model"]:
        """Apply unified Berezinian-Contour transformation."""
        # Generate optimal path
        path, tangent_vectors = self.generate_contour_path(activation)
        
        # Compute Berezinian weights
        weights = self.compute_berezinian_weights(path)
        
        # Compute weighted contour integral
        integral = self.weighted_contour_integral(activation, path, weights, tangent_vectors)
        
        # Apply transformation with strength
        return activation + strength * integral.unsqueeze(-1) * torch.mean(path, dim=0)

    def modify_layer(
        self,
        layer: int,
        pattern: Float[Tensor, "d_model"],
        strength: float = 1.0,
        modify_attention: bool = True,
        modify_mlp: bool = True
    ):
        """Modify a specific layer using unified transformation."""
        if modify_attention:
            W_O = self.model.blocks[layer].attn.W_O.data
            transformed = self.unified_transform(W_O, strength)
            self.model.blocks[layer].attn.W_O.data = transformed

        if modify_mlp:
            W_out = self.model.blocks[layer].mlp.W_out.data
            transformed = self.unified_transform(W_out, strength)
            self.model.blocks[layer].mlp.W_out.data = transformed

    def detect_patterns(
        self,
        cache_dict: Dict[str, Float[Tensor, "batch seq d_model"]],
        threshold: float = None
    ) -> Dict[str, Float[Tensor, "batch"]]:
        """Detect patterns using unified approach."""
        threshold = threshold or self.config.preserve_threshold
        patterns = {}
        
        for key, activations in cache_dict.items():
            batch_size, seq_len, d_model = activations.shape
            activations = activations.view(-1, d_model)
            
            # Apply unified transformation
            transformed = self.unified_transform(activations)
            
            # Compute pattern scores
            scores = torch.norm(transformed - activations, dim=-1)
            scores = scores.view(batch_size, seq_len)
            
            # Threshold significant patterns
            patterns[key] = scores * (scores > threshold * scores.std())
            
        return patterns

    def cache_activations(
        self,
        texts: List[str],
        batch_size: int = 8
    ):
        """Cache activations for analysis."""
        self.cache = {}
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.model.tokenizer(batch, padding=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tokens.input_ids, output_hidden_states=True)
                
            for layer in range(self.n_layers):
                key = f"layer_{layer}"
                if key not in self.cache:
                    self.cache[key] = []
                self.cache[key].append(outputs.hidden_states[layer].cpu())
                
        # Concatenate cached activations
        self.cache = {k: torch.cat(v, dim=0) for k, v in self.cache.items()}

    def apply_unified_modification(
        self,
        layers: Optional[List[int]] = None,
        strength: float = 1.0,
        modify_attention: bool = True,
        modify_mlp: bool = True
    ):
        """Apply unified modification across specified layers."""
        if layers is None:
            layers = range(self.n_layers)
            
        if not self.cache:
            raise ValueError("No cached activations. Run cache_activations first.")
            
        for layer in layers:
            activations = self.cache[f"layer_{layer}"]
            pattern = torch.mean(activations, dim=(0,1))  # Average pattern
            self.modify_layer(
                layer,
                pattern,
                strength,
                modify_attention,
                modify_mlp
            )

    def reset_model(self):
        """Reset model to original state."""
        self.model.load_state_dict(self.original_state)
        self.cache = {}

    def save_state(self, path: str):
        """Save current model state."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'cache': self.cache
        }, path)

    def load_state(self, path: str):
        """Load saved model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.config = checkpoint['config']
        self.cache = checkpoint['cache']

    @staticmethod
    def batch_data(data: List[str], batch_size: int) -> List[List[str]]:
        """Helper function to batch data."""
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

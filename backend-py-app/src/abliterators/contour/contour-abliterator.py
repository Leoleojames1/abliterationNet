import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from jaxtyping import Float
import einops

class ContourAbliterator:
    def __init__(
        self,
        model: str,
        dataset: Tuple[List[str], List[str]],
        device: str = 'cuda',
        contour_resolution: int = 100,
        integration_method: str = 'trapezoidal'
    ):
        # Initialize base abliterator
        super().__init__(model, dataset, device)
        self.contour_resolution = contour_resolution
        self.integration_method = integration_method
        
    def compute_contour_integral(
        self,
        vector_field: Float[Tensor, "batch d_model"],
        contour_path: Float[Tensor, "points d_model"],
        tangent_vectors: Optional[Float[Tensor, "points d_model"]] = None
    ) -> Float[Tensor, "batch"]:
        """
        Compute line integral along a contour path in the activation space.
        
        Args:
            vector_field: The vector field to integrate over
            contour_path: Points defining the contour path
            tangent_vectors: Optional pre-computed tangent vectors
        """
        if tangent_vectors is None:
            # Compute tangent vectors using finite differences
            tangent_vectors = contour_path[1:] - contour_path[:-1]
            # Pad to match original size
            tangent_vectors = F.pad(tangent_vectors, (0, 0, 0, 1))
            
        # Normalize tangent vectors
        tangent_vectors = tangent_vectors / (torch.norm(tangent_vectors, dim=-1, keepdim=True) + 1e-8)
        
        # Interpolate vector field along contour path
        field_values = einops.einsum(
            vector_field, contour_path,
            "batch d_model, points d_model -> batch points"
        )
        
        # Compute dot product with tangent vectors
        integrand = einops.einsum(
            field_values, tangent_vectors,
            "batch points, points d_model -> batch points"
        )
        
        # Perform numerical integration
        if self.integration_method == 'trapezoidal':
            integral = torch.trapz(integrand, dim=-1)
        elif self.integration_method == 'simpson':
            integral = torch.simpson(integrand, dim=-1)
        else:
            # Simple rectangular integration
            integral = torch.sum(integrand, dim=-1) / self.contour_resolution
            
        return integral

    def generate_eigenvector_contour(
        self,
        activation: Float[Tensor, "batch d_model"],
        n_eigenvectors: int = 3
    ) -> Tuple[Float[Tensor, "points d_model"], Float[Tensor, "points d_model"]]:
        """
        Generate a contour path using principal eigenvectors of the activation covariance.
        """
        # Compute activation covariance
        centered = activation - torch.mean(activation, dim=0, keepdim=True)
        cov = torch.mm(centered.T, centered) / (activation.shape[0] - 1)
        
        # Get top eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        top_eigenvectors = eigenvectors[:, -n_eigenvectors:]
        
        # Generate contour path along eigenvectors
        t = torch.linspace(0, 2*np.pi, self.contour_resolution, device=activation.device)
        contour_path = torch.zeros(self.contour_resolution, activation.shape[1], device=activation.device)
        
        for i in range(n_eigenvectors):
            angle = 2 * np.pi * i / n_eigenvectors
            contour_path += torch.outer(
                torch.cos(t + angle),
                top_eigenvectors[:, i]
            )
            
        # Compute tangent vectors analytically
        tangent_vectors = torch.zeros_like(contour_path)
        for i in range(n_eigenvectors):
            angle = 2 * np.pi * i / n_eigenvectors
            tangent_vectors += torch.outer(
                -torch.sin(t + angle),
                top_eigenvectors[:, i]
            )
            
        return contour_path, tangent_vectors

    def detect_activation_patterns(
        self,
        cache_dict: Dict[str, Float[Tensor, "batch seq d_model"]],
        threshold: float = 0.5,
        mode: str = 'strongest'
    ) -> Dict[str, Float[Tensor, "batch"]]:
        """
        Detect activation patterns using contour integrals.
        
        Args:
            cache_dict: Dictionary of activation caches
            threshold: Detection threshold
            mode: 'strongest' or 'all' values
        """
        pattern_scores = {}
        
        for key, activations in cache_dict.items():
            batch_size, seq_len, d_model = activations.shape
            activations = activations.view(-1, d_model)
            
            # Generate contour path using eigenvectors
            contour_path, tangent_vectors = self.generate_eigenvector_contour(activations)
            
            # Compute contour integrals
            integrals = self.compute_contour_integral(
                activations, 
                contour_path, 
                tangent_vectors
            )
            
            # Process based on mode
            if mode == 'strongest':
                # Keep only strongest signals
                mask = torch.abs(integrals) > threshold * torch.max(torch.abs(integrals))
                pattern_scores[key] = integrals * mask
            else:  # mode == 'all'
                # Keep all values above threshold
                mask = torch.abs(integrals) > threshold * torch.std(integrals)
                pattern_scores[key] = integrals * mask
                
            # Reshape back to batch x seq
            pattern_scores[key] = pattern_scores[key].view(batch_size, seq_len)
            
        return pattern_scores

    def apply_contour_ablation(
        self,
        direction: Float[Tensor, "d_model"],
        strength: float = 1.0,
        layers: Optional[List[int]] = None
    ):
        """
        Apply ablation using contour integral information.
        """
        if layers is None:
            layers = list(range(1, self.model.cfg.n_layers))
            
        # Generate contour path for the ablation direction
        contour_path, tangent_vectors = self.generate_eigenvector_contour(
            direction.unsqueeze(0),
            n_eigenvectors=1
        )
        
        for layer in layers:
            # Get layer weights
            W_O = self.layer_attn(layer)
            W_mlp = self.layer_mlp(layer)
            
            # Compute contour integrals for both weights
            W_O_integral = self.compute_contour_integral(W_O, contour_path, tangent_vectors)
            W_mlp_integral = self.compute_contour_integral(W_mlp, contour_path, tangent_vectors)
            
            # Apply ablation based on integral values
            self.layer_attn(layer, W_O - strength * W_O_integral.view(-1, 1) * direction)
            self.layer_mlp(layer, W_mlp - strength * W_mlp_integral.view(-1, 1) * direction)
            
    def enhance_activation(
        self,
        cache_dict: Dict[str, Float[Tensor, "batch seq d_model"]],
        target_pattern: Float[Tensor, "d_model"],
        strength: float = 1.0
    ):
        """
        Enhance specific activation patterns using contour integrals.
        """
        pattern_scores = self.detect_activation_patterns(cache_dict)
        
        for key, scores in pattern_scores.items():
            layer = int(key.split('.')[1])  # Extract layer number from key
            
            # Compute enhancement direction using contour integral
            contour_path, tangent_vectors = self.generate_eigenvector_contour(target_pattern.unsqueeze(0))
            integral = self.compute_contour_integral(
                cache_dict[key].view(-1, target_pattern.shape[0]),
                contour_path,
                tangent_vectors
            )
            
            # Apply enhancement
            enhancement = strength * integral.view(-1, 1) * target_pattern
            self.apply_contour_ablation(enhancement, strength=1.0, layers=[layer])


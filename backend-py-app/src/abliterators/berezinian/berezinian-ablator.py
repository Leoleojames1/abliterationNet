import torch
import torch.nn.functional as F
import einops
from typing import Dict, List, Tuple, Optional
from torch import Tensor
from jaxtyping import Float, Int

class BerezinianAblator(ModelAbliterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def calculate_berezinian(
        self,
        matrix: Float[Tensor, "d_model d_model"],
        is_even: bool = True
    ) -> Float[Tensor, "scalar"]:
        """Calculate the Berezinian (superdeterminant) of a matrix.
        
        For even supermatrices of form [A B; C D], Ber(X) = det(A-BD^(-1)C)det(D)^(-1)
        For odd supermatrices, we first transform with J = [0 I; -I 0]
        """
        n = matrix.shape[0] // 2
        A = matrix[:n, :n]
        B = matrix[:n, n:]
        C = matrix[n:, :n]
        D = matrix[n:, n:]
        
        if is_even:
            # Handle even case
            try:
                D_inv = torch.linalg.inv(D)
                schur = A - B @ D_inv @ C
                return torch.linalg.det(schur) / torch.linalg.det(D)
            except torch.linalg.LinAlgError:
                # Handle singular matrices
                return torch.tensor(0.0, device=matrix.device)
        else:
            # Handle odd case by first applying J transformation
            J = torch.zeros_like(matrix)
            J[:n, n:] = torch.eye(n, device=matrix.device)
            J[n:, :n] = -torch.eye(n, device=matrix.device)
            transformed = J @ matrix
            
            # Now calculate as even case
            return self.calculate_berezinian(transformed, is_even=True)

    def get_super_projection(
        self,
        layer: int,
        direction: Float[Tensor, "d_model"],
        is_even: bool = True
    ) -> Float[Tensor, "d_model d_model"]:
        """Calculate superprojection matrix using Berezinian properties"""
        d_model = self.hidden_size
        half_dim = d_model // 2
        
        # Create projection matrix
        proj_matrix = torch.outer(direction, direction)
        
        # Split into blocks for super-structure
        blocks = {
            'A': proj_matrix[:half_dim, :half_dim],
            'B': proj_matrix[:half_dim, half_dim:],
            'C': proj_matrix[half_dim:, :half_dim],
            'D': proj_matrix[half_dim:, half_dim:]
        }
        
        # Calculate Berezinian-weighted projection
        ber = self.calculate_berezinian(proj_matrix, is_even)
        return proj_matrix * ber

    def apply_berezinian_ablation(
        self,
        directions: List[Float[Tensor, "d_model"]],
        layers: Optional[List[int]] = None,
        W_O: bool = True,
        mlp: bool = True,
        is_even: bool = True
    ):
        """Apply Berezinian-based ablation across specified layers"""
        if layers is None:
            layers = self.get_whitelisted_layers()
            
        for direction in directions:
            for layer in layers:
                # Get super-projection for this direction
                super_proj = self.get_super_projection(layer, direction, is_even)
                
                # Apply to attention and MLP weights as specified
                if W_O:
                    W_O_matrix = self.layer_attn(layer)
                    self.layer_attn(layer, W_O_matrix - super_proj @ W_O_matrix)
                    
                if mlp:
                    mlp_matrix = self.layer_mlp(layer)
                    self.layer_mlp(layer, mlp_matrix - super_proj @ mlp_matrix)

    def berezinian_scores(
        self,
        N: int = 4,
        is_even: bool = True,
        **kwargs
    ) -> Dict[str, Float[Tensor, "d_model"]]:
        """Calculate ablation scores using Berezinian properties"""
        if not self.harmful:
            raise ValueError("No cached activations found. Run cache_activations first.")
            
        # Get directions using existing refusal_dirs
        directions = self.refusal_dirs()
        
        scores = {}
        for key, direction in directions.items():
            # Calculate super-projection
            super_proj = self.get_super_projection(
                int(re.search(r"\.(\d+)\.", key)[1]) if '.' in key else 0,
                direction,
                is_even
            )
            
            # Calculate scores using projection
            with self:
                self.apply_berezinian_ablation([direction], is_even=is_even)
                scores[key] = self.measure_scores(N=N, **kwargs)
                
        return scores

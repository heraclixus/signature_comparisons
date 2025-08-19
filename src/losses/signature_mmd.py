"""
Signature Maximum Mean Discrepancy Loss (Extracted from sigker_nsdes)

This implements MMD loss using signature kernels, originally implemented
in the sigker_nsdes discriminators.
"""

import torch
from typing import Any, Optional


class SignatureMMDLoss:
    """
    Maximum Mean Discrepancy Loss using Signature Kernels.
    
    Computes MMD between generated and real path distributions in signature space:
    MMD²(P, Q) = ||μ_P - μ_Q||²_H
    
    where μ_P, μ_Q are mean embeddings in the signature kernel RKHS.
    """
    
    def __init__(self, signature_kernel: Any, max_batch: int = 128,
                 adversarial: bool = False, path_dim: int = 2):
        """
        Initialize Signature MMD Loss.
        
        Args:
            signature_kernel: Signature kernel object (e.g., from sigkernel package)
            max_batch: Maximum batch size for kernel computation
            adversarial: Whether to include learnable scaling parameters
            path_dim: Dimension of paths (for adversarial scaling)
        """
        self.signature_kernel = signature_kernel
        self.max_batch = max_batch
        self.adversarial = adversarial
        
        if adversarial:
            # Learnable scaling parameters
            inits = torch.ones(path_dim)
            self._sigma = torch.nn.Parameter(inits, requires_grad=True)
        else:
            self._sigma = None
    
    def _compute_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD between two sets of paths using signature kernel.
        
        Args:
            X: First set of paths, shape (batch, time, channels)
            Y: Second set of paths, shape (batch, time, channels)
            
        Returns:
            MMD value
        """
        return self.signature_kernel.compute_mmd(X, Y, max_batch=self.max_batch)
    
    def __call__(self, generated_paths: torch.Tensor, 
                 real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature MMD loss.
        
        Args:
            generated_paths: Generated paths, shape (batch, time, channels)
            real_paths: Real paths, shape (batch, time, channels)
            
        Returns:
            MMD loss value
        """
        mu = generated_paths.clone().type(torch.float64)
        nu = real_paths.clone().type(torch.float64)
        
        # Apply learnable scaling if adversarial
        if self._sigma is not None:
            # Only scale non-time dimensions (skip first channel which is time)
            mu[..., 1:] *= self._sigma
            # Note: In adversarial training, real paths typically not scaled
        
        return self._compute_mmd(mu, nu)
    
    def parameters(self):
        """Return learnable parameters (for adversarial case)."""
        if self._sigma is not None:
            return [self._sigma]
        return []


class ScaledSignatureMMDLoss(SignatureMMDLoss):
    """
    MMD Loss with multiple fixed scalings.
    """
    
    def __init__(self, signature_kernel: Any, path_scalings: list,
                 max_batch: int = 128, **kwargs):
        """
        Initialize Scaled Signature MMD Loss.
        
        Args:
            signature_kernel: Signature kernel object
            path_scalings: List of scaling factors to apply
            max_batch: Maximum batch size
        """
        super().__init__(signature_kernel, max_batch, **kwargs)
        self.path_scalings = path_scalings
    
    def __call__(self, generated_paths: torch.Tensor, 
                 real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled MMD loss over multiple scalings.
        
        Args:
            generated_paths: Generated paths
            real_paths: Real paths
            
        Returns:
            Average MMD loss over scalings
        """
        total_loss = 0
        
        for scaling in self.path_scalings:
            # Apply fixed scaling
            mu = generated_paths.clone().type(torch.float64) * scaling
            nu = real_paths.clone().type(torch.float64)
            
            # Apply learnable scaling if adversarial
            if self._sigma is not None:
                mu[..., 1:] *= self._sigma
            
            total_loss += self._compute_mmd(mu, nu)
        
        return total_loss / len(self.path_scalings)


class WeightedSignatureMMDLoss(SignatureMMDLoss):
    """
    Weighted MMD Loss with learnable weights for multiple scalings.
    """
    
    def __init__(self, signature_kernel: Any, path_scalings: list, 
                 weights: list, max_batch: int = 128, **kwargs):
        """
        Initialize Weighted Signature MMD Loss.
        
        Args:
            signature_kernel: Signature kernel object
            path_scalings: List of scaling factors
            weights: List of weights for each scaling
            max_batch: Maximum batch size
        """
        super().__init__(signature_kernel, max_batch, **kwargs)
        
        assert len(path_scalings) == len(weights), \
            "Number of scalings must match number of weights"
        
        self.path_scalings = path_scalings
        
        if self.adversarial:
            # Learnable weights
            _weights = torch.tensor(weights, dtype=torch.float)
            self._weights = torch.nn.Parameter(_weights, requires_grad=True)
        else:
            self._weights = torch.tensor(weights, dtype=torch.float)
    
    def __call__(self, generated_paths: torch.Tensor, 
                 real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MMD loss.
        
        Args:
            generated_paths: Generated paths
            real_paths: Real paths
            
        Returns:
            Weighted MMD loss
        """
        total_loss = 0
        weights_sum = torch.sum(torch.abs(self._weights))
        
        for i, scaling in enumerate(self.path_scalings):
            # Apply fixed scaling
            mu = generated_paths.clone().type(torch.float64) * scaling
            nu = real_paths.clone().type(torch.float64)
            
            # Apply learnable scaling if adversarial
            if self._sigma is not None:
                mu[..., 1:] *= self._sigma
            
            # Weight the MMD contribution
            weight = torch.abs(self._weights[i]) / weights_sum
            total_loss += weight * self._compute_mmd(mu, nu)
        
        return total_loss
    
    def parameters(self):
        """Return learnable parameters."""
        params = super().parameters()
        if self.adversarial:
            params.append(self._weights)
        return params

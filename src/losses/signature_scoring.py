"""
Signature Kernel Scoring Rule Loss (Extracted from sigker_nsdes)

This implements the proper scoring rule loss using signature kernels,
originally implemented in sigker_nsdes discriminators.
"""

import torch
from typing import Callable, Optional, Any


class SignatureScoringLoss:
    """
    Signature Kernel Scoring Rule Loss Function.
    
    Implements the proper scoring rule:
    S(P, Y) = E_P[k(X, X)] - 2 * E_P[k(X, Y)]
    
    where P is the generated distribution and Y is real data.
    This is a proper scoring rule that is minimized when P matches the true distribution.
    """
    
    def __init__(self, signature_kernel: Any, max_batch: int = 128, 
                 adversarial: bool = False, path_dim: int = 2):
        """
        Initialize Signature Scoring Loss.
        
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
            # Learnable scaling parameters (as in original implementation)
            inits = torch.ones(path_dim)
            self._sigma = torch.nn.Parameter(inits, requires_grad=True)
        else:
            self._sigma = None
    
    def _scoring_rule(self, X: torch.Tensor, y: torch.Tensor, 
                     pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the scoring rule between generated samples X and real sample y.
        
        Args:
            X: Generated samples, shape (batch, time, channels)
            y: Real sample, shape (time, channels)
            pi: Optional scaling parameter
            
        Returns:
            Scoring rule value
        """
        if pi is None:
            return self.signature_kernel.compute_scoring_rule(
                X, y.unsqueeze(0), max_batch=self.max_batch
            )
        else:
            # Scaled version for phi-kernel
            piX = X.clone() * pi
            K_XX = self.signature_kernel.compute_Gram(
                piX, X, sym=False, max_batch=self.max_batch
            )
            K_Xy = self.signature_kernel.compute_Gram(
                piX, y.unsqueeze(0), sym=False, max_batch=self.max_batch
            )
            
            mK_XX = (torch.sum(K_XX) - torch.sum(torch.diag(K_XX))) / \
                   (K_XX.shape[0] * (K_XX.shape[0] - 1.))
            
            return mK_XX - 2. * torch.mean(K_Xy)
    
    def __call__(self, generated_paths: torch.Tensor, 
                 real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature scoring rule loss.
        
        Args:
            generated_paths: Generated paths, shape (batch, time, channels)
            real_paths: Real paths, shape (batch, time, channels)
            
        Returns:
            Scoring rule loss
        """
        mu = generated_paths.clone().type(torch.float64)
        nu = real_paths.clone().type(torch.float64)
        
        # Apply learnable scaling if adversarial
        if self._sigma is not None:
            # Only scale non-time dimensions (skip first channel which is time)
            mu[..., 1:] *= self._sigma
        
        # For now, use first real sample as target (can be extended)
        return self._scoring_rule(mu, nu[0])
    
    def parameters(self):
        """Return learnable parameters (for adversarial case)."""
        if self._sigma is not None:
            return [self._sigma]
        return []


class WeightedSignatureScoringLoss(SignatureScoringLoss):
    """
    Weighted version using multiple scalings (phi-kernel approximation).
    """
    
    def __init__(self, signature_kernel: Any, n_scalings: int = 5, 
                 max_batch: int = 128, **kwargs):
        """
        Initialize Weighted Signature Scoring Loss.
        
        Args:
            signature_kernel: Signature kernel object
            n_scalings: Number of random scalings to use
            max_batch: Maximum batch size
        """
        super().__init__(signature_kernel, max_batch, **kwargs)
        self._scalings = torch.zeros(n_scalings).exponential_()
        self.n_scalings = n_scalings
    
    def __call__(self, generated_paths: torch.Tensor, 
                 real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted scoring rule loss.
        
        Args:
            generated_paths: Generated paths
            real_paths: Real paths
            
        Returns:
            Weighted scoring rule loss
        """
        mu = generated_paths.clone().type(torch.float64)
        nu = real_paths.clone().type(torch.float64)
        
        if self._sigma is not None:
            mu[..., 1:] *= self._sigma
        
        loss = 0
        for scale in self._scalings:
            loss += self._scoring_rule(mu, nu[0], pi=torch.sqrt(scale))
        
        return loss / self.n_scalings

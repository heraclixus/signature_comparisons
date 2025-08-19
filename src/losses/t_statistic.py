"""
T-Statistic Loss Function (Extracted from deep_signature_transform)

This implements the T-statistic loss from Chevyrev & Oberhauser (2018) paper,
originally implemented in src/dataset/generative_model.py
"""

import torch
import numpy as np
from typing import Callable, Optional


def scalar_orders(dim: int, order: int):
    """The order of the scalar basis elements as one moves along the signature."""
    for i in range(order + 1):
        for _ in range(dim ** i):
            yield i


def psi(x: torch.Tensor, M: float = 4, a: float = 1) -> torch.Tensor:
    """Psi function, as defined in Chevyrev & Oberhauser (2018)."""
    if x <= M:
        return x
    return M + M ** (1 + a) * (M ** (-a) - x ** (-a)) / a


def normalise_instance(x: torch.Tensor, order: int) -> torch.Tensor:
    """Normalise signature instance using Newton-Raphson method."""
    x = torch.cat([torch.tensor([1.], device=x.device), x])

    a = x ** 2
    a[0] -= psi(torch.norm(x))
    
    x0 = 1.  # Starting point for Newton-Raphson
    
    moments = torch.tensor([x0 ** (2 * m) for m in range(len(x))], device=x.device)
    polx0 = torch.dot(a, moments)
    
    d_moments = torch.tensor([2 * m * x0 ** (2 * m - 1) for m in range(len(x))], device=x.device)
    d_polx0 = torch.dot(a, d_moments)
    x1 = x0 - polx0 / d_polx0

    if x1 < 0.2:
        x1 = 1.
    
    lambda_ = torch.tensor([x1 ** t for t in scalar_orders(2, order)], device=x.device)
    
    return lambda_ * x


def normalise(x: torch.Tensor, order: int) -> torch.Tensor:
    """Normalise signature batch."""
    return torch.stack([normalise_instance(sig, order) for sig in x])


class TStatisticLoss:
    """
    T-Statistic Loss Function for signature-based generation.
    
    Based on Chevyrev & Oberhauser (2018): "Signature moments to characterize 
    laws of stochastic processes."
    
    The loss compares signature distributions using:
    Loss = log(T1 - 2*T2 + T3)
    
    where:
    - T1 = mean(S_real ⊗ S_real)
    - T2 = mean(S_real ⊗ S_generated) 
    - T3 = mean(S_generated ⊗ S_generated)
    """
    
    def __init__(self, signature_transform: Callable, sig_depth: int = 4, 
                 normalise_sigs: bool = True):
        """
        Initialize T-Statistic loss.
        
        Args:
            signature_transform: Function that computes signatures from paths
            sig_depth: Depth of signature computation
            normalise_sigs: Whether to normalize signatures
        """
        self.signature_transform = signature_transform
        self.sig_depth = sig_depth
        self.normalise_sigs = normalise_sigs
        
        # Will be set when real data is provided
        self.orig_signatures = None
        self.T1 = None
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set the real data and precompute T1 term.
        
        Args:
            real_paths: Real path data, shape (batch, channels, length)
        """
        self.orig_signatures = self.signature_transform(real_paths)
        if self.normalise_sigs:
            self.orig_signatures = normalise(self.orig_signatures, self.sig_depth)
        
        self.T1 = torch.mean(torch.mm(self.orig_signatures, self.orig_signatures.t()))
    
    def __call__(self, generated_output: torch.Tensor, 
                 add_timeline: bool = True) -> torch.Tensor:
        """
        Compute T-statistic loss.
        
        Args:
            generated_output: Generated path values (without time dimension)
            add_timeline: Whether to add timeline coordinates
            
        Returns:
            T-statistic loss value
        """
        if self.orig_signatures is None:
            raise ValueError("Must call set_real_data() first")
        
        device = generated_output.device
        
        # Move precomputed terms to correct device
        T1 = self.T1.to(device)
        orig_signatures = self.orig_signatures.to(device)
        
        if add_timeline:
            # Add timeline coordinates (as in original implementation)
            timeline = torch.tensor(
                np.linspace(0, 1, generated_output.shape[1] + 1), 
                dtype=torch.float32, device=device
            )
            paths = torch.stack([
                torch.stack([timeline, torch.cat([torch.tensor([0.], device=device), path])])
                for path in generated_output
            ])
        else:
            paths = generated_output
        
        # Compute signatures for generated paths
        generated_sigs = self.signature_transform(paths)
        
        if self.normalise_sigs:
            generated_sigs = normalise(generated_sigs, self.sig_depth)
        
        # Compute T2 and T3 terms
        T2 = torch.mean(torch.mm(orig_signatures, generated_sigs.t()))
        T3 = torch.mean(torch.mm(generated_sigs, generated_sigs.t()))
        
        return torch.log(T1 - 2 * T2 + T3)
    
    def create_loss_function(self, real_paths: torch.Tensor) -> Callable:
        """
        Create a loss function closure with real data (original interface).
        
        Args:
            real_paths: Real path data
            
        Returns:
            Loss function that takes generated output and returns loss
        """
        self.set_real_data(real_paths)
        
        def loss_fn(output, *args):
            return self(output, add_timeline=True)
        
        return loss_fn

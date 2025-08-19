"""
Signature Kernel Implementation (Extracted from sigker_nsdes)

This provides signature kernel functionality originally implemented
in sigker_nsdes discriminators.
"""

import torch
from typing import Union, List, Any, Optional
import warnings

# Try to import sigkernel (may not be available in all environments)
try:
    import sigkernel
    SIGKERNEL_AVAILABLE = True
except ImportError:
    SIGKERNEL_AVAILABLE = False
    warnings.warn("sigkernel not available. Signature kernel functionality will be limited.")


class SignatureKernel:
    """
    Wrapper for signature kernel functionality.
    
    Provides a unified interface for signature kernel computations
    that can work with different backends (sigkernel, custom implementations).
    """
    
    def __init__(self, kernel_type: str = "rbf", dyadic_order: int = 4, 
                 sigma: float = 1.0):
        """
        Initialize signature kernel.
        
        Args:
            kernel_type: Type of static kernel ("rbf" or "linear")
            dyadic_order: Dyadic partitioning order for PDE solver
            sigma: RBF kernel bandwidth parameter
        """
        self.kernel_type = kernel_type.lower()
        self.dyadic_order = dyadic_order
        self.sigma = sigma
        
        if SIGKERNEL_AVAILABLE:
            self._kernel = self._init_sigkernel()
        else:
            self._kernel = None
            warnings.warn("sigkernel not available, using placeholder")
    
    def _init_sigkernel(self):
        """Initialize sigkernel backend."""
        if self.kernel_type == "rbf":
            static_kernel = sigkernel.RBFKernel(sigma=self.sigma)
        elif self.kernel_type == "linear":
            static_kernel = sigkernel.LinearKernel()
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        return sigkernel.SigKernel(
            static_kernel=static_kernel, 
            dyadic_order=self.dyadic_order
        )
    
    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor, 
                    max_batch: int = 128) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy between two sets of paths.
        
        Args:
            X: First set of paths, shape (batch, time, channels)
            Y: Second set of paths, shape (batch, time, channels)
            max_batch: Maximum batch size for computation
            
        Returns:
            MMD value
        """
        if self._kernel is None:
            raise RuntimeError("sigkernel not available")
        
        return self._kernel.compute_mmd(X, Y, max_batch=max_batch)
    
    def compute_scoring_rule(self, X: torch.Tensor, y: torch.Tensor,
                           max_batch: int = 128) -> torch.Tensor:
        """
        Compute scoring rule between generated samples and target.
        
        Args:
            X: Generated samples, shape (batch, time, channels)
            y: Target sample, shape (1, time, channels)
            max_batch: Maximum batch size for computation
            
        Returns:
            Scoring rule value
        """
        if self._kernel is None:
            raise RuntimeError("sigkernel not available")
        
        return self._kernel.compute_scoring_rule(X, y, max_batch=max_batch)
    
    def compute_Gram(self, X: torch.Tensor, Y: torch.Tensor, 
                     sym: bool = True, max_batch: int = 128) -> torch.Tensor:
        """
        Compute Gram matrix between two sets of paths.
        
        Args:
            X: First set of paths
            Y: Second set of paths
            sym: Whether matrix is symmetric
            max_batch: Maximum batch size
            
        Returns:
            Gram matrix
        """
        if self._kernel is None:
            raise RuntimeError("sigkernel not available")
        
        return self._kernel.compute_Gram(X, Y, sym=sym, max_batch=max_batch)


class SummedSignatureKernel:
    """
    Sum of multiple signature kernels for ensemble methods.
    """
    
    def __init__(self, kernels: List[SignatureKernel]):
        """
        Initialize summed kernel.
        
        Args:
            kernels: List of signature kernels to sum
        """
        self.kernels = kernels
        self.n_kernels = len(kernels)
    
    def compute_mmd(self, X: torch.Tensor, Y: torch.Tensor, 
                    max_batch: int = 128) -> torch.Tensor:
        """Compute average MMD across all kernels."""
        total = 0
        for kernel in self.kernels:
            total += kernel.compute_mmd(X, Y, max_batch=max_batch)
        return total / self.n_kernels
    
    def compute_scoring_rule(self, X: torch.Tensor, y: torch.Tensor,
                           max_batch: int = 128) -> torch.Tensor:
        """Compute average scoring rule across all kernels."""
        total = 0
        for kernel in self.kernels:
            total += kernel.compute_scoring_rule(X, y, max_batch=max_batch)
        return total / self.n_kernels


def get_signature_kernel(kernel_type: str = "rbf", dyadic_order: int = 4, 
                        sigma: float = 1.0) -> SignatureKernel:
    """
    Factory function for creating signature kernels.
    
    Args:
        kernel_type: Type of static kernel ("rbf" or "linear")
        dyadic_order: Dyadic partitioning order
        sigma: RBF kernel bandwidth
        
    Returns:
        SignatureKernel instance
    """
    return SignatureKernel(kernel_type, dyadic_order, sigma)


def get_multi_scale_kernel(kernel_type: str = "rbf", dyadic_order: int = 4,
                          sigmas: List[float] = [0.5, 1.0, 2.0]) -> SummedSignatureKernel:
    """
    Create multi-scale signature kernel with different bandwidths.
    
    Args:
        kernel_type: Type of static kernel
        dyadic_order: Dyadic partitioning order
        sigmas: List of bandwidth parameters
        
    Returns:
        SummedSignatureKernel with multiple scales
    """
    kernels = [
        SignatureKernel(kernel_type, dyadic_order, sigma) 
        for sigma in sigmas
    ]
    return SummedSignatureKernel(kernels)


# Placeholder implementations for when sigkernel is not available
class PlaceholderSignatureKernel:
    """Placeholder when sigkernel is not available."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def compute_mmd(self, X, Y, max_batch=128):
        raise NotImplementedError("sigkernel not available")
    
    def compute_scoring_rule(self, X, y, max_batch=128):
        raise NotImplementedError("sigkernel not available")
    
    def compute_Gram(self, X, Y, sym=True, max_batch=128):
        raise NotImplementedError("sigkernel not available")

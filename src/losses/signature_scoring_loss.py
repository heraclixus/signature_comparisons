"""
Signature Kernel Scoring Rule Loss for Distributional Diffusion.

This module implements the signature kernel-based scoring rule loss following
the method described in "Path Diffusion with Signature Kernels".
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional

# Import existing signature kernel infrastructure
try:
    from models.sigker_nsdes.src.gan.discriminators import (
        initialise_signature_kernel, 
        SigKerScoreDiscriminator
    )
    SIGKER_NSDES_AVAILABLE = True
except ImportError:
    SIGKER_NSDES_AVAILABLE = False
    warnings.warn(
        "sigker_nsdes not available. Some functionality may be limited."
    )

# Import signature kernel from existing infrastructure
try:
    from signatures.signature_kernels import get_signature_kernel
    SIGNATURE_KERNELS_AVAILABLE = True
except ImportError:
    SIGNATURE_KERNELS_AVAILABLE = False


class SignatureScoringLoss(nn.Module):
    """
    Signature kernel scoring rule loss following the paper:
    S_λ,sig(P, Y) = (λ/2) * E_{X,X'~P}[k_sig(X,X')] - E_{X~P}[k_sig(X, Y)]
    
    Empirical estimate:
    Ŝ_λ,sig = (λ/2) * (1/[m(m-1)]) * Σ_{i≠j} k_sig(X̃_0^(i), X̃_0^(j)) - (2/m) * Σ_i k_sig(X̃_0^(i), X_0)
    
    where:
    - X̃_0^(i) are samples from learned distribution P_θ(·|X_t, t)
    - X_0 is the real data sample
    - m is the population size
    - λ ∈ [0,1] is the generalized kernel score parameter
    """
    
    def __init__(
        self,
        signature_level: int = 4,
        lambda_param: float = 1.0,
        kernel_type: str = "rbf",
        dyadic_order: int = 4,
        sigma: float = 1.0,
        max_batch: int = 64
    ):
        """
        Initialize signature scoring loss.
        
        Args:
            signature_level: Signature truncation level (not used with sigkernel)
            lambda_param: λ parameter for generalized kernel score
            kernel_type: Type of static kernel ("rbf" or "linear")
            dyadic_order: Dyadic partitioning order for PDE solver
            sigma: RBF kernel bandwidth parameter
            max_batch: Maximum batch size for kernel computation
        """
        super().__init__()
        self.signature_level = signature_level
        self.lambda_param = lambda_param
        self.max_batch = max_batch
        
        # Initialize signature kernel using existing infrastructure
        self.sig_kernel = self._init_signature_kernel(
            kernel_type=kernel_type,
            dyadic_order=dyadic_order,
            sigma=sigma
        )
    
    def _init_signature_kernel(self, kernel_type: str, dyadic_order: int, sigma: float):
        """Initialize signature kernel using available infrastructure."""
        
        # Try sigker_nsdes infrastructure first
        if SIGKER_NSDES_AVAILABLE:
            try:
                return initialise_signature_kernel(
                    kernel_type=kernel_type,
                    dyadic_order=dyadic_order,
                    sigma=sigma
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize sigker_nsdes kernel: {e}")
        
        # Try existing signature_kernels infrastructure
        if SIGNATURE_KERNELS_AVAILABLE:
            try:
                return get_signature_kernel(
                    kernel_type=kernel_type,
                    dyadic_order=dyadic_order,
                    sigma=sigma
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize signature kernel: {e}")
        
        # Fallback error
        raise RuntimeError(
            "No signature kernel infrastructure available. "
            "Please ensure sigkernel package is installed and "
            "either sigker_nsdes or signature_kernels module is available."
        )
    
    def forward(self, generated_samples: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """
        Compute signature scoring rule loss following Equation (26) in the paper.
        
        Args:
            generated_samples: (batch_size, population_size, dim, seq_len)
            real_sample: (batch_size, dim, seq_len)
            
        Returns:
            loss: Signature scoring rule loss
        """
        batch_size, m, dim, seq_len = generated_samples.shape
        
        # Ensure we have enough samples for pairwise computation
        if m < 2:
            raise ValueError(f"Population size must be >= 2, got {m}")
        
        # Process in smaller chunks if batch is too large for memory efficiency
        if batch_size > self.max_batch:
            return self._compute_chunked_loss(generated_samples, real_sample)
        
        total_loss = 0.0
        
        for b in range(batch_size):
            # Get samples for this batch element
            gen_batch = generated_samples[b]  # (m, dim, seq_len)
            real_batch = real_sample[b:b+1]   # (1, dim, seq_len)
            
            # Convert to sigkernel format: (samples, time, channels)
            # sigkernel expects (batch, time, channels)
            gen_paths = gen_batch.transpose(1, 2).double()  # (m, seq_len, dim)
            real_path = real_batch.transpose(1, 2).double()  # (1, seq_len, dim)
            
            # Compute pairwise signature kernels between generated samples
            K_XX = self.sig_kernel.compute_Gram(
                gen_paths, gen_paths, sym=True, max_batch=min(self.max_batch, m)
            )
            
            # Self-similarity term: (λ/2) * (1/[m(m-1)]) * Σ_{i≠j} k_sig(X̃_0^(i), X̃_0^(j))
            # More efficient: total_sum - diagonal_sum instead of masking
            diagonal_sum = torch.diag(K_XX).sum()
            total_sum = K_XX.sum()
            off_diagonal_sum = total_sum - diagonal_sum
            self_sim = off_diagonal_sum / (m * (m - 1))
            
            # Cross-similarity term: (2/m) * Σ_i k_sig(X̃_0^(i), X_0)
            K_XY = self.sig_kernel.compute_Gram(
                gen_paths, real_path, sym=False, max_batch=min(self.max_batch, m)
            )
            cross_sim = K_XY.mean()
            
            # Signature scoring rule with λ parameter (following paper formulation)
            batch_loss = (self.lambda_param / 2.0) * self_sim - cross_sim
            total_loss += batch_loss
        
        return total_loss / batch_size
    
    def _compute_chunked_loss(self, generated_samples: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """Compute loss in chunks for large batches."""
        batch_size = generated_samples.shape[0]
        chunk_size = self.max_batch
        total_loss = 0.0
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_gen = generated_samples[i:end_idx]
            chunk_real = real_sample[i:end_idx]
            
            # Create a temporary loss instance to avoid recursion
            chunk_batch_size = end_idx - i
            chunk_loss = 0.0
            
            for b in range(chunk_batch_size):
                gen_batch = chunk_gen[b]
                real_batch = chunk_real[b:b+1]
                
                gen_paths = gen_batch.transpose(1, 2).double()
                real_path = real_batch.transpose(1, 2).double()
                
                m = gen_batch.shape[0]
                K_XX = self.sig_kernel.compute_Gram(gen_paths, gen_paths, sym=True, max_batch=min(self.max_batch, m))
                K_XY = self.sig_kernel.compute_Gram(gen_paths, real_path, sym=False, max_batch=min(self.max_batch, m))
                
                diagonal_sum = torch.diag(K_XX).sum()
                total_sum = K_XX.sum()
                self_sim = (total_sum - diagonal_sum) / (m * (m - 1))
                cross_sim = K_XY.mean()
                
                batch_loss = (self.lambda_param / 2.0) * self_sim - cross_sim
                chunk_loss += batch_loss
            
            total_loss += chunk_loss
        
        return total_loss / batch_size


class AdaptedSigKerScoreDiscriminator(nn.Module):
    """
    Alternative implementation that directly adapts SigKerScoreDiscriminator.
    This reuses the entire scoring rule infrastructure from sigker_nsdes.
    """
    
    def __init__(
        self, 
        kernel_type: str = "rbf", 
        dyadic_order: int = 4, 
        path_dim: int = 1,
        sigma: float = 1.0, 
        lambda_param: float = 1.0,
        max_batch: int = 64,
        **kwargs
    ):
        """
        Initialize adapted discriminator for distributional diffusion.
        
        Args:
            kernel_type: Type of static kernel
            dyadic_order: Dyadic partitioning order
            path_dim: Dimension of paths (excluding time)
            sigma: RBF kernel bandwidth
            lambda_param: λ parameter for generalized kernel score
            max_batch: Maximum batch size for computation
        """
        super().__init__()
        
        if not SIGKER_NSDES_AVAILABLE:
            raise ImportError(
                "SigKerScoreDiscriminator requires sigker_nsdes to be available"
            )
        
        self.lambda_param = lambda_param
        
        # Initialize base discriminator (non-adversarial)
        self.base_discriminator = SigKerScoreDiscriminator(
            kernel_type=kernel_type,
            dyadic_order=dyadic_order,
            path_dim=path_dim,
            sigma=sigma,
            adversarial=False,  # No adversarial scaling
            max_batch=max_batch,
            **kwargs
        )
    
    def compute_population_loss(
        self, 
        generated_samples: torch.Tensor, 
        real_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for population of generated samples vs real sample.
        
        Args:
            generated_samples: (batch_size, population_size, dim, seq_len)
            real_sample: (batch_size, dim, seq_len)
            
        Returns:
            loss: Population-based scoring rule loss
        """
        batch_size, m, dim, seq_len = generated_samples.shape
        total_loss = 0.0
        
        for b in range(batch_size):
            gen_batch = generated_samples[b]  # (m, dim, seq_len)
            real_batch = real_sample[b]       # (dim, seq_len)
            
            # Convert to expected format (batch, stream, channel)
            # Note: sigker_nsdes expects time as first channel
            gen_paths = gen_batch.transpose(1, 2)  # (m, seq_len, dim)
            real_path = real_batch.transpose(0, 1)  # (seq_len, dim)
            
            # Use existing scoring rule computation with λ scaling
            score = self.base_discriminator.forward(gen_paths, real_path)
            batch_loss = self.lambda_param * score
            total_loss += batch_loss
        
        return total_loss / batch_size
    
    def forward(self, generated_samples: torch.Tensor, real_sample: torch.Tensor) -> torch.Tensor:
        """Forward pass - alias for compute_population_loss."""
        return self.compute_population_loss(generated_samples, real_sample)


def create_signature_scoring_loss(
    method: str = "direct",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating signature scoring loss.
    
    Args:
        method: "direct" for SignatureScoringLoss, "adapted" for AdaptedSigKerScoreDiscriminator
        **kwargs: Arguments passed to the loss constructor
        
    Returns:
        Signature scoring loss module
    """
    if method == "direct":
        return SignatureScoringLoss(**kwargs)
    elif method == "adapted":
        return AdaptedSigKerScoreDiscriminator(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'direct' or 'adapted'.")

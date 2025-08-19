"""
Signature-based Discriminators for Adversarial Training

This module implements discriminators that use signature-based metrics to distinguish
between generated and real stochastic paths. These can be combined with existing
generators to create adversarial variants of our baseline models.

Key Features:
- Compatible with existing generators (CannedNet, Neural SDE)
- Uses signature-based metrics (MMD, Scoring Rule, T-Statistic)
- Optional learnable scaling parameters for adversarial training
- Memory-efficient implementations with fallbacks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from signatures.truncated import TruncatedSignature
from signatures.signature_kernels import get_signature_kernel, SIGKERNEL_AVAILABLE
from losses.t_statistic import TStatisticLoss
from losses.signature_mmd import SignatureMMDLoss
from losses.signature_scoring import SignatureScoringLoss


class AdversarialDiscriminatorBase(nn.Module):
    """
    Base class for adversarial discriminators.
    
    Provides common functionality for learnable scaling parameters
    and signature-based discrimination.
    """
    
    def __init__(self, path_dim: int = 1, adversarial: bool = True, 
                 initial_scaling: float = 1.0):
        """
        Initialize base adversarial discriminator.
        
        Args:
            path_dim: Dimension of path values (excluding time)
            adversarial: Whether to use learnable scaling parameters
            initial_scaling: Initial value for scaling parameters
        """
        super().__init__()
        
        self.path_dim = path_dim
        self.adversarial = adversarial
        
        if adversarial:
            # Learnable scaling parameters for path values
            self.path_scaling = nn.Parameter(
                torch.full((path_dim,), initial_scaling), 
                requires_grad=True
            )
        else:
            # Fixed scaling
            self.register_buffer('path_scaling', torch.ones(path_dim))
        
        print(f"{'Adversarial' if adversarial else 'Non-adversarial'} discriminator initialized")
    
    def apply_scaling(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Apply learned scaling to path values.
        
        Args:
            paths: Input paths, shape (batch, channels, time) or (batch, time)
            
        Returns:
            Scaled paths
        """
        if paths.dim() == 3:
            # Format: (batch, channels, time) - scale value channels only
            scaled_paths = paths.clone()
            scaled_paths[:, 1:, :] *= self.path_scaling.view(-1, 1)
            return scaled_paths
        elif paths.dim() == 2:
            # Format: (batch, time) - scale all values
            return paths * self.path_scaling[0]
        else:
            raise ValueError(f"Unsupported path tensor shape: {paths.shape}")
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """Get information about current scaling parameters."""
        if self.adversarial:
            return {
                'adversarial': True,
                'current_scaling': self.path_scaling.detach().cpu().numpy().tolist(),
                'scaling_grad': self.path_scaling.grad.detach().cpu().numpy().tolist() if self.path_scaling.grad is not None else None
            }
        else:
            return {
                'adversarial': False,
                'fixed_scaling': self.path_scaling.cpu().numpy().tolist()
            }


class SignatureMMDDiscriminator(AdversarialDiscriminatorBase):
    """
    MMD-based discriminator using signature features.
    
    This discriminator computes Maximum Mean Discrepancy between generated
    and real paths using signature transformations.
    """
    
    def __init__(self, path_dim: int = 1, adversarial: bool = True,
                 signature_depth: int = 4, kernel_type: str = 'rbf',
                 sigma: float = 1.0, use_sigkernel: bool = True):
        """
        Initialize signature MMD discriminator.
        
        Args:
            path_dim: Dimension of path values
            adversarial: Whether to use learnable scaling
            signature_depth: Depth of signature computation
            kernel_type: Type of kernel for MMD ('rbf' or 'linear')
            sigma: RBF kernel bandwidth
            use_sigkernel: Whether to use sigkernel package if available
        """
        super().__init__(path_dim, adversarial)
        
        self.signature_depth = signature_depth
        self.kernel_type = kernel_type
        self.sigma = sigma
        
        # Initialize signature transform
        self.signature_transform = TruncatedSignature(depth=signature_depth)
        
        # Initialize MMD loss function
        if use_sigkernel and SIGKERNEL_AVAILABLE:
            # Use signature kernel if available
            self.mmd_loss = self._create_sigkernel_mmd()
            self.implementation = "sigkernel"
        else:
            # Create a simple MMD loss using signature features
            self.mmd_loss = self._create_simple_mmd_loss()
            self.implementation = "fallback"
        
        print(f"MMD discriminator created using {self.implementation} implementation")
    
    def _create_sigkernel_mmd(self):
        """Create sigkernel-based MMD loss."""
        try:
            sig_kernel = get_signature_kernel(
                kernel_type=self.kernel_type,
                dyadic_order=6,  # Moderate precision for efficiency
                sigma=self.sigma
            )
            
            def sigkernel_mmd_loss(gen_paths, real_paths):
                # Convert to sigkernel format: (batch, time, channels)
                gen_sk = gen_paths.transpose(1, 2).double()
                real_sk = real_paths.transpose(1, 2).double()
                return sig_kernel.compute_mmd(gen_sk, real_sk, max_batch=64)
            
            return sigkernel_mmd_loss
            
        except Exception as e:
            warnings.warn(f"Failed to create sigkernel MMD: {e}. Using fallback.")
            return self._create_simple_mmd_loss()
    
    def forward(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD discriminator loss.
        
        Args:
            generated_paths: Generated paths (requires grad for adversarial training)
            real_paths: Real paths (no grad required)
            
        Returns:
            MMD loss value
        """
        # Apply adversarial scaling to generated paths only
        scaled_generated = self.apply_scaling(generated_paths)
        
        # Compute MMD loss
        return self.mmd_loss(scaled_generated, real_paths)
    
    def _create_simple_mmd_loss(self):
        """Create simple MMD loss using signature features."""
        def simple_mmd_loss(gen_paths, real_paths):
            # Compute signature features
            gen_sigs = self.signature_transform(gen_paths)
            real_sigs = self.signature_transform(real_paths)
            
            # Use subset to match batch sizes
            batch_size = min(gen_sigs.shape[0], real_sigs.shape[0])
            gen_sigs = gen_sigs[:batch_size]
            real_sigs = real_sigs[:batch_size]
            
            # Compute MMD using RBF kernel
            # MMD² = E[k(X,X)] - 2E[k(X,Y)] + E[k(Y,Y)]
            kxx = self._rbf_kernel(gen_sigs, gen_sigs)
            kxy = self._rbf_kernel(gen_sigs, real_sigs)
            kyy = self._rbf_kernel(real_sigs, real_sigs)
            
            mmd = torch.mean(kxx) - 2 * torch.mean(kxy) + torch.mean(kyy)
            return mmd
        
        return simple_mmd_loss
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        return torch.exp(-distances / (2 * self.sigma**2))


class SignatureScoringDiscriminator(AdversarialDiscriminatorBase):
    """
    Scoring rule-based discriminator using signature features.
    
    This discriminator uses the signature kernel scoring rule to measure
    the quality of generated paths against individual real samples.
    """
    
    def __init__(self, path_dim: int = 1, adversarial: bool = True,
                 signature_depth: int = 4, kernel_type: str = 'rbf',
                 sigma: float = 1.0, use_sigkernel: bool = True):
        """
        Initialize signature scoring discriminator.
        
        Args:
            path_dim: Dimension of path values
            adversarial: Whether to use learnable scaling
            signature_depth: Depth of signature computation
            kernel_type: Type of kernel ('rbf' or 'linear')
            sigma: RBF kernel bandwidth
            use_sigkernel: Whether to use sigkernel package if available
        """
        super().__init__(path_dim, adversarial)
        
        self.signature_depth = signature_depth
        self.kernel_type = kernel_type
        self.sigma = sigma
        
        # Initialize signature transform
        self.signature_transform = TruncatedSignature(depth=signature_depth)
        
        # Initialize scoring loss function
        if use_sigkernel and SIGKERNEL_AVAILABLE:
            self.scoring_loss = self._create_sigkernel_scoring()
            self.implementation = "sigkernel"
        else:
            # Create a simple scoring loss using signature features
            self.scoring_loss = self._create_simple_scoring_loss()
            self.implementation = "fallback"
        
        print(f"Scoring discriminator created using {self.implementation} implementation")
    
    def _create_sigkernel_scoring(self):
        """Create sigkernel-based scoring loss."""
        try:
            sig_kernel = get_signature_kernel(
                kernel_type=self.kernel_type,
                dyadic_order=6,
                sigma=self.sigma
            )
            
            def sigkernel_scoring_loss(gen_paths, real_paths):
                # Convert to sigkernel format: (batch, time, channels)
                gen_sk = gen_paths.transpose(1, 2).double()
                real_sk = real_paths.transpose(1, 2).double()
                
                # Use first real sample as reference
                real_ref = real_sk[:1]
                return sig_kernel.compute_scoring_rule(gen_sk, real_ref, max_batch=64)
            
            return sigkernel_scoring_loss
            
        except Exception as e:
            warnings.warn(f"Failed to create sigkernel scoring: {e}. Using fallback.")
            return self._create_simple_scoring_loss()
    
    def _create_simple_scoring_loss(self):
        """Create simple scoring loss using signature features."""
        def simple_scoring_loss(gen_paths, real_paths):
            # Compute signature features
            gen_sigs = self.signature_transform(gen_paths)
            real_sigs = self.signature_transform(real_paths)
            
            # Use subset of real signatures to match generated batch size
            batch_size = min(gen_sigs.shape[0], real_sigs.shape[0])
            gen_sigs = gen_sigs[:batch_size]
            real_sigs = real_sigs[:batch_size]
            
            # Simple RBF kernel similarities
            similarities = self._rbf_kernel(gen_sigs, real_sigs)
            
            # Scoring rule approximation
            score = -torch.log(similarities.mean() + 1e-8)
            return score
        
        return simple_scoring_loss
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        return torch.exp(-distances / (2 * self.sigma**2))
    
    def forward(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute scoring rule discriminator loss.
        
        Args:
            generated_paths: Generated paths (requires grad for adversarial training)
            real_paths: Real paths (no grad required)
            
        Returns:
            Scoring rule loss value
        """
        # Apply adversarial scaling to generated paths only
        scaled_generated = self.apply_scaling(generated_paths)
        
        # Compute scoring rule loss
        return self.scoring_loss(scaled_generated, real_paths)


class TStatisticDiscriminator(AdversarialDiscriminatorBase):
    """
    T-statistic based discriminator using signature features.
    
    This discriminator uses the T-statistic (Wasserstein-like) loss to measure
    distributional differences between generated and real paths.
    """
    
    def __init__(self, path_dim: int = 1, adversarial: bool = True,
                 signature_depth: int = 4, real_data: Optional[torch.Tensor] = None):
        """
        Initialize T-statistic discriminator.
        
        Args:
            path_dim: Dimension of path values
            adversarial: Whether to use learnable scaling
            signature_depth: Depth of signature computation
            real_data: Real data for T-statistic initialization (optional)
        """
        super().__init__(path_dim, adversarial)
        
        self.signature_depth = signature_depth
        
        # Initialize signature transform and T-statistic loss
        self.signature_transform = TruncatedSignature(depth=signature_depth)
        self.tstat_loss = TStatisticLoss(signature_transform=self.signature_transform)
        
        # Initialize with real data if provided
        if real_data is not None:
            self.set_real_data(real_data)
        
        print(f"T-statistic discriminator created with depth {signature_depth}")
    
    def set_real_data(self, real_data: torch.Tensor):
        """Set real data for T-statistic computation."""
        try:
            self.tstat_loss.set_real_data(real_data)
            print(f"T-statistic discriminator initialized with real data shape: {real_data.shape}")
        except Exception as e:
            print(f"⚠️ T-statistic initialization failed: {e}")
            print(f"   Creating new T-statistic loss with current signature depth")
            # Recreate T-statistic loss with correct depth
            self.signature_transform = TruncatedSignature(depth=self.signature_depth)
            self.tstat_loss = TStatisticLoss(signature_transform=self.signature_transform)
            self.tstat_loss.set_real_data(real_data)
            print(f"✅ T-statistic discriminator reinitialized successfully")
    
    def forward(self, generated_paths: torch.Tensor, real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute T-statistic discriminator loss.
        
        Args:
            generated_paths: Generated paths (requires grad for adversarial training)
            real_paths: Real paths (optional, uses pre-set data if None)
            
        Returns:
            T-statistic loss value
        """
        # Apply adversarial scaling to generated paths only
        scaled_generated = self.apply_scaling(generated_paths)
        
        # Extract just the values (remove time channel) for T-statistic
        if scaled_generated.dim() == 3 and scaled_generated.shape[1] == 2:
            # Format: (batch, channels, time) -> extract values only
            generated_values = scaled_generated[:, 1, :]  # Shape: (batch, time)
        else:
            generated_values = scaled_generated
        
        # Compute T-statistic loss (uses pre-set real data)
        return self.tstat_loss(generated_values, add_timeline=True)


def create_discriminator(discriminator_type: str, path_dim: int = 1, 
                        adversarial: bool = True, **kwargs) -> AdversarialDiscriminatorBase:
    """
    Factory function for creating discriminators.
    
    Args:
        discriminator_type: Type of discriminator ('mmd', 'scoring')
        path_dim: Dimension of path values
        adversarial: Whether to use learnable scaling
        **kwargs: Additional discriminator-specific parameters
        
    Returns:
        Initialized discriminator
        
    Note: 'tstatistic' discriminator removed due to compatibility issues with adversarial training
    """
    discriminator_type = discriminator_type.lower()
    
    if discriminator_type == 'mmd':
        return SignatureMMDDiscriminator(path_dim, adversarial, **kwargs)
    elif discriminator_type == 'scoring':
        return SignatureScoringDiscriminator(path_dim, adversarial, **kwargs)
    elif discriminator_type == 'tstatistic':
        raise ValueError(f"T-statistic discriminator not supported in adversarial training due to "
                        f"signature dimension compatibility issues. Use 'mmd' or 'scoring' instead.")
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}. Supported: 'mmd', 'scoring'")


# Test discriminator implementations
if __name__ == "__main__":
    print("Testing Signature Discriminators")
    print("=" * 40)
    
    # Create test data
    batch_size = 8
    time_steps = 50
    path_dim = 1
    
    generated_paths = torch.randn(batch_size, 2, time_steps, requires_grad=True)
    real_paths = torch.randn(batch_size, 2, time_steps)
    
    # Test each working discriminator type (T-statistic excluded)
    discriminator_types = ['mmd', 'scoring']
    
    for disc_type in discriminator_types:
        print(f"\n📊 Testing {disc_type.upper()} discriminator...")
        
        try:
            # Test both adversarial and non-adversarial
            for adversarial in [False, True]:
                print(f"  {'Adversarial' if adversarial else 'Non-adversarial'} mode:")
                
                discriminator = create_discriminator(
                    disc_type, path_dim, adversarial, 
                    signature_depth=3  # Small for testing
                )
                
                # Forward pass
                loss = discriminator(generated_paths, real_paths)
                print(f"    Loss: {loss.item():.6f}")
                print(f"    Requires grad: {loss.requires_grad}")
                
                # Test backward pass
                if loss.requires_grad:
                    loss.backward(retain_graph=True)
                    print(f"    ✅ Backward pass successful")
                    
                    if adversarial:
                        scaling_grad = discriminator.path_scaling.grad
                        print(f"    Scaling gradients: {scaling_grad}")
                
                # Clear gradients
                if generated_paths.grad is not None:
                    generated_paths.grad.zero_()
        
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    print(f"\n✅ Discriminator testing complete!")

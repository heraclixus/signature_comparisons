"""
C3 Model Implementation: Latent SDE + Signature MMD Loss

This implements a hybrid model that combines:
- V1 Latent SDE architecture for powerful generative modeling
- Signature MMD loss for distributional quality

Architecture:
- Generator: TorchSDE Latent SDE (same as V1)
- Loss: Hybrid ELBO + Signature MMD
- Training: Multi-objective optimization

The goal is to leverage latent SDE's generative power while enforcing
signature-based MMD constraints for better distributional matching.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Import base components
from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod

# Import existing latent SDE components (reuse V1)
from models.latent_sde.implementations.v1_latent_sde import TorchSDELatentSDE
from models.latent_sde.latent_sde_losses import ELBOLoss

# Import signature MMD loss components (reuse from A3, B4)
from models.deep_signature_transform import siglayer

try:
    import torchsde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    import warnings
    warnings.warn("torchsde not available. C3 model will not work.")


class SimplifiedMMDLoss:
    """
    Simplified MMD loss using signature features for C3 hybrid model.
    (Reused from A3, B4 implementations)
    """
    
    def __init__(self, signature_transform, real_paths: torch.Tensor, sigma: float = 1.0):
        """
        Initialize simplified MMD loss.
        
        Args:
            signature_transform: Signature transform to use
            real_paths: Real paths for MMD comparison
            sigma: RBF kernel bandwidth
        """
        self.signature_transform = signature_transform
        self.sigma = sigma
        
        # Precompute real signatures
        self.real_signatures = self.signature_transform(real_paths)
        
        print(f"Signature MMD loss initialized with sigma={sigma}")
    
    def __call__(self, generated_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified MMD using signature features.
        
        Args:
            generated_paths: Generated paths
            
        Returns:
            MMD loss
        """
        # Compute signature features for generated paths
        gen_sigs = self.signature_transform(generated_paths)
        
        # Compute RBF kernel MMD
        return self._rbf_mmd(gen_sigs, self.real_signatures)
    
    def _rbf_mmd(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute RBF kernel MMD between signature features.
        
        Args:
            X: Generated signature features
            Y: Real signature features
            
        Returns:
            MMD value
        """
        # Compute kernel matrices
        XX = self._rbf_kernel(X, X)
        YY = self._rbf_kernel(Y, Y)
        XY = self._rbf_kernel(X, Y)
        
        # MMD² = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
        mmd_squared = XX.mean() + YY.mean() - 2 * XY.mean()
        
        # Return MMD (not squared, and ensure positive)
        return torch.sqrt(torch.clamp(mmd_squared, min=1e-8))
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        # Compute pairwise distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        # RBF kernel
        return torch.exp(-distances / (2 * self.sigma**2))


class C3Model(BaseSignatureModel):
    """
    C3: Hybrid Latent SDE + Signature MMD Model
    
    Combines:
    - V1 Latent SDE architecture for generation
    - Signature MMD loss for distributional quality
    - Hybrid training with weighted multi-objective loss
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5,
                 hidden_size: int = 200, sig_depth: int = 4,
                 elbo_weight: float = 1.0, mmd_weight: float = 55.0,
                 mmd_sigma: float = 1.0):
        """
        Initialize C3 hybrid model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for loss initialization
            theta: OU process mean reversion rate
            mu: OU process long-term mean
            sigma: OU process volatility
            hidden_size: Hidden layer size for latent SDE
            sig_depth: Signature depth for MMD
            elbo_weight: Weight for ELBO loss term
            mmd_weight: Weight for signature MMD loss term
            mmd_sigma: RBF kernel bandwidth for MMD
        """
        # Create model configuration
        config = ModelConfig(
            model_id="C3",
            name="Hybrid Latent SDE + Signature MMD",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_MMD,
            signature_method=SignatureMethod.TRUNCATED,
            description="V1 Latent SDE + Signature MMD loss hybrid"
        )
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde is required for C3 model")
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.hidden_size = hidden_size
        self.sig_depth = sig_depth
        self.elbo_weight = elbo_weight
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma
        self.real_data = real_data  # Store for loss initialization
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        # Initialize losses after model is built
        self._initialize_losses(real_data)
        
        print(f"C3 Hybrid model initialized:")
        print(f"   OU parameters: θ={theta}, μ={mu}, σ={sigma}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Signature depth: {sig_depth}")
        print(f"   Loss weights: ELBO={elbo_weight}, MMD={mmd_weight}")
        print(f"   MMD sigma: {mmd_sigma}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """Build model components using existing implementations."""
        # Create latent SDE model (reuse V1 implementation)
        self.latent_sde = TorchSDELatentSDE(
            theta=self.theta,
            mu=self.mu,
            sigma=self.sigma,
            hidden_size=self.hidden_size
        )
        
        # Create signature transform (reuse existing)
        self.signature_transform = siglayer.Signature(self.sig_depth)
        
        print(f"✅ C3 model components built successfully")
    
    def _initialize_losses(self, real_data: torch.Tensor):
        """Initialize both ELBO and signature MMD losses."""
        # Initialize ELBO loss (reuse existing)
        self.elbo_loss = ELBOLoss(
            reconstruction_loss="mse",
            kl_weight=1.0,
            num_mc_samples=1
        )
        
        # Initialize signature MMD loss (reuse from A3, B4)
        self.mmd_loss = SimplifiedMMDLoss(
            signature_transform=self.signature_transform,
            real_paths=real_data,
            sigma=self.mmd_sigma
        )
        
        print(f"✅ C3 hybrid losses initialized")
        print(f"   ELBO loss: {self.elbo_loss.loss_type}")
        print(f"   Signature MMD: depth={self.sig_depth}, sigma={self.mmd_sigma}")
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for generation."""
        batch_size = batch.size(0) if batch.dim() > 1 else 1
        time_steps = batch.size(2) if batch.dim() == 3 else 64  # Use new default
        
        return self.generate_samples(batch_size, time_steps)
    
    def generate_samples(self, batch_size: int, time_steps: int = 64) -> torch.Tensor:
        """
        Generate samples using the latent SDE.
        
        Args:
            batch_size: Number of samples to generate
            time_steps: Number of time points
            
        Returns:
            Generated paths, shape (batch, 2, time_steps) - [time, value]
        """
        self.eval()
        
        with torch.no_grad():
            # Create time grid
            ts = torch.linspace(0, 1.0, time_steps)
            
            # Sample from posterior using latent SDE
            ys = self.latent_sde.sample_posterior(ts, batch_size)  # (time_steps, batch, 1)
            
            # Format output to match our interface
            # Convert from (time_steps, batch, 1) to (batch, 2, time_steps)
            time_channel = ts.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            value_channel = ys.squeeze(-1).t().unsqueeze(1)  # (batch, 1, time_steps)
            
            output = torch.cat([time_channel, value_channel], dim=1)
            
            return output
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute hybrid ELBO + signature MMD loss.
        
        Args:
            generated_output: Generated paths (used for MMD)
            real_paths: Real paths (used for ELBO)
            
        Returns:
            Combined loss tensor
        """
        if real_paths is None:
            raise ValueError("C3 model requires real_paths for ELBO computation")
        
        # Compute ELBO loss using V1's approach
        batch_size = real_paths.size(0)
        time_steps = real_paths.size(2)
        
        # Extract observations (remove time channel)
        observations = real_paths[:, 1, :]  # (batch, time_steps)
        
        # Create time grid
        ts = torch.linspace(0, 1.0, time_steps, device=real_paths.device)
        
        # Forward pass through latent SDE for ELBO
        ys, kl = self.latent_sde(ts, batch_size)  # ys: (time_steps, batch, 1)
        ys = ys.squeeze(-1)  # (time_steps, batch)
        
        # Likelihood: p(observations | latent_trajectories)
        likelihood = torch.distributions.Laplace(loc=ys.t(), scale=0.1)  # (batch, time_steps)
        log_likelihood = likelihood.log_prob(observations).sum(dim=1).mean(dim=0)
        
        # ELBO loss = -log_likelihood + KL
        elbo_loss = -log_likelihood + kl
        
        # Compute signature MMD loss on generated output
        mmd_loss = self.mmd_loss(generated_output)
        
        # Combine losses with weights
        total_loss = (self.elbo_weight * elbo_loss + 
                     self.mmd_weight * mmd_loss)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'elbo_loss': elbo_loss.item(),
            'mmd_loss': mmd_loss.item(),
            'total_loss': total_loss.item(),
            'elbo_weight': self.elbo_weight,
            'mmd_weight': self.mmd_weight,
            'kl_divergence': kl.item(),
            'log_likelihood': log_likelihood.item()
        }
        
        return total_loss
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get detailed breakdown of loss components."""
        return getattr(self, 'last_loss_components', {})
    
    def compute_training_loss(self, batch: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss for a batch.
        
        Args:
            batch: Input batch (used for batch size)
            real_paths: Real path data
            
        Returns:
            Loss for backpropagation
        """
        self.train()
        
        # Generate samples using latent SDE
        generated = self.forward(batch)
        
        # Compute hybrid loss
        loss = self.compute_loss(generated, real_paths)
        
        return loss


def create_c3_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> C3Model:
    """
    Factory function to create C3 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for loss initialization
        **kwargs: Additional model parameters
        
    Returns:
        Initialized C3 model
    """
    # Default parameters optimized for hybrid training
    default_params = {
        'theta': 2.0,           # OU mean reversion
        'mu': 0.0,              # OU long-term mean
        'sigma': 0.5,           # OU volatility
        'hidden_size': 200,     # Latent SDE hidden size
        'sig_depth': 4,         # Signature depth
        'elbo_weight': 1.0,     # ELBO loss weight
        'mmd_weight': 55.0,     # Signature MMD weight (~10% contribution for strong signature constraint)
        'mmd_sigma': 1.0        # RBF kernel bandwidth
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating C3 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return C3Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# C3 will be registered automatically when imported

"""
C1 Model Implementation: Latent SDE + T-Statistic Loss

This implements a hybrid model that combines:
- V1 Latent SDE architecture for powerful generative modeling
- T-Statistic signature loss for distributional quality

Architecture:
- Generator: TorchSDE Latent SDE (same as V1)
- Loss: Hybrid ELBO + T-Statistic
- Training: Multi-objective optimization

The goal is to leverage latent SDE's generative power while enforcing
signature-based distributional constraints for better path quality.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# Import base components
from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod

# Import existing latent SDE components (reuse V1)
from models.latent_sde.implementations.v1_latent_sde import TorchSDELatentSDE
from models.latent_sde.latent_sde_losses import ELBOLoss

# Import signature loss components (reuse from existing models)
from losses.t_statistic import TStatisticLoss
from models.deep_signature_transform import siglayer

try:
    import torchsde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    import warnings
    warnings.warn("torchsde not available. C1 model will not work.")


class C1Model(BaseSignatureModel):
    """
    C1: Hybrid Latent SDE + T-Statistic Model
    
    Combines:
    - V1 Latent SDE architecture for generation
    - T-Statistic signature loss for distributional quality
    - Hybrid training with weighted multi-objective loss
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5,
                 hidden_size: int = 200, sig_depth: int = 4,
                 elbo_weight: float = 1.0, tstat_weight: float = 0.2,
                 normalise_sigs: bool = True):
        """
        Initialize C1 hybrid model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for loss initialization
            theta: OU process mean reversion rate
            mu: OU process long-term mean
            sigma: OU process volatility
            hidden_size: Hidden layer size for latent SDE
            sig_depth: Signature depth for T-statistic
            elbo_weight: Weight for ELBO loss term
            tstat_weight: Weight for T-statistic loss term
            normalise_sigs: Whether to normalize signatures
        """
        # Create model configuration
        config = ModelConfig(
            model_id="C1",
            name="Hybrid Latent SDE + T-Statistic",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.T_STATISTIC,
            signature_method=SignatureMethod.TRUNCATED,
            description="V1 Latent SDE + T-Statistic signature loss hybrid"
        )
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde is required for C1 model")
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.hidden_size = hidden_size
        self.sig_depth = sig_depth
        self.elbo_weight = elbo_weight
        self.tstat_weight = tstat_weight
        self.normalise_sigs = normalise_sigs
        self.real_data = real_data  # Store for loss initialization
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        # Initialize losses after model is built
        self._initialize_losses(real_data)
        
        print(f"C1 Hybrid model initialized:")
        print(f"   OU parameters: θ={theta}, μ={mu}, σ={sigma}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Signature depth: {sig_depth}")
        print(f"   Loss weights: ELBO={elbo_weight}, T-Stat={tstat_weight}")
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
        
        print(f"✅ C1 model components built successfully")
    
    def _initialize_losses(self, real_data: torch.Tensor):
        """Initialize both ELBO and T-statistic losses."""
        # Initialize ELBO loss (reuse existing)
        self.elbo_loss = ELBOLoss(
            reconstruction_loss="mse",
            kl_weight=1.0,
            num_mc_samples=1
        )
        
        # Initialize T-statistic loss (reuse existing)
        self.tstat_loss = TStatisticLoss(
            signature_transform=self.signature_transform,
            sig_depth=self.sig_depth,
            normalise_sigs=self.normalise_sigs
        )
        
        # Set real data for T-statistic computation
        self.tstat_loss.set_real_data(real_data)
        
        print(f"✅ C1 hybrid losses initialized")
        print(f"   ELBO loss: {self.elbo_loss.loss_type}")
        print(f"   T-statistic: depth={self.sig_depth}, normalize={self.normalise_sigs}")
    
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
        Compute hybrid ELBO + T-statistic loss.
        
        Args:
            generated_output: Generated paths (used for T-statistic)
            real_paths: Real paths (used for ELBO computation)
            
        Returns:
            Combined loss tensor
        """
        if real_paths is None:
            raise ValueError("C1 model requires real_paths for ELBO computation")
        
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
        
        # Compute T-statistic loss on generated output
        # Extract values (remove time channel) for T-statistic
        if generated_output.dim() == 3 and generated_output.shape[1] == 2:
            generated_values = generated_output[:, 1, :]  # Shape: (batch, time)
        else:
            generated_values = generated_output
        
        tstat_loss = self.tstat_loss(generated_values, add_timeline=True)
        
        # Combine losses with weights
        total_loss = (self.elbo_weight * elbo_loss + 
                     self.tstat_weight * tstat_loss)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'elbo_loss': elbo_loss.item(),
            'tstat_loss': tstat_loss.item(),
            'total_loss': total_loss.item(),
            'elbo_weight': self.elbo_weight,
            'tstat_weight': self.tstat_weight,
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


def create_c1_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> C1Model:
    """
    Factory function to create C1 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for loss initialization
        **kwargs: Additional model parameters
        
    Returns:
        Initialized C1 model
    """
    # Default parameters optimized for hybrid training
    default_params = {
        'theta': 2.0,           # OU mean reversion
        'mu': 0.0,              # OU long-term mean
        'sigma': 0.5,           # OU volatility
        'hidden_size': 200,     # Latent SDE hidden size
        'sig_depth': 4,         # Signature depth
        'elbo_weight': 1.0,     # ELBO loss weight
        'tstat_weight': 0.2,    # T-statistic loss weight (~0.3% contribution, meaningful regularization)
        'normalise_sigs': True  # Normalize signatures
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating C1 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return C1Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# C1 will be registered automatically when imported
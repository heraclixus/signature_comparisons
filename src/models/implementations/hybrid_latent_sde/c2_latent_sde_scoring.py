"""
C2 Model Implementation: Latent SDE + Signature Scoring Loss

This implements a hybrid model that combines:
- V1 Latent SDE architecture for powerful generative modeling
- Signature Scoring loss for distributional quality

Architecture:
- Generator: TorchSDE Latent SDE (same as V1)
- Loss: Hybrid ELBO + Signature Scoring
- Training: Multi-objective optimization

The goal is to leverage latent SDE's generative power while enforcing
signature-based scoring rule constraints for better distributional matching.
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

# Import signature scoring loss components (reuse from B5)
from models.deep_signature_transform import siglayer

try:
    import torchsde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    import warnings
    warnings.warn("torchsde not available. C2 model will not work.")


class SimplifiedScoringLoss:
    """
    Simplified signature scoring loss for C2 hybrid model.
    (Reused from B5 implementation)
    """
    
    def __init__(self, signature_transform, real_paths: torch.Tensor, sigma: float = 1.0, device: torch.device = None):
        """
        Initialize simplified scoring loss.
        
        Args:
            signature_transform: Signature transform to use
            real_paths: Real paths for scoring comparison
            sigma: RBF kernel bandwidth
            device: Device to use for computations
        """
        self.signature_transform = signature_transform
        self.sigma = sigma
        self.device = device or torch.device('cpu')
        
        # Move real_paths to device and precompute real signatures
        real_paths = real_paths.to(self.device)
        self.real_signatures = self.signature_transform(real_paths).to(self.device)
        
        print(f"Signature scoring loss initialized with sigma={sigma}, device={self.device}")
    
    def __call__(self, generated_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified scoring loss using signature features.
        
        Args:
            generated_paths: Generated paths
            
        Returns:
            Scoring loss
        """
        # Move generated paths to device and compute signature features
        generated_paths = generated_paths.to(self.device)
        gen_sigs = self.signature_transform(generated_paths).to(self.device)
        
        # Simplified scoring rule: negative log-likelihood approximation
        # Using RBF kernel similarity
        similarities = self._rbf_kernel(gen_sigs, self.real_signatures)
        
        # Score based on similarity to real signatures
        score = -torch.log(similarities.mean() + 1e-8)
        
        return score
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        # Ensure both tensors are on the same device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute pairwise distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        # RBF kernel
        return torch.exp(-distances / (2 * self.sigma**2))


class C2Model(BaseSignatureModel):
    """
    C2: Hybrid Latent SDE + Signature Scoring Model
    
    Combines:
    - V1 Latent SDE architecture for generation
    - Signature Scoring loss for distributional quality
    - Hybrid training with weighted multi-objective loss
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5,
                 hidden_size: int = 200, sig_depth: int = 4,
                 elbo_weight: float = 1.0, scoring_weight: float = 2.8,
                 scoring_sigma: float = 1.0):
        """
        Initialize C2 hybrid model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for loss initialization
            theta: OU process mean reversion rate
            mu: OU process long-term mean
            sigma: OU process volatility
            hidden_size: Hidden layer size for latent SDE
            sig_depth: Signature depth for scoring
            elbo_weight: Weight for ELBO loss term
            scoring_weight: Weight for signature scoring loss term
            scoring_sigma: RBF kernel bandwidth for scoring
        """
        # Create model configuration
        config = ModelConfig(
            model_id="C2",
            name="Hybrid Latent SDE + Signature Scoring",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_SCORING,
            signature_method=SignatureMethod.TRUNCATED,
            description="V1 Latent SDE + Signature Scoring loss hybrid"
        )
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde is required for C2 model")
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.hidden_size = hidden_size
        self.sig_depth = sig_depth
        self.elbo_weight = elbo_weight
        self.scoring_weight = scoring_weight
        self.scoring_sigma = scoring_sigma
        self.real_data = real_data  # Store for loss initialization
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        # Initialize losses after model is built
        self._initialize_losses(real_data)
        
        print(f"C2 Hybrid model initialized:")
        print(f"   OU parameters: θ={theta}, μ={mu}, σ={sigma}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Signature depth: {sig_depth}")
        print(f"   Loss weights: ELBO={elbo_weight}, Scoring={scoring_weight}")
        print(f"   Scoring sigma: {scoring_sigma}")
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
        
        print(f"✅ C2 model components built successfully")
    
    def to(self, device):
        """Override to() method to ensure all components are moved to device."""
        # Move base model
        super().to(device)
        
        # Move latent SDE component
        if hasattr(self, 'latent_sde') and self.latent_sde is not None:
            self.latent_sde = self.latent_sde.to(device)
        
        # Update device tracking
        self.device = device
        
        return self
    
    def _initialize_losses(self, real_data: torch.Tensor):
        """Initialize both ELBO and signature scoring losses."""
        # Initialize ELBO loss (reuse existing)
        self.elbo_loss = ELBOLoss(
            reconstruction_loss="mse",
            kl_weight=1.0,
            num_mc_samples=1
        )
        
        # Initialize signature scoring loss (reuse from B5)
        self.scoring_loss = SimplifiedScoringLoss(
            signature_transform=self.signature_transform,
            real_paths=real_data,
            sigma=self.scoring_sigma,
            device=self.device
        )
        
        print(f"✅ C2 hybrid losses initialized")
        print(f"   ELBO loss: {self.elbo_loss.loss_type}")
        print(f"   Signature scoring: depth={self.sig_depth}, sigma={self.scoring_sigma}")
    
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
            ts = torch.linspace(0, 1.0, time_steps, device=self.device)
            
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
        Compute hybrid ELBO + signature scoring loss.
        
        Args:
            generated_output: Generated paths (used for scoring)
            real_paths: Real paths (used for ELBO)
            
        Returns:
            Combined loss tensor
        """
        if real_paths is None:
            raise ValueError("C2 model requires real_paths for ELBO computation")
        
        # Compute ELBO loss using V1's approach
        batch_size = real_paths.size(0)
        time_steps = real_paths.size(2)
        
        # Extract observations (remove time channel)
        observations = real_paths[:, 1, :]  # (batch, time_steps)
        
        # Create time grid
        ts = torch.linspace(0, 1.0, time_steps, device=self.device)
        
        # Forward pass through latent SDE for ELBO
        ys, kl = self.latent_sde(ts, batch_size)  # ys: (time_steps, batch, 1)
        ys = ys.squeeze(-1)  # (time_steps, batch)
        
        # Likelihood: p(observations | latent_trajectories)
        likelihood = torch.distributions.Laplace(loc=ys.t(), scale=0.1)  # (batch, time_steps)
        log_likelihood = likelihood.log_prob(observations).sum(dim=1).mean(dim=0)
        
        # ELBO loss = -log_likelihood + KL
        elbo_loss = -log_likelihood + kl
        
        # Compute signature scoring loss on generated output
        scoring_loss = self.scoring_loss(generated_output)
        
        # Combine losses with weights
        total_loss = (self.elbo_weight * elbo_loss + 
                     self.scoring_weight * scoring_loss)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'elbo_loss': elbo_loss.item(),
            'scoring_loss': scoring_loss.item(),
            'total_loss': total_loss.item(),
            'elbo_weight': self.elbo_weight,
            'scoring_weight': self.scoring_weight,
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


def create_c2_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> C2Model:
    """
    Factory function to create C2 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for loss initialization
        **kwargs: Additional model parameters
        
    Returns:
        Initialized C2 model
    """
    # Default parameters optimized for hybrid training
    default_params = {
        'theta': 2.0,           # OU mean reversion
        'mu': 0.0,              # OU long-term mean
        'sigma': 0.5,           # OU volatility
        'hidden_size': 200,     # Latent SDE hidden size
        'sig_depth': 4,         # Signature depth
        'elbo_weight': 1.0,     # ELBO loss weight
        'scoring_weight': 2.8,  # Signature scoring weight (~10% contribution for strong signature constraint)
        'scoring_sigma': 1.0    # RBF kernel bandwidth
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating C2 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return C2Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# C2 will be registered automatically when imported

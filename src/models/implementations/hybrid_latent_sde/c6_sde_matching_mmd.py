"""
C6 Model Implementation: SDE Matching + Signature MMD Loss

This implements a hybrid model that combines:
- V2 SDE Matching architecture for powerful generative modeling
- Signature MMD loss for distributional quality

Architecture:
- Generator: V2 SDE Matching (Prior + Posterior networks)
- Loss: Hybrid SDE Matching + Signature MMD
- Training: Multi-objective optimization

The goal is to leverage SDE Matching's generative power while enforcing
signature-based MMD constraints for better distributional matching.
This completes the C-series hybrid model collection.
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

# Import existing SDE Matching components (reuse V2)
from models.sdematching.implementations.v2_sde_matching import solve_sde
from models.sdematching.matching_network import MatchingSDE
from models.sdematching.prior_network import PriorInitDistribution, PriorSDE, PriorObservation
from models.sdematching.posterior_network import PosteriorEncoder, PosteriorAffine

# Import signature MMD loss components (reuse from C3)
from models.deep_signature_transform import siglayer


class SimplifiedMMDLoss:
    """
    Simplified MMD loss using signature features for C6 hybrid model.
    (Reused from C3 implementation)
    """
    
    def __init__(self, signature_transform, real_paths: torch.Tensor, sigma: float = 1.0, device: torch.device = None):
        """
        Initialize simplified MMD loss.
        
        Args:
            signature_transform: Signature transform to use
            real_paths: Real paths for MMD comparison
            sigma: RBF kernel bandwidth
            device: Device to use for computations
        """
        self.signature_transform = signature_transform
        self.sigma = sigma
        self.device = device or torch.device('cpu')
        
        # Move real_paths to device and precompute real signatures
        real_paths = real_paths.to(self.device)
        self.real_signatures = self.signature_transform(real_paths).to(self.device)
        
        print(f"Signature MMD loss initialized with sigma={sigma}, device={self.device}")
    
    def __call__(self, generated_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified MMD using signature features.
        
        Args:
            generated_paths: Generated paths
            
        Returns:
            MMD loss
        """
        # Move generated paths to device and compute signature features
        generated_paths = generated_paths.to(self.device)
        gen_sigs = self.signature_transform(generated_paths).to(self.device)
        
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
        # Ensure both tensors are on the same device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute pairwise distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        # RBF kernel
        return torch.exp(-distances / (2 * self.sigma**2))


class C6Model(BaseSignatureModel):
    """
    C6: Hybrid SDE Matching + Signature MMD Model
    
    Combines:
    - V2 SDE Matching architecture for generation
    - Signature MMD loss for distributional quality
    - Hybrid training with weighted multi-objective loss
    
    This completes the C-series hybrid model collection.
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 data_size: int = 1, latent_size: int = 4, hidden_size: int = 64,
                 noise_std: float = 0.1, sig_depth: int = 4,
                 sde_weight: float = 1.0, mmd_weight: float = 7.6,
                 mmd_sigma: float = 1.0):
        """
        Initialize C6 hybrid model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for loss initialization
            data_size: Observable data dimension
            latent_size: Latent state dimension
            hidden_size: Hidden layer size
            noise_std: Observation noise standard deviation
            sig_depth: Signature depth for MMD
            sde_weight: Weight for SDE matching loss term
            mmd_weight: Weight for signature MMD loss term
            mmd_sigma: RBF kernel bandwidth for MMD
        """
        # Create model configuration
        config = ModelConfig(
            model_id="C6",
            name="Hybrid SDE Matching + Signature MMD",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_MMD,
            signature_method=SignatureMethod.TRUNCATED,
            description="V2 SDE Matching + Signature MMD loss hybrid - completes C-series"
        )
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.data_size = data_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.sig_depth = sig_depth
        self.sde_weight = sde_weight
        self.mmd_weight = mmd_weight
        self.mmd_sigma = mmd_sigma
        self.real_data = real_data  # Store for loss initialization
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        # Initialize losses after model is built
        self._initialize_losses(real_data)
        
        print(f"C6 Hybrid model initialized:")
        print(f"   Data size: {data_size}")
        print(f"   Latent size: {latent_size}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Signature depth: {sig_depth}")
        print(f"   Loss weights: SDE={sde_weight}, MMD={mmd_weight}")
        print(f"   MMD sigma: {mmd_sigma}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """Build model components using existing implementations."""
        # Create SDE Matching components (reuse V2 implementation)
        self.p_init_distr = PriorInitDistribution(self.latent_size)
        self.p_sde = PriorSDE(self.latent_size, self.hidden_size)
        self.p_observe = PriorObservation(self.latent_size, self.data_size, self.noise_std)
        
        self.q_enc = PosteriorEncoder(self.data_size, self.hidden_size)
        self.q_affine = PosteriorAffine(self.latent_size, self.hidden_size)
        
        # Create matching SDE
        self.sde_matching = MatchingSDE(
            self.p_init_distr,
            self.p_sde,
            self.p_observe,
            self.q_enc,
            self.q_affine
        )
        
        # Create signature transform (reuse existing)
        self.signature_transform = siglayer.Signature(self.sig_depth)
        
        print(f"✅ C6 model components built successfully")
    
    def to(self, device):
        """Override to() method to ensure all components are moved to device."""
        # Move base model
        super().to(device)
        
        # Move SDE matching components
        if hasattr(self, 'sde_matching') and self.sde_matching is not None:
            self.sde_matching = self.sde_matching.to(device)
        if hasattr(self, 'p_sde') and self.p_sde is not None:
            self.p_sde = self.p_sde.to(device)
        if hasattr(self, 'p_observe') and self.p_observe is not None:
            self.p_observe = self.p_observe.to(device)
        
        # Update device tracking
        self.device = device
        
        return self
    
    def _initialize_losses(self, real_data: torch.Tensor):
        """Initialize signature MMD loss."""
        # Initialize signature MMD loss (reuse from C3)
        self.mmd_loss = SimplifiedMMDLoss(
            signature_transform=self.signature_transform,
            real_paths=real_data,
            sigma=self.mmd_sigma,
            device=self.device
        )
        
        print(f"✅ C6 hybrid losses initialized")
        print(f"   Signature MMD: depth={self.sig_depth}, sigma={self.mmd_sigma}")
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for generation."""
        batch_size = batch.size(0) if batch.dim() > 1 else 1
        time_steps = batch.size(2) if batch.dim() == 3 else 64  # Use new default
        
        return self.generate_samples(batch_size, time_steps)
    
    def generate_samples(self, batch_size: int, time_steps: int = 64, 
                        T: float = 1.0) -> torch.Tensor:
        """
        Generate samples using SDE Matching approach.
        
        Args:
            batch_size: Number of samples to generate
            time_steps: Number of time points
            T: Final time
            
        Returns:
            Generated paths, shape (batch, 2, time_steps) - [time, value]
        """
        self.eval()
        
        with torch.no_grad():
            # 1. Sample initial latent state from prior
            z0 = self.p_init_distr().rsample([batch_size])[:, 0]  # (batch, latent_size)
            
            # 2. Solve prior SDE in latent space
            zs = solve_sde(self.p_sde, z0, 0., T, n_steps=time_steps-1)  # (time_steps, batch, latent_size)
            
            # 3. Map latent trajectories to observations
            zs_flat = zs.reshape(-1, self.latent_size)  # (time_steps * batch, latent_size)
            xs_flat, _ = self.p_observe.get_coeffs(zs_flat)  # (time_steps * batch, data_size)
            xs = xs_flat.reshape(time_steps, batch_size, self.data_size)  # (time_steps, batch, data_size)
            
            # 4. Format output to match our interface
            # Convert from (time_steps, batch, data_size) to (batch, 2, time_steps)
            ts = torch.linspace(0, T, time_steps, device=self.device)
            time_channel = ts.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            value_channel = xs.permute(1, 2, 0)  # (batch, data_size, time_steps)
            
            output = torch.cat([time_channel, value_channel], dim=1)
            
            return output
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute hybrid SDE matching + signature MMD loss.
        
        Args:
            generated_output: Generated paths (used for MMD)
            real_paths: Real paths (used for SDE matching)
            
        Returns:
            Combined loss tensor
        """
        if real_paths is None:
            raise ValueError("C6 model requires real_paths for SDE matching computation")
        
        # Compute SDE matching loss using V2's approach
        # Format data for SDE matching
        # Convert from (batch, 2, time_steps) to (batch, time_steps, data_size)
        xs = real_paths[:, 1:, :].permute(0, 2, 1)  # Extract values and transpose
        
        # Create time grid
        batch_size = real_paths.size(0)
        time_steps = real_paths.size(2)
        ts = torch.linspace(0, 1.0, time_steps, device=self.device)
        ts = ts.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch, time_steps, 1)
        
        # Compute SDE matching loss
        sde_loss = self.sde_matching(xs, ts).mean()
        
        # Compute signature MMD loss on generated output
        mmd_loss = self.mmd_loss(generated_output)
        
        # Combine losses with weights
        total_loss = (self.sde_weight * sde_loss + 
                     self.mmd_weight * mmd_loss)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'sde_loss': sde_loss.item(),
            'mmd_loss': mmd_loss.item(),
            'total_loss': total_loss.item(),
            'sde_weight': self.sde_weight,
            'mmd_weight': self.mmd_weight
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
        
        # Generate samples using SDE matching
        generated = self.forward(batch)
        
        # Compute hybrid loss
        loss = self.compute_loss(generated, real_paths)
        
        return loss


def create_c6_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> C6Model:
    """
    Factory function to create C6 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for loss initialization
        **kwargs: Additional model parameters
        
    Returns:
        Initialized C6 model
    """
    # Default parameters optimized for hybrid training
    # Weight calculated for ~10% signature contribution to SDE matching loss
    default_params = {
        'data_size': 1,         # Observable data dimension
        'latent_size': 4,       # Latent state dimension
        'hidden_size': 64,      # Hidden layer size
        'noise_std': 0.1,       # Observation noise
        'sig_depth': 4,         # Signature depth
        'sde_weight': 1.0,      # SDE matching loss weight
        'mmd_weight': 7.6,      # Signature MMD weight (~10% contribution for strong signature constraint)
        'mmd_sigma': 1.0        # RBF kernel bandwidth
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating C6 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return C6Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# C6 will be registered automatically when imported

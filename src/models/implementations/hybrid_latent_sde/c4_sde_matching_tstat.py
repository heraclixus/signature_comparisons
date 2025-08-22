"""
C4 Model Implementation: SDE Matching + T-Statistic Loss

This implements a hybrid model that combines:
- V2 SDE Matching architecture for powerful generative modeling
- T-Statistic signature loss for distributional quality

Architecture:
- Generator: V2 SDE Matching (Prior + Posterior networks)
- Loss: Hybrid SDE Matching + T-Statistic
- Training: Multi-objective optimization

The goal is to leverage SDE Matching's generative power while enforcing
signature-based T-statistic constraints for better distributional matching.
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

# Import T-statistic loss components (reuse from C1)
from losses.t_statistic import TStatisticLoss
from models.deep_signature_transform import siglayer


class C4Model(BaseSignatureModel):
    """
    C4: Hybrid SDE Matching + T-Statistic Model
    
    Combines:
    - V2 SDE Matching architecture for generation
    - T-Statistic signature loss for distributional quality
    - Hybrid training with weighted multi-objective loss
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 data_size: int = 1, latent_size: int = 4, hidden_size: int = 64,
                 noise_std: float = 0.1, sig_depth: int = 4,
                 sde_weight: float = 1.0, tstat_weight: float = 0.7,
                 normalise_sigs: bool = True):
        """
        Initialize C4 hybrid model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for loss initialization
            data_size: Observable data dimension
            latent_size: Latent state dimension
            hidden_size: Hidden layer size
            noise_std: Observation noise standard deviation
            sig_depth: Signature depth for T-statistic
            sde_weight: Weight for SDE matching loss term
            tstat_weight: Weight for T-statistic loss term
            normalise_sigs: Whether to normalize signatures
        """
        # Create model configuration
        config = ModelConfig(
            model_id="C4",
            name="Hybrid SDE Matching + T-Statistic",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.T_STATISTIC,
            signature_method=SignatureMethod.TRUNCATED,
            description="V2 SDE Matching + T-Statistic signature loss hybrid"
        )
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.data_size = data_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        self.sig_depth = sig_depth
        self.sde_weight = sde_weight
        self.tstat_weight = tstat_weight
        self.normalise_sigs = normalise_sigs
        self.real_data = real_data  # Store for loss initialization
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        # Initialize losses after model is built
        self._initialize_losses(real_data)
        
        print(f"C4 Hybrid model initialized:")
        print(f"   Data size: {data_size}")
        print(f"   Latent size: {latent_size}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Signature depth: {sig_depth}")
        print(f"   Loss weights: SDE={sde_weight}, T-Stat={tstat_weight}")
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
        
        print(f"✅ C4 model components built successfully")
    
    def _initialize_losses(self, real_data: torch.Tensor):
        """Initialize T-statistic loss."""
        # Initialize T-statistic loss (reuse from C1)
        self.tstat_loss = TStatisticLoss(
            signature_transform=self.signature_transform,
            sig_depth=self.sig_depth,
            normalise_sigs=self.normalise_sigs
        )
        
        # Set real data for T-statistic computation
        self.tstat_loss.set_real_data(real_data)
        
        print(f"✅ C4 hybrid losses initialized")
        print(f"   T-statistic: depth={self.sig_depth}, normalize={self.normalise_sigs}")
    
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
        Compute hybrid SDE matching + T-statistic loss.
        
        Args:
            generated_output: Generated paths (used for T-statistic)
            real_paths: Real paths (used for SDE matching)
            
        Returns:
            Combined loss tensor
        """
        if real_paths is None:
            raise ValueError("C4 model requires real_paths for SDE matching computation")
        
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
        
        # Compute T-statistic loss on generated output
        # Extract values (remove time channel) for T-statistic
        if generated_output.dim() == 3 and generated_output.shape[1] == 2:
            generated_values = generated_output[:, 1, :]  # Shape: (batch, time)
        else:
            generated_values = generated_output
        
        tstat_loss = self.tstat_loss(generated_values, add_timeline=True)
        
        # Combine losses with weights
        total_loss = (self.sde_weight * sde_loss + 
                     self.tstat_weight * tstat_loss)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'sde_loss': sde_loss.item(),
            'tstat_loss': tstat_loss.item(),
            'total_loss': total_loss.item(),
            'sde_weight': self.sde_weight,
            'tstat_weight': self.tstat_weight
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


def create_c4_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> C4Model:
    """
    Factory function to create C4 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for loss initialization
        **kwargs: Additional model parameters
        
    Returns:
        Initialized C4 model
    """
    # Default parameters optimized for hybrid training
    default_params = {
        'data_size': 1,         # Observable data dimension
        'latent_size': 4,       # Latent state dimension
        'hidden_size': 64,      # Hidden layer size
        'noise_std': 0.1,       # Observation noise
        'sig_depth': 4,         # Signature depth
        'sde_weight': 1.0,      # SDE matching loss weight
        'tstat_weight': 0.7,    # T-statistic weight (~10% contribution for strong signature constraint)
        'normalise_sigs': True  # Normalize signatures
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating C4 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return C4Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# C4 will be registered automatically when imported

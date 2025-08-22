"""
V2: SDE Matching Model Implementation

This integrates the SDE Matching framework into our evaluation pipeline.
The SDE Matching approach uses:
- Prior network: Neural SDE generator in latent space
- Posterior network: GRU encoder + time-dependent decoder
- Training: 3-component loss (prior KL + SDE matching + reconstruction)

This provides another Neural SDE-based approach for comparison with:
- Signature-based models (A1-A4, B1-B5)
- Adversarial models (*_ADV)
- Latent SDE models (V1)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod
from models.sdematching.matching_network import MatchingSDE
from models.sdematching.prior_network import PriorInitDistribution, PriorSDE, PriorObservation
from models.sdematching.posterior_network import PosteriorEncoder, PosteriorAffine


def solve_sde(
    sde: nn.Module,
    z: torch.Tensor,
    ts: float,
    tf: float,
    n_steps: int
) -> torch.Tensor:
    """
    Simple Euler-Maruyama SDE solver.
    
    Args:
        sde: SDE module with (drift, vol) = sde(z, t)
        z: Initial condition
        ts: Start time
        tf: End time
        n_steps: Number of integration steps
        
    Returns:
        Path tensor, shape (n_steps+1, batch, latent_dim)
    """
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]
    dt = (tf - ts) / n_steps
    dt_sqrt = abs(dt) ** 0.5
    
    path = [z]
    for t in tt:
        f, g = sde(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * dt_sqrt
        path.append(z)
    
    return torch.stack(path)


class V2SDEMatchingModel(BaseSignatureModel):
    """
    V2: SDE Matching Model
    
    Architecture:
    - Prior: Learnable initial distribution + Neural SDE + Observation model
    - Posterior: GRU encoder + Time-dependent affine transform
    - Training: SDE matching loss (prior KL + drift matching + reconstruction)
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 data_size: int = 1, latent_size: int = 4, hidden_size: int = 64,
                 noise_std: float = 0.1):
        """
        Initialize V2 SDE Matching model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for compatibility
            data_size: Observable data dimension
            latent_size: Latent state dimension
            hidden_size: Hidden layer size
            noise_std: Observation noise standard deviation
        """
        # Create model configuration
        config = ModelConfig(
            model_id="V2",
            name="SDE Matching",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_SCORING,  # Placeholder
            signature_method=SignatureMethod.TRUNCATED,  # Not used
            description="SDE Matching with prior/posterior networks"
        )
        
        super().__init__(config)
        
        self.data_size = data_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.noise_std = noise_std
        
        # Create SDE Matching components
        self.p_init_distr = PriorInitDistribution(latent_size)
        self.p_sde = PriorSDE(latent_size, hidden_size)
        self.p_observe = PriorObservation(latent_size, data_size, noise_std)
        
        self.q_enc = PosteriorEncoder(data_size, hidden_size)
        self.q_affine = PosteriorAffine(latent_size, hidden_size)
        
        # Create matching SDE
        self.sde_matching = MatchingSDE(
            self.p_init_distr,
            self.p_sde,
            self.p_observe,
            self.q_enc,
            self.q_affine
        )
        
        print(f"V2 SDE Matching model initialized:")
        print(f"   Data size: {data_size}")
        print(f"   Latent size: {latent_size}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Noise std: {noise_std}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """Build model components (required by base class)."""
        pass
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for generation."""
        batch_size = batch.size(0) if batch.dim() > 1 else 1
        time_steps = batch.size(2) if batch.dim() == 3 else 100
        
        return self.generate_samples(batch_size, time_steps)
    
    def generate_samples(self, batch_size: int, time_steps: int = 100, 
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
            # Get device from model parameters
            device = next(self.parameters()).device
            z0 = z0.to(device)
            
            # 2. Solve prior SDE in latent space
            zs = solve_sde(self.p_sde, z0, 0., T, n_steps=time_steps-1)  # (time_steps, batch, latent_size)
            
            # 3. Map latent trajectories to observations
            zs_flat = zs.reshape(-1, self.latent_size)  # (time_steps * batch, latent_size)
            xs_flat, _ = self.p_observe.get_coeffs(zs_flat)  # (time_steps * batch, data_size)
            xs = xs_flat.reshape(time_steps, batch_size, self.data_size)  # (time_steps, batch, data_size)
            
            # 4. Format output to match our interface
            # Convert from (time_steps, batch, data_size) to (batch, 2, time_steps)
            ts = torch.linspace(0, T, time_steps, device=device)
            time_channel = ts.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            value_channel = xs.permute(1, 2, 0)  # (batch, data_size, time_steps)
            
            output = torch.cat([time_channel, value_channel], dim=1)
            
            return output
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute SDE matching loss.
        
        Args:
            generated_output: Not used (model generates internally)
            real_paths: Real trajectory data
            
        Returns:
            SDE matching loss
        """
        if real_paths is None:
            raise ValueError("SDE Matching requires real_paths")
        
        # Format data for SDE matching
        # Convert from (batch, 2, time_steps) to (batch, time_steps, data_size)
        xs = real_paths[:, 1:, :].permute(0, 2, 1)  # Extract values and transpose
        
        # Create time grid
        batch_size = real_paths.size(0)
        time_steps = real_paths.size(2)
        ts = torch.linspace(0, 1.0, time_steps, device=real_paths.device)
        ts = ts.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch, time_steps, 1)
        
        # Compute SDE matching loss
        loss = self.sde_matching(xs, ts)
        
        return loss.mean()
    
    def compute_training_loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute SDE matching loss with component breakdown.
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Format data for SDE matching
        xs = batch[:, 1:, :].permute(0, 2, 1)  # (batch, time_steps, data_size)
        
        # Create time grid
        batch_size = batch.size(0)
        time_steps = batch.size(2)
        ts = torch.linspace(0, 1.0, time_steps, device=batch.device)
        ts = ts.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (batch, time_steps, 1)
        
        # Compute SDE matching loss components
        ctx = self.q_enc(xs)
        
        # Component losses
        loss_prior = self.sde_matching.loss_prior(ctx).mean()
        
        # Random time for diffusion loss
        t = torch.rand(batch_size, 1, device=batch.device) * (ts[:, -1] - ts[:, 0]) + ts[:, 0]
        loss_diff = self.sde_matching.loss_diff(ctx, t).mean()
        
        # Random observation for reconstruction loss
        rng = torch.arange(batch_size, device=batch.device)
        u = torch.randint(time_steps, [batch_size], device=batch.device)
        t_u = ts[rng, u]
        x_u = xs[rng, u]
        loss_recon = self.sde_matching.loss_recon(ctx, x_u, t_u).mean()
        
        # Total loss
        total_loss = loss_prior + loss_diff + loss_recon
        
        # Metrics for monitoring (compatible with training loop)
        metrics = {
            'loss': total_loss.item(),
            'prior_loss': loss_prior.item(),
            'diffusion_loss': loss_diff.item(),
            'reconstruction_loss': loss_recon.item(),
            'kl_loss': loss_prior.item()  # Use prior loss as KL equivalent for compatibility
        }
        
        return total_loss, metrics


def create_v2_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   data_size: int = 1, latent_size: int = 4, hidden_size: int = 64,
                   noise_std: float = 0.1, config_overrides: Dict[str, Any] = None) -> V2SDEMatchingModel:
    """
    Create V2 SDE Matching model.
    
    Args:
        example_batch: Example batch for shape inference
        real_data: Real data for compatibility
        data_size: Observable data dimension
        latent_size: Latent state dimension
        hidden_size: Hidden layer size
        noise_std: Observation noise std
        config_overrides: Optional configuration overrides (including device)
        
    Returns:
        Initialized V2 model
    """
    # Infer data dimension if needed
    if len(example_batch.shape) == 3:
        inferred_data_size = example_batch.shape[1] - 1  # Remove time dimension
        data_size = inferred_data_size if data_size is None else data_size
    
    model = V2SDEMatchingModel(
        example_batch=example_batch,
        real_data=real_data,
        data_size=data_size,
        latent_size=latent_size,
        hidden_size=hidden_size,
        noise_std=noise_std
    )
    
    return model


def test_v2_model():
    """Test V2 SDE Matching model creation and functionality."""
    print("🧪 Testing V2 SDE Matching Model...")
    
    # Create test data (OU process format)
    batch_size = 8
    time_steps = 50  # Shorter for testing
    
    # Create time-value format data
    time_channel = torch.linspace(0, 1, time_steps).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
    # Create OU-like data
    value_channel = torch.cumsum(torch.randn(batch_size, 1, time_steps) * 0.1, dim=-1)
    test_data = torch.cat([time_channel, value_channel], dim=1)
    
    try:
        # Create V2 model
        model = create_v2_model(
            example_batch=test_data[:4],
            real_data=test_data,
            data_size=1,
            latent_size=4,
            hidden_size=64,
            noise_std=0.1
        )
        
        print(f"✅ V2 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test generation
        generated = model.generate_samples(batch_size=4, time_steps=time_steps)
        print(f"✅ Generation test: {generated.shape}")
        
        # Check generated statistics
        gen_values = generated[:, 1, :].numpy()
        print(f"   Generated stats: mean={np.mean(gen_values):.4f}, std={np.std(gen_values):.4f}")
        
        # Test training step
        model.train()
        loss, metrics = model.compute_training_loss(test_data)
        print(f"✅ Training step test: loss={loss.item():.4f}")
        print(f"   Metrics: {metrics}")
        
        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ V2 model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_v2_model()

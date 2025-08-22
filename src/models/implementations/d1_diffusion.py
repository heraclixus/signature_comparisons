"""
D1 Model Implementation: Time Series Diffusion Model

This implements a diffusion-based model using the TSDiff framework:
- Generator: Transformer-based denoising network
- Loss: Diffusion denoising loss (MSE between predicted and actual noise)
- Training: Reverse diffusion process learning

Architecture based on Morgan Stanley's Stochastic Process Diffusion paper:
https://github.com/morganstanley/MSML/tree/main/papers/Stochastic_Process_Diffusion

The model treats entire time series as single entities and applies structured
noise using Gaussian Processes, then learns to reverse the diffusion process.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import base components
from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod

# Import TSDiff components (reuse existing)
from models.tsdiff.diffusion.discrete_diffusion import DiscreteDiffusion
from models.tsdiff.diffusion.noise import GaussianProcess
from models.tsdiff.diffusion.beta_scheduler import get_beta_scheduler
from models.tsdiff.utils.positional_encoding import PositionalEncoding
from models.tsdiff.utils.feedforward import FeedForward


class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based denoising network for time series diffusion.
    (Adapted from TSDiff example.py)
    """
    
    def __init__(self, dim: int = 1, hidden_dim: int = 64, max_i: int = 100, 
                 num_layers: int = 6, num_heads: int = 4):
        """
        Initialize transformer denoising network.
        
        Args:
            dim: Time series dimension (1 for univariate)
            hidden_dim: Hidden dimension
            max_i: Maximum diffusion steps
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Positional encodings (reuse TSDiff)
        self.t_enc = PositionalEncoding(hidden_dim, max_value=1.0)
        self.i_enc = PositionalEncoding(hidden_dim, max_value=max_i)
        
        # Input projection
        self.input_proj = FeedForward(dim, [], hidden_dim)
        
        # Combine encodings
        self.proj = FeedForward(3 * hidden_dim, [], hidden_dim, final_activation=nn.ReLU())
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = FeedForward(hidden_dim, [], dim)
        
        print(f"TimeSeriesTransformer initialized:")
        print(f"   Dim: {dim}, Hidden: {hidden_dim}, Layers: {num_layers}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer.
        
        Args:
            x: Noisy time series [B, S, D]
            t: Time coordinates [B, S, 1]  
            i: Diffusion step [B, S, 1]
            
        Returns:
            Predicted noise [B, S, D]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Project inputs
        x_proj = self.input_proj(x)  # [B, S, hidden_dim]
        t_proj = self.t_enc(t)       # [B, S, hidden_dim]
        i_proj = self.i_enc(i)       # [B, S, hidden_dim]
        
        # Combine all inputs
        combined = torch.cat([x_proj, t_proj, i_proj], dim=-1)  # [B, S, 3*hidden_dim]
        hidden = self.proj(combined)  # [B, S, hidden_dim]
        
        # Apply transformer layers with residual connections
        for transformer_layer in self.transformer_layers:
            attn_output, _ = transformer_layer(hidden, hidden, hidden)
            hidden = hidden + torch.relu(attn_output)
        
        # Output projection to predict noise
        output = self.output_proj(hidden)  # [B, S, dim]
        
        return output


class DataFormatAdapter:
    """
    Adapter to convert between our dataset format and TSDiff format.
    """
    
    @staticmethod
    def our_to_tsdiff(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert our format to TSDiff format.
        
        Args:
            data: Our format [B, 2, T] where dim 1 is [time, values]
            
        Returns:
            x: Values [B, T, 1]
            t: Time coordinates [B, T, 1]
        """
        # Extract time and values
        t = data[:, 0:1, :].transpose(1, 2)  # [B, T, 1]
        x = data[:, 1:2, :].transpose(1, 2)  # [B, T, 1]
        
        return x, t
    
    @staticmethod
    def tsdiff_to_our(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Convert TSDiff format to our format.
        
        Args:
            x: Values [B, T, 1]
            t: Time coordinates [B, T, 1]
            
        Returns:
            data: Our format [B, 2, T] where dim 1 is [time, values]
        """
        # Transpose and combine
        t_channel = t.transpose(1, 2)  # [B, 1, T]
        x_channel = x.transpose(1, 2)  # [B, 1, T]
        
        data = torch.cat([t_channel, x_channel], dim=1)  # [B, 2, T]
        
        return data


class D1Model(BaseSignatureModel):
    """
    D1: Time Series Diffusion Model
    
    Based on TSDiff framework from Morgan Stanley's Stochastic Process Diffusion paper.
    Treats entire time series as entities and applies GP-structured diffusion.
    
    Architecture:
    - Generator: Transformer-based denoising network
    - Diffusion: Discrete diffusion with GP-structured noise
    - Loss: Diffusion denoising loss (MSE)
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 hidden_dim: int = 64, num_layers: int = 6, num_heads: int = 4,
                 diffusion_steps: int = 100, gp_sigma: float = 0.05,
                 beta_start: float = 1e-4, beta_end: float = 0.2):
        """
        Initialize D1 diffusion model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for training
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            diffusion_steps: Number of diffusion timesteps
            gp_sigma: Gaussian process sigma for noise structure
            beta_start: Starting noise level
            beta_end: Ending noise level
        """
        # Create model configuration
        config = ModelConfig(
            model_id="D1",
            name="Time Series Diffusion Model",
            generator_type=GeneratorType.NEURAL_SDE,  # Closest match
            loss_type=LossType.SIGNATURE_SCORING,     # Placeholder
            signature_method=SignatureMethod.TRUNCATED,  # Not used
            description="TSDiff-based diffusion model for time series generation"
        )
        
        # Store hyperparameters BEFORE calling super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.diffusion_steps = diffusion_steps
        self.gp_sigma = gp_sigma
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.real_data = real_data  # Store for training
        
        # Initialize base class (this will call _build_model)
        super().__init__(config)
        
        print(f"D1 Diffusion model initialized:")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Transformer layers: {num_layers}")
        print(f"   Attention heads: {num_heads}")
        print(f"   Diffusion steps: {diffusion_steps}")
        print(f"   GP sigma: {gp_sigma}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """Build model components using existing TSDiff implementations."""
        # Create denoising network (reuse TSDiff transformer concept)
        self.denoising_network = TimeSeriesTransformer(
            dim=1,  # Univariate time series
            hidden_dim=self.hidden_dim,
            max_i=self.diffusion_steps,
            num_layers=self.num_layers,
            num_heads=self.num_heads
        )
        
        # Create beta scheduler (reuse TSDiff beta scheduler)
        from models.tsdiff.diffusion.beta_scheduler import BetaLinear
        self.beta_scheduler = BetaLinear(start=self.beta_start, end=self.beta_end)
        
        # Create noise function
        noise_fn = lambda: GaussianProcess(dim=1, sigma=self.gp_sigma)
        
        # Create diffusion process (reuse TSDiff discrete diffusion)
        self.diffusion = DiscreteDiffusion(
            dim=1,
            num_steps=self.diffusion_steps,
            beta_fn=self.beta_scheduler,
            noise_fn=noise_fn,
            is_time_series=True,
            predict_gaussian_noise=True
        )
        
        # Get beta and alpha schedules for sampling
        diffusion_indices = torch.linspace(0, 1, self.diffusion_steps)
        self.betas = self.beta_scheduler(diffusion_indices)
        self.alphas = torch.cumprod(1 - self.betas, dim=0)
        
        print(f"✅ D1 model components built successfully")
    

    
    def _get_gp_covariance(self, t: torch.Tensor) -> torch.Tensor:
        """Get GP covariance matrix (from TSDiff example)."""
        s = t - t.transpose(-1, -2)
        diag = torch.eye(t.shape[-2]).to(t) * 1e-5  # Numerical stability
        return torch.exp(-torch.square(s / self.gp_sigma)) + diag
    
    def _add_noise(self, x: torch.Tensor, t: torch.Tensor, i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add GP-structured noise to time series (from TSDiff example)."""
        # Generate Gaussian noise
        noise_gaussian = torch.randn_like(x)
        
        # Apply GP structure
        cov = self._get_gp_covariance(t)
        L = torch.linalg.cholesky(cov)
        noise = L @ noise_gaussian
        
        # Apply diffusion schedule
        alpha = self.alphas[i.long()].to(x)
        x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
        
        return x_noisy, noise
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for generation."""
        batch_size = batch.size(0) if batch.dim() > 1 else 1
        time_steps = batch.size(2) if batch.dim() == 3 else 64
        
        return self.generate_samples(batch_size, time_steps)
    
    def generate_samples(self, batch_size: int, time_steps: int = 64) -> torch.Tensor:
        """
        Generate samples using reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            time_steps: Number of time points
            
        Returns:
            Generated paths, shape (batch, 2, time_steps) - [time, value]
        """
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Create time grid
            t_grid = torch.linspace(0, 1, time_steps).view(1, -1, 1).to(device)
            t = t_grid.repeat(batch_size, 1, 1)  # [B, T, 1]
            
            # Initialize with GP-structured noise
            cov = self._get_gp_covariance(t)
            L = torch.linalg.cholesky(cov)
            x = L @ torch.randn_like(t)  # [B, T, 1]
            
            # Reverse diffusion process
            for diff_step in reversed(range(self.diffusion_steps)):
                alpha = self.alphas[diff_step].to(device)
                beta = self.betas[diff_step].to(device)
                
                # Create noise for this step
                z = L @ torch.randn_like(t)
                
                # Diffusion step index
                i = torch.full_like(t, diff_step, dtype=torch.float)
                
                # Predict noise using denoising network
                pred_noise = self.denoising_network(x, t, i)
                
                # Reverse diffusion step
                x = (x - beta * pred_noise / (1 - alpha).sqrt()) / (1 - beta).sqrt() + beta.sqrt() * z
            
            # Convert back to our format
            output = DataFormatAdapter.tsdiff_to_our(x, t)
            
            return output
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute diffusion denoising loss.
        
        Args:
            generated_output: Not used (diffusion trains on real data)
            real_paths: Real path data for diffusion training
            
        Returns:
            Diffusion loss tensor
        """
        if real_paths is None:
            raise ValueError("D1 model requires real_paths for diffusion training")
        
        # Convert to TSDiff format
        x, t = DataFormatAdapter.our_to_tsdiff(real_paths)
        
        # Sample random diffusion step for each sample
        batch_size = x.shape[0]
        i = torch.randint(0, self.diffusion_steps, size=(batch_size,), device=x.device)
        i = i.view(-1, 1, 1).expand_as(x)  # [B, S, 1]
        
        # Add noise at step i
        x_noisy, noise = self._add_noise(x, t, i)
        
        # Predict noise using denoising network
        pred_noise = self.denoising_network(x_noisy, t, i)
        
        # Compute MSE loss between predicted and actual noise
        loss = torch.mean((pred_noise - noise) ** 2)
        
        # Store loss components for monitoring
        self.last_loss_components = {
            'diffusion_loss': loss.item(),
            'total_loss': loss.item()
        }
        
        return loss
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get detailed breakdown of loss components."""
        return getattr(self, 'last_loss_components', {})
    
    def compute_training_loss(self, batch: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute training loss for a batch.
        
        Args:
            batch: Input batch (not used for diffusion)
            real_paths: Real path data
            
        Returns:
            Loss for backpropagation
        """
        self.train()
        
        # Diffusion training uses real data directly
        loss = self.compute_loss(None, real_paths)
        
        return loss


def create_d1_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   **kwargs) -> D1Model:
    """
    Factory function to create D1 model.
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for training
        **kwargs: Additional model parameters
        
    Returns:
        Initialized D1 model
    """
    # Default parameters optimized for time series diffusion
    default_params = {
        'hidden_dim': 64,           # Transformer hidden dimension
        'num_layers': 6,            # Transformer layers
        'num_heads': 4,             # Attention heads
        'diffusion_steps': 100,     # Diffusion timesteps
        'gp_sigma': 0.05,           # GP noise structure parameter
        'beta_start': 1e-4,         # Starting noise level
        'beta_end': 0.2             # Ending noise level
    }
    
    # Update with any provided parameters
    params = {**default_params, **kwargs}
    
    print(f"Creating D1 model with parameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    return D1Model(example_batch, real_data, **params)


# Model registration is handled by the main implementations module
# D1 will be registered automatically when imported

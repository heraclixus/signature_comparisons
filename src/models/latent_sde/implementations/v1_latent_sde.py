"""
V1: TorchSDE-based Latent SDE Model

This adapts the official torchsde latent SDE example to work with our OU process dataset
and evaluation pipeline. This serves as a reference implementation to understand
what's going wrong with our custom latent SDE.

Based on: https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Optional, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

try:
    import torchsde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    import warnings
    warnings.warn("torchsde not available. TorchSDE Latent SDE will not work.")

from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod


def _stable_division(a, b, epsilon=1e-7):
    """Stable division to avoid numerical issues."""
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class TorchSDELatentSDE(torchsde.SDEIto):
    """
    Latent SDE adapted from torchsde example for OU process data.
    
    This implements a latent SDE where:
    - Prior: Ornstein-Uhlenbeck process (matches our data!)
    - Posterior: Learned neural SDE with positional encoding
    - Observations: Direct latent state (no separate decoder needed)
    """
    
    def __init__(self, theta=1.0, mu=0.0, sigma=0.5, hidden_size=200):
        """
        Initialize TorchSDE Latent SDE.
        
        Args:
            theta: OU process mean reversion rate
            mu: OU process long-term mean  
            sigma: OU process volatility
            hidden_size: Hidden size for posterior network
        """
        super(TorchSDELatentSDE, self).__init__(noise_type="diagonal")
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde is required for TorchSDE Latent SDE")
        
        # OU process parameters (prior)
        logvar = math.log(sigma ** 2 / (2. * theta))
        
        # Prior drift (OU process)
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        
        # Prior initial distribution p(y0)
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[logvar]]))
        
        # Posterior drift network (learned)
        # Takes in: [sin(t), cos(t), y] → drift correction
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize final layer to zero (start close to prior)
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)
        
        # Posterior initial distribution q(y0) (learnable)
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor([[logvar]]), requires_grad=True)
        
        print(f"TorchSDE Latent SDE initialized:")
        print(f"   OU parameters: θ={theta}, μ={mu}, σ={sigma}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def f(self, t, y):
        """Posterior drift (learned)."""
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        # Positional encoding for time + current state
        return self.net(torch.cat((torch.sin(t), torch.cos(t), y), dim=-1))
    
    def g(self, t, y):
        """Shared diffusion (same for prior and posterior)."""
        return self.sigma.repeat(y.size(0), 1)
    
    def h(self, t, y):
        """Prior drift (OU process)."""
        return self.theta * (self.mu - y)
    
    def f_aug(self, t, y):
        """Drift for augmented dynamics with log q/p term."""
        y = y[:, 0:1]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)
    
    def g_aug(self, t, y):
        """Diffusion for augmented dynamics with log q/p term."""
        y = y[:, 0:1]
        g = self.g(t, y)
        g_logqp = torch.zeros_like(y)
        return torch.cat([g, g_logqp], dim=1)
    
    def forward(self, ts, batch_size, eps=None):
        """
        Forward pass: sample from posterior and compute KL.
        
        Args:
            ts: Time points
            batch_size: Batch size
            eps: Random noise (optional)
            
        Returns:
            Tuple of (trajectories, kl_divergence)
        """
        eps = torch.randn(batch_size, 1).to(self.qy0_std) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        
        # Compute KL divergence at t=0
        qy0 = torch.distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = torch.distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = torch.distributions.kl_divergence(qy0, py0).sum(dim=1)
        
        # Augmented initial condition [y0, 0] for path-wise KL computation
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        
        # Solve augmented SDE
        aug_ys = torchsde.sdeint(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method='euler',
            dt=0.01,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        
        # Extract trajectories and path-wise KL
        ys = aug_ys[:, :, 0:1]  # Trajectories
        logqp_path = aug_ys[-1, :, 1]  # Path-wise KL at final time
        
        # Total KL = KL(t=0) + KL(path)
        logqp = (logqp0 + logqp_path).mean(dim=0)
        
        return ys, logqp
    
    def sample_prior(self, ts, batch_size, eps=None):
        """Sample from prior (OU process)."""
        eps = torch.randn(batch_size, 1).to(self.py0_mean) if eps is None else eps
        y0 = self.py0_mean + eps * self.py0_std
        return torchsde.sdeint(self, y0, ts, method='euler', dt=0.01, names={'drift': 'h'})
    
    def sample_posterior(self, ts, batch_size, eps=None):
        """Sample from posterior (learned SDE)."""
        eps = torch.randn(batch_size, 1).to(self.qy0_mean) if eps is None else eps
        y0 = self.qy0_mean + eps * self.qy0_std
        return torchsde.sdeint(self, y0, ts, method='euler', dt=0.01)
    
    @property
    def py0_std(self):
        """Prior initial standard deviation."""
        return torch.exp(.5 * self.py0_logvar)
    
    @property
    def qy0_std(self):
        """Posterior initial standard deviation."""
        return torch.exp(.5 * self.qy0_logvar)


class V1TorchSDEModel(BaseSignatureModel):
    """
    V1 using TorchSDE reference implementation.
    
    This adapts the official torchsde latent SDE to work with our OU process
    dataset and provides a fair comparison with signature-based models.
    """
    
    def __init__(self, example_batch: torch.Tensor, real_data: torch.Tensor,
                 theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5,
                 hidden_size: int = 200):
        """
        Initialize V1 TorchSDE model.
        
        Args:
            example_batch: Example batch for compatibility
            real_data: Real data for compatibility
            theta: OU process mean reversion rate
            mu: OU process long-term mean
            sigma: OU process volatility
            hidden_size: Hidden layer size for posterior network
        """
        # Create model configuration
        config = ModelConfig(
            model_id="V1_TorchSDE",
            name="TorchSDE Latent SDE",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_SCORING,  # Placeholder
            signature_method=SignatureMethod.TRUNCATED,  # Not used
            description="TorchSDE reference latent SDE implementation"
        )
        
        super().__init__(config)
        
        # Create latent SDE model
        self.latent_sde = TorchSDELatentSDE(
            theta=theta,
            mu=mu,
            sigma=sigma,
            hidden_size=hidden_size
        )
        
        # Store parameters for reference
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.hidden_size = hidden_size
        
        print(f"V1 TorchSDE model initialized:")
        print(f"   OU parameters: θ={theta}, μ={mu}, σ={sigma}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_model(self):
        """Build model components (required by base class)."""
        pass
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for generation."""
        batch_size = batch.size(0) if batch.dim() > 1 else 1
        time_steps = batch.size(2) if batch.dim() == 3 else 100
        
        return self.generate_samples(batch_size, time_steps)
    
    def generate_samples(self, batch_size: int, time_steps: int = 100) -> torch.Tensor:
        """
        Generate samples using TorchSDE latent SDE.
        
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
            
            # Sample from posterior
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
        Compute latent SDE loss using torchsde approach.
        
        Args:
            generated_output: Not used (model generates internally)
            real_paths: Real OU process data
            
        Returns:
            Loss tensor
        """
        if real_paths is None:
            raise ValueError("TorchSDE Latent SDE requires real_paths")
        
        batch_size = real_paths.size(0)
        time_steps = real_paths.size(2)
        
        # Extract observations (remove time channel)
        observations = real_paths[:, 1, :]  # (batch, time_steps)
        
        # Create time grid
        ts = torch.linspace(0, 1.0, time_steps)
        
        # Forward pass through latent SDE
        ys, kl = self.latent_sde(ts, batch_size)  # ys: (time_steps, batch, 1)
        ys = ys.squeeze(-1)  # (time_steps, batch)
        
        # Likelihood: p(observations | latent_trajectories)
        # Use Laplace likelihood as in original torchsde example
        likelihood = torch.distributions.Laplace(loc=ys.t(), scale=0.1)  # (batch, time_steps)
        log_likelihood = likelihood.log_prob(observations).sum(dim=1).mean(dim=0)
        
        # ELBO = log_likelihood - KL
        # Loss = -ELBO = -log_likelihood + KL
        loss = -log_likelihood + kl
        
        return loss
    
    def compute_training_loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Training step for TorchSDE latent SDE.
        
        Args:
            batch: Training batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Extract components for loss computation
        batch_size = batch.size(0)
        time_steps = batch.size(2)
        ts = torch.linspace(0, 1.0, time_steps)
        
        # Forward pass through latent SDE
        ys, kl = self.latent_sde(ts, batch_size)  # ys: (time_steps, batch, 1)
        ys = ys.squeeze(-1)  # (time_steps, batch)
        
        # Extract observations
        observations = batch[:, 1, :]  # (batch, time_steps)
        
        # Likelihood: p(observations | latent_trajectories)
        likelihood = torch.distributions.Laplace(loc=ys.t(), scale=0.1)  # (batch, time_steps)
        log_likelihood = likelihood.log_prob(observations).sum(dim=1).mean(dim=0)
        
        # ELBO = log_likelihood - KL
        # Loss = -ELBO = -log_likelihood + KL
        loss = -log_likelihood + kl
        
        # Compute components for monitoring
        reconstruction_loss = -log_likelihood
        
        metrics = {
            'loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl.item(),
            'log_likelihood': log_likelihood.item(),
            'elbo': -loss.item()
        }
        
        return loss, metrics


def create_v1_model(example_batch: torch.Tensor, real_data: torch.Tensor,
                   theta: float = 2.0, mu: float = 0.0, sigma: float = 0.5,
                   hidden_size: int = 200) -> V1TorchSDEModel:
    """
    Create V1 Latent SDE model (TorchSDE-based).
    
    Args:
        example_batch: Example batch for compatibility
        real_data: Real data for compatibility  
        theta: OU process mean reversion rate
        mu: OU process long-term mean
        sigma: OU process volatility
        hidden_size: Hidden size for posterior network
        
    Returns:
        Initialized V1 TorchSDE model
    """
    model = V1TorchSDEModel(
        example_batch=example_batch,
        real_data=real_data,
        theta=theta,
        mu=mu,
        sigma=sigma,
        hidden_size=hidden_size
    )
    
    return model


def test_v1_model():
    """Test V1 model creation and basic functionality."""
    print("🧪 Testing V1 Latent SDE Model (TorchSDE-based)...")
    
    if not TORCHSDE_AVAILABLE:
        print("❌ torchsde not available, skipping test")
        return False
    
    # Create test data (OU process format)
    batch_size = 8
    time_steps = 100
    
    # Create time-value format data
    time_channel = torch.linspace(0, 1, time_steps).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
    # Create OU-like data
    value_channel = torch.cumsum(torch.randn(batch_size, 1, time_steps) * 0.1, dim=-1)
    test_data = torch.cat([time_channel, value_channel], dim=1)
    
    try:
        # Create V1 model
        model = create_v1_model(
            example_batch=test_data[:4],
            real_data=test_data,
            theta=2.0,  # OU mean reversion
            mu=0.0,     # OU long-term mean
            sigma=0.5,  # OU volatility
            hidden_size=64  # Smaller for testing
        )
        
        print(f"✅ V1 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test generation
        generated = model.generate_samples(batch_size=4)
        print(f"✅ Generation test: {generated.shape}")
        
        # Check generated statistics
        gen_values = generated[:, 1, :].numpy()
        print(f"   Generated stats: mean={np.mean(gen_values):.4f}, std={np.std(gen_values):.4f}")
        
        # Test training step
        model.train()
        loss, metrics = model.compute_training_loss(test_data)
        print(f"✅ Training step test: loss={loss.item():.4f}")
        print(f"   Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"❌ V1 model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_v1_model()

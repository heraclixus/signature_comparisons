"""
Distributional Diffusion Model for Time Series.

This module implements the distributional diffusion model from 
"Path Diffusion with Signature Kernels" that learns full distributions 
P_θ(·|X_t, t) using signature kernel scoring rules.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Union, Callable

# Import existing tsdiff components
try:
    from models.tsdiff.diffusion.noise import OrnsteinUhlenbeck
    from models.tsdiff.diffusion.beta_scheduler import BetaLinear
    TSDIFF_AVAILABLE = True
except ImportError:
    TSDIFF_AVAILABLE = False
    warnings.warn("tsdiff components not available")

# Import signature scoring loss
try:
    from losses.signature_scoring_loss import SignatureScoringLoss, create_signature_scoring_loss
    SIGNATURE_LOSS_AVAILABLE = True
except ImportError:
    SIGNATURE_LOSS_AVAILABLE = False
    warnings.warn("signature_scoring_loss not available")


class DistributionalDiffusion(nn.Module):
    """
    Distributional diffusion model that learns full distributions P_θ(·|X_t, t)
    using signature kernel scoring rules, following the method in the paper.
    
    Key features:
    - Uses OU process covariance structure for forward diffusion
    - Population-based training with multiple samples per step
    - Signature kernel scoring rule loss S_λ,sig(P_θ, X_0)
    - DDIM-like deterministic sampling with coarse time grids
    """
    
    def __init__(
        self,
        dim: int,                    # d: dimension of each time point
        seq_len: int,                # M: number of time points
        gamma: float = 1.0,          # OU process parameter γ
        population_size: int = 8,    # m: population size for training
        lambda_param: float = 1.0,   # λ: generalized kernel score parameter
        signature_level: int = 4,    # s: signature truncation level
        kernel_type: str = "rbf",    # Signature kernel type
        dyadic_order: int = 4,       # Dyadic partitioning order
        sigma: float = 1.0,          # RBF kernel bandwidth
        max_batch: int = 64,         # Maximum batch size for kernel computation
        **kwargs
    ):
        """
        Initialize distributional diffusion model.
        
        Args:
            dim: Dimension of each time point (d)
            seq_len: Number of time points (M)
            gamma: OU process parameter (γ in paper)
            population_size: Population size for training (m in paper)
            lambda_param: Generalized kernel score parameter (λ in paper)
            signature_level: Signature truncation level (s in paper)
            kernel_type: Type of static kernel ("rbf" or "linear")
            dyadic_order: Dyadic partitioning order for PDE solver
            sigma: RBF kernel bandwidth parameter
            max_batch: Maximum batch size for kernel computation
        """
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.population_size = population_size
        self.lambda_param = lambda_param
        
        # Initialize OU process for forward diffusion
        if TSDIFF_AVAILABLE:
            self.ou_process = OrnsteinUhlenbeck(dim=seq_len, theta=gamma)
            
            # Create time grid for OU process
            self.register_buffer('t_grid', torch.linspace(0, 1, seq_len).unsqueeze(-1))
            
            # Get OU covariance matrices (reusing existing methods)
            Sigma = self.ou_process.covariance(self.t_grid)  # (seq_len, seq_len)
            L = self.ou_process.covariance_cholesky(self.t_grid)  # Cholesky factor
            
        else:
            # Fallback: manually create OU covariance
            warnings.warn("Using fallback OU covariance computation")
            t_grid = torch.linspace(0, 1, seq_len)
            Sigma = self._build_ou_covariance(t_grid, gamma)
            L = torch.linalg.cholesky(Sigma)
            self.register_buffer('t_grid', t_grid.unsqueeze(-1))
        
        # Kronecker product for full covariance: Σ̃ = I_d ⊗ Σ
        self.register_buffer('Sigma', Sigma)
        self.register_buffer('L', L)
        
        # Use contiguous tensors for Kronecker product
        Sigma_full = torch.kron(torch.eye(dim), Sigma.contiguous())
        L_full = torch.kron(torch.eye(dim), L.contiguous())
        
        self.register_buffer('Sigma_full', Sigma_full.contiguous())
        self.register_buffer('L_full', L_full.contiguous())
        
        # Initialize signature scoring loss
        if SIGNATURE_LOSS_AVAILABLE:
            self.scoring_loss = create_signature_scoring_loss(
                method="direct",
                signature_level=signature_level,
                lambda_param=lambda_param,
                kernel_type=kernel_type,
                dyadic_order=dyadic_order,
                sigma=sigma,
                max_batch=max_batch
            )
        else:
            raise ImportError("SignatureScoringLoss not available")
    
    def _build_ou_covariance(self, t_grid: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Fallback method to build OU process covariance matrix.
        Σ_{ij} = exp(-γ|t_i - t_j|)
        """
        t_diff = t_grid.unsqueeze(0) - t_grid.unsqueeze(1)  # |t_i - t_j|
        return torch.exp(-gamma * torch.abs(t_diff))
    
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward diffusion process following the paper:
        p(X_t | X_0) = N(α_t X_0, Σ(t))
        where α_t = 1-t, σ_t = 1-α_t, Σ(t) = σ_t * Σ_OU
        
        Args:
            x0: Clean data (batch_size, dim, seq_len)
            t: Diffusion time (batch_size,)
            
        Returns:
            x_t: Noisy data (batch_size, dim, seq_len)
        """
        batch_size = x0.shape[0]
        
        # Flatten to (batch_size, dim*seq_len) for covariance application
        x0_flat = x0.view(batch_size, -1)
        
        # Diffusion parameters (following paper: α_t = 1-t, σ_t = 1-α_t)
        alpha_t = 1 - t.view(-1, 1)  # (batch_size, 1)
        sigma_t = 1 - alpha_t
        
        # Sample noise from N(0, σ_t * Σ̃)
        z = torch.randn_like(x0_flat)
        
        # Apply OU covariance structure
        # noise = σ_t^{1/2} * L_full @ z
        noise = sigma_t.sqrt() * (self.L_full @ z.unsqueeze(-1)).squeeze(-1)
        
        # Apply forward diffusion: X_t = α_t * X_0 + noise
        x_t_flat = alpha_t * x0_flat + noise
        
        return x_t_flat.view(batch_size, self.dim, self.seq_len)
    
    def get_loss(self, generator: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute signature scoring rule loss following Algorithm 1 from the paper.
        
        Args:
            generator: Neural network P_θ(·|X_t, t, Z)
            x0: Real data batch (batch_size, dim, seq_len)
            
        Returns:
            loss: Signature scoring rule loss L_sig
        """
        batch_size = x0.shape[0]
        
        # Step 1: Sample t_i ~ U([0,1]) for i ∈ [n]
        t = torch.rand(batch_size, device=x0.device)
        
        # Step 2: Sample X_0^i ~ P_0 for i ∈ [n] (already provided as x0)
        
        # Step 3: Sample X_{t_i}^i using forward diffusion process
        x_t = self.forward_diffusion(x0, t)
        
        # Step 4 & 5: Sample ξ_{ij} ~ N(0, I_{Md}) and generate X̃_0^{(i)}
        generated_samples = []
        for _ in range(self.population_size):
            # Sample ξ ~ N(0, I_{Md})
            xi = torch.randn_like(x_t)
            
            # Generate sample from learned distribution P_θ(·|X_t, t, ξ)
            x_gen = generator(x_t, t, xi)
            generated_samples.append(x_gen)
        
        # Stack samples: (batch, population_size, dim, seq_len)
        generated_samples = torch.stack(generated_samples, dim=1)
        
        # Step 6: Compute L_sig using signature scoring rule
        loss = self.scoring_loss(generated_samples, x0)
        
        return loss
    
    def sample(
        self, 
        generator: nn.Module, 
        num_samples: int, 
        num_coarse_steps: int = 20,
        device: str = 'cpu',
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using DDIM-like coarse sampling following Algorithm 2.
        
        Args:
            generator: Neural network P_θ(·|X_t, t, Z)
            num_samples: Number of samples to generate
            num_coarse_steps: Number of coarse time steps S
            device: Device to run on
            
        Returns:
            Generated samples (num_samples, dim, seq_len)
        """
        # Get coarse time schedule: τ_0 = 1 > τ_1 > ... > τ_S = 0
        tau_schedule = torch.linspace(1.0, 0.0, num_coarse_steps + 1).to(device)
        
        # Initialize from prior: X_{τ_0} = L̃ Z_0 where Z_0 ~ N(0, I_{Md})
        z0 = torch.randn(num_samples, self.dim * self.seq_len, device=device)
        x_tau = (self.L_full.to(device) @ z0.unsqueeze(-1)).squeeze(-1)
        x_tau = x_tau.view(num_samples, self.dim, self.seq_len)
        
        # Backward sampling loop: k ∈ {S, ..., 1}
        generator.eval()
        with torch.no_grad():
            for k in range(len(tau_schedule) - 1, 0, -1):
                tau_k = tau_schedule[k]          # Current time
                tau_k_minus_1 = tau_schedule[k-1]  # Next time (smaller)
                
                # Sample Z ~ N(0, I_{Md})
                z = torch.randn_like(x_tau)
                
                # Sample X̃_0 ~ P_θ(·|X_{τ_k}, τ_k, Z)
                t_tensor = tau_k.expand(num_samples).to(device)
                x_tilde_0 = generator(x_tau, t_tensor, z)
                
                # Compute D_{τ_k} = (X_{τ_k} - √ᾱ_{τ_k} X̃_0) / √(1-ᾱ_{τ_k})
                alpha_tau_k = 1 - tau_k  # ᾱ_{τ_k} = 1 - τ_k
                alpha_tau_k_minus_1 = 1 - tau_k_minus_1
                
                D_tau_k = (x_tau - torch.sqrt(alpha_tau_k) * x_tilde_0) / torch.sqrt(1 - alpha_tau_k)
                
                # Deterministic update: X_{τ_{k-1}} = √ᾱ_{τ_{k-1}} X̃_0 + √(1-ᾱ_{τ_{k-1}}) D_{τ_k}
                x_tau = (torch.sqrt(alpha_tau_k_minus_1) * x_tilde_0 + 
                        torch.sqrt(1 - alpha_tau_k_minus_1) * D_tau_k)
        
        return x_tau
    
    def log_prob(
        self,
        generator: nn.Module,
        x: torch.Tensor,
        num_samples: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Estimate log probability of data under the learned distribution.
        This is a simplified implementation - full ELBO computation would be more complex.
        
        Args:
            generator: Trained generator network
            x: Data samples (batch_size, dim, seq_len)
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated log probabilities (batch_size,)
        """
        batch_size = x.shape[0]
        
        # Sample multiple diffusion times for estimation
        t_samples = torch.rand(num_samples, device=x.device)
        
        log_probs = []
        
        generator.eval()
        with torch.no_grad():
            for t in t_samples:
                # Forward diffusion
                t_batch = t.expand(batch_size)
                x_t = self.forward_diffusion(x, t_batch)
                
                # Generate samples from learned distribution
                generated_samples = []
                for _ in range(self.population_size):
                    xi = torch.randn_like(x_t)
                    x_gen = generator(x_t, t_batch, xi)
                    generated_samples.append(x_gen)
                
                generated_samples = torch.stack(generated_samples, dim=1)
                
                # Compute negative scoring rule as proxy for log probability
                score = self.scoring_loss(generated_samples, x)
                log_probs.append(-score)
        
        # Average over time samples
        return torch.stack(log_probs).mean(dim=0)


def create_distributional_diffusion(config: dict) -> DistributionalDiffusion:
    """
    Factory function for creating distributional diffusion models.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        DistributionalDiffusion instance
    """
    return DistributionalDiffusion(**config)

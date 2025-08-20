"""
Latent SDE Loss Functions

Implements ELBO (Evidence Lower BOund) and related loss functions for
training latent SDE models with variational inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class ELBOLoss(nn.Module):
    """
    Evidence Lower BOund (ELBO) loss for latent SDE training.
    
    ELBO = E_q[log p(X|Z)] - KL(q(Z_0|X) || p(Z_0))
         = Reconstruction Loss - KL Divergence
    """
    
    def __init__(self, reconstruction_loss: str = "mse", kl_weight: float = 1.0,
                 num_mc_samples: int = 1):
        """
        Initialize ELBO loss.
        
        Args:
            reconstruction_loss: Type of reconstruction loss ("mse", "l1")
            kl_weight: Weight for KL divergence term (β in β-VAE)
            num_mc_samples: Number of Monte Carlo samples for reconstruction
        """
        super().__init__()
        
        self.kl_weight = kl_weight
        self.num_mc_samples = num_mc_samples
        
        # Reconstruction loss function
        if reconstruction_loss == "mse":
            self.reconstruction_fn = F.mse_loss
        elif reconstruction_loss == "l1":
            self.reconstruction_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")
        
        self.loss_type = reconstruction_loss
        print(f"ELBO loss initialized:")
        print(f"   Reconstruction: {reconstruction_loss}")
        print(f"   KL weight: {kl_weight}")
        print(f"   MC samples: {num_mc_samples}")
    
    def forward(self, model, real_data: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss.
        
        Args:
            model: Latent SDE model
            real_data: Real observations, shape (batch, 2, time_steps)
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Forward pass through model
        reconstructed_data, latent_mean, latent_logvar = model(real_data, self.num_mc_samples)
        
        # 1. Reconstruction loss
        # Extract value channel (ignore time channel)
        real_values = real_data[:, 1:, :]  # (batch, data_dim, time_steps)
        recon_values = reconstructed_data[:, 1:, :]  # (batch, data_dim, time_steps)
        
        reconstruction_loss = self.reconstruction_fn(recon_values, real_values, reduction='mean')
        
        # 2. KL divergence
        kl_loss = model.compute_kl_divergence(latent_mean, latent_logvar)
        
        # 3. Total ELBO loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Loss components for monitoring
        loss_components = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'elbo': -total_loss.item()  # ELBO is negative loss
        }
        
        return total_loss, loss_components


class LatentSDELoss(nn.Module):
    """
    Wrapper for latent SDE loss functions.
    
    Provides a consistent interface with other loss functions in the framework
    while implementing ELBO-based training.
    """
    
    def __init__(self, real_data: torch.Tensor, reconstruction_loss: str = "mse",
                 kl_weight: float = 1.0, num_mc_samples: int = 1):
        """
        Initialize latent SDE loss.
        
        Args:
            real_data: Real data for initialization (shape inference)
            reconstruction_loss: Type of reconstruction loss
            kl_weight: Weight for KL divergence term
            num_mc_samples: Number of MC samples
        """
        super().__init__()
        
        self.elbo_loss = ELBOLoss(
            reconstruction_loss=reconstruction_loss,
            kl_weight=kl_weight,
            num_mc_samples=num_mc_samples
        )
        
        # Store data info for consistency checks
        self.data_shape = real_data.shape
        
        print(f"Latent SDE loss initialized with real data shape: {real_data.shape}")
    
    def forward(self, generated_data: torch.Tensor, real_data: torch.Tensor) -> torch.Tensor:
        """
        Compute loss (interface compatible with signature-based losses).
        
        Note: For latent SDE, we need the model itself, not just generated data.
        This will be called differently in the training loop.
        
        Args:
            generated_data: Not used directly (model generates internally)
            real_data: Real observations
            
        Returns:
            Loss tensor
        """
        # This interface is for compatibility with existing training loops
        # The actual ELBO computation happens in the training script
        raise NotImplementedError(
            "Latent SDE loss requires model access. Use elbo_loss.forward(model, real_data) instead."
        )
    
    def compute_elbo(self, model, real_data: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute ELBO loss with model access.
        
        Args:
            model: Latent SDE model
            real_data: Real observations
            
        Returns:
            Tuple of (loss, components)
        """
        return self.elbo_loss(model, real_data)


class SignatureEnhancedELBO(ELBOLoss):
    """
    Enhanced ELBO that uses signature-based reconstruction loss.
    
    This will be used for V2 model to combine latent SDE with signature methods.
    """
    
    def __init__(self, signature_loss_fn, kl_weight: float = 1.0, num_mc_samples: int = 1):
        """
        Initialize signature-enhanced ELBO.
        
        Args:
            signature_loss_fn: Signature-based loss function (e.g., MMD, scoring)
            kl_weight: Weight for KL divergence
            num_mc_samples: Number of MC samples
        """
        # Initialize parent with dummy reconstruction loss
        super().__init__("mse", kl_weight, num_mc_samples)
        
        # Override with signature-based reconstruction
        self.signature_loss = signature_loss_fn
        self.loss_type = "signature_enhanced"
        
        print(f"Signature-enhanced ELBO initialized:")
        print(f"   Signature loss: {type(signature_loss_fn).__name__}")
        print(f"   KL weight: {kl_weight}")
    
    def forward(self, model, real_data: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compute signature-enhanced ELBO."""
        # Get model outputs
        reconstructed_data, latent_mean, latent_logvar = model(real_data, self.num_mc_samples)
        
        # 1. Signature-based reconstruction loss
        reconstruction_loss = self.signature_loss(reconstructed_data, real_data)
        
        # 2. KL divergence (same as standard ELBO)
        kl_loss = model.compute_kl_divergence(latent_mean, latent_logvar)
        
        # 3. Total loss
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        loss_components = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'elbo': -total_loss.item()
        }
        
        return total_loss, loss_components

"""
D2: Distributional Diffusion Model Integration

This module integrates our distributional diffusion model with the existing
signature comparison pipeline, making it compatible with the BaseSignatureModel interface.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple, Union
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod

# Import our distributional diffusion components
try:
    from models.tsdiff.diffusion.distributional_diffusion import DistributionalDiffusion
    from models.distributional_generator import create_distributional_generator
    from losses.signature_scoring_loss import create_signature_scoring_loss
    DISTRIBUTIONAL_AVAILABLE = True
except ImportError as e:
    DISTRIBUTIONAL_AVAILABLE = False
    warnings.warn(f"Distributional diffusion components not available: {e}")


class D2DistributionalDiffusion(BaseSignatureModel):
    """
    D2: Distributional Diffusion Model with Signature Kernel Scoring Rules.
    
    This model implements the "Path Diffusion with Signature Kernels" approach,
    learning full distributions P_θ(·|X_t, t) using signature kernel scoring rules
    and DDIM-like coarse sampling.
    
    Key Features:
    - OU process forward diffusion
    - Population-based training with signature scoring rules
    - DDIM-like coarse sampling for acceleration
    - Compatible with existing evaluation pipeline
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize D2 model."""
        if not DISTRIBUTIONAL_AVAILABLE:
            raise ImportError("Distributional diffusion components not available")
        
        # Extract configuration first
        self.dim = config.data_config.get('dim', 1)
        self.seq_len = config.data_config.get('seq_len', 64)
        self.device = config.training_config.get('device', 'cpu')
        
        # Model hyperparameters
        self.gamma = config.generator_config.get('gamma', 1.0)
        self.population_size = config.loss_config.get('population_size', 8)
        self.lambda_param = config.loss_config.get('lambda_param', 1.0)
        self.num_coarse_steps = config.generator_config.get('num_coarse_steps', 20)
        
        # Call parent init (which will call _build_model)
        super().__init__(config)
    
    def _build_model(self):
        """Build the distributional diffusion model."""
        # Create distributional diffusion model
        ddm_kwargs = {
            'dim': self.dim,
            'seq_len': self.seq_len,
            'gamma': self.gamma,
            'population_size': self.population_size,
            'lambda_param': self.lambda_param
        }
        # Add signature config
        ddm_kwargs.update(self.config.signature_config)
        
        self.ddm = DistributionalDiffusion(**ddm_kwargs)
        
        # Create generator network
        gen_kwargs = {
            'generator_type': "feedforward",
            'data_size': self.dim,
            'seq_len': self.seq_len
        }
        # Add generator config, excluding our custom parameters
        for key, value in self.config.generator_config.items():
            if key not in ['gamma', 'num_coarse_steps']:
                gen_kwargs[key] = value
        
        self.generator = create_distributional_generator(**gen_kwargs)
        
        # Move to device
        self.ddm = self.ddm.to(self.device)
        self.generator = self.generator.to(self.device)
        
        # Store for BaseSignatureModel compatibility
        self.model = self.generator  # Main trainable component
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for compatibility with BaseSignatureModel.
        
        For distributional diffusion, this generates samples given input noise.
        """
        batch_size = x.shape[0]
        
        # Use input as noise for generation
        if x.shape != (batch_size, self.dim, self.seq_len):
            # Reshape input to expected format
            if x.numel() == batch_size * self.dim * self.seq_len:
                x = x.view(batch_size, self.dim, self.seq_len)
            else:
                # Generate random noise if input doesn't match
                x = torch.randn(batch_size, self.dim, self.seq_len, device=self.device)
        
        # Generate samples using DDIM-like sampling
        self.generator.eval()
        with torch.no_grad():
            samples = self.ddm.sample(
                generator=self.generator,
                num_samples=batch_size,
                num_coarse_steps=self.num_coarse_steps,
                device=self.device
            )
        
        return samples
    
    def compute_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute distributional diffusion loss.
        
        Args:
            generated: Not used directly (we generate internally)
            target: Real data samples (batch_size, dim, seq_len)
            
        Returns:
            Signature scoring rule loss
        """
        # Ensure target is on correct device
        target = target.to(self.device)
        
        # Compute distributional diffusion loss
        loss = self.ddm.get_loss(self.generator, target)
        
        return loss
    
    def generate_samples(self, num_samples: int, **kwargs) -> torch.Tensor:
        """
        Generate samples using the trained model.
        
        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated samples (num_samples, dim, seq_len)
        """
        self.generator.eval()
        
        with torch.no_grad():
            samples = self.ddm.sample(
                generator=self.generator,
                num_samples=num_samples,
                num_coarse_steps=kwargs.get('num_coarse_steps', self.num_coarse_steps),
                device=self.device
            )
        
        return samples
    
    def fit(self, train_data: torch.Tensor, 
            val_data: Optional[torch.Tensor] = None,
            **kwargs) -> Dict[str, Any]:
        """
        Train the distributional diffusion model.
        
        Args:
            train_data: Training data (num_samples, dim, seq_len)
            val_data: Optional validation data
            **kwargs: Training parameters
            
        Returns:
            Training history
        """
        # Extract training parameters
        num_epochs = kwargs.get('num_epochs', self.config.training_config.get('num_epochs', 100))
        batch_size = kwargs.get('batch_size', self.config.training_config.get('batch_size', 32))
        learning_rate = kwargs.get('learning_rate', self.config.training_config.get('learning_rate', 1e-4))
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.generator.parameters(), 
            lr=learning_rate,
            weight_decay=self.config.training_config.get('weight_decay', 0.0)
        )
        
        # Training loop
        train_data = train_data.to(self.device)
        num_batches = len(train_data) // batch_size
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        self.generator.train()
        self.ddm.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = train_data[start_idx:end_idx]
                
                # Training step
                optimizer.zero_grad()
                loss = self.ddm.get_loss(self.generator, batch_data)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), 
                    self.config.training_config.get('gradient_clip', 1.0)
                )
                
                optimizer.step()
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            history['train_loss'].append(avg_loss)
            
            # Validation loss
            if val_data is not None:
                val_data = val_data.to(self.device)
                self.generator.eval()
                with torch.no_grad():
                    val_loss = self.ddm.get_loss(self.generator, val_data)
                history['val_loss'].append(val_loss.item())
                self.generator.train()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        return history
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'ddm_state_dict': self.ddm.state_dict(),
            'config': self.config.to_dict()
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.ddm.load_state_dict(checkpoint['ddm_state_dict'])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': 'D2_Distributional_Diffusion',
            'dim': self.dim,
            'seq_len': self.seq_len,
            'gamma': self.gamma,
            'population_size': self.population_size,
            'lambda_param': self.lambda_param,
            'num_coarse_steps': self.num_coarse_steps,
            'generator_params': sum(p.numel() for p in self.generator.parameters()),
            'ddm_params': sum(p.numel() for p in self.ddm.parameters()),
            'device': self.device
        }


def create_d2_config(
    dim: int = 1,
    seq_len: int = 64,
    gamma: float = 1.0,
    population_size: int = 8,
    lambda_param: float = 1.0,
    num_coarse_steps: int = 20,
    **kwargs
) -> ModelConfig:
    """
    Create configuration for D2 model.
    
    Args:
        dim: Data dimension
        seq_len: Sequence length
        gamma: OU process parameter
        population_size: Population size for training
        lambda_param: Lambda parameter for scoring rule
        num_coarse_steps: Number of coarse sampling steps
        **kwargs: Additional configuration parameters
        
    Returns:
        ModelConfig for D2
    """
    return ModelConfig(
        model_id="D2",
        name="Distributional Diffusion + Signature Kernel Scoring",
        description="Path diffusion with signature kernel scoring rules and DDIM-like coarse sampling",
        generator_type=GeneratorType.NEURAL_SDE,  # Closest match
        loss_type=LossType.SIGNATURE_SCORING,
        signature_method=SignatureMethod.PDE_SOLVED,
        status="implemented",
        priority="high",
        
        # Data configuration
        data_config={
            'dim': dim,
            'seq_len': seq_len
        },
        
        # Generator configuration
        generator_config={
            'gamma': gamma,
            'num_coarse_steps': num_coarse_steps,
            'hidden_size': kwargs.get('hidden_size', 128),
            'num_layers': kwargs.get('num_layers', 3),
            'activation': kwargs.get('activation', 'relu')
        },
        
        # Loss configuration
        loss_config={
            'population_size': population_size,
            'lambda_param': lambda_param
        },
        
        # Signature configuration
        signature_config={
            'kernel_type': kwargs.get('kernel_type', 'rbf'),
            'dyadic_order': kwargs.get('dyadic_order', 4),
            'sigma': kwargs.get('sigma', 1.0),
            'max_batch': kwargs.get('max_batch', 64)
        },
        
        # Training configuration
        training_config={
            'num_epochs': kwargs.get('num_epochs', 100),
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 1e-4),
            'weight_decay': kwargs.get('weight_decay', 0.0),
            'gradient_clip': kwargs.get('gradient_clip', 1.0),
            'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        }
    )


# Factory function for easy creation
def create_d2_model(**kwargs) -> D2DistributionalDiffusion:
    """Create D2 model with default or custom configuration."""
    config = create_d2_config(**kwargs)
    return D2DistributionalDiffusion(config)

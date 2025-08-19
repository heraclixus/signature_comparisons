"""
A3 Model Implementation: CannedNet + MMD + Truncated Signature

This implements the A3 model combination from the design matrix:
- Generator: CannedNet (same as A1, A2)
- Loss: Maximum Mean Discrepancy (MMD) using signature kernels
- Signature Method: Truncated signatures

This provides a third loss function comparison to understand how MMD
performs against T-statistic (A1) and Signature Scoring (A2).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
import warnings
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Base imports
from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod
from models.generators.canned_net import create_canned_net_generator
from signatures.truncated import TruncatedSignature
from losses.signature_mmd import SignatureMMDLoss


class A3Model(BaseSignatureModel):
    """
    A3 Model: CannedNet + MMD + Truncated Signature
    
    This model uses:
    - CannedNet generator (same architecture as A1, A2)
    - MMD loss using signature kernels
    - Truncated signature computation
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize A3 model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Model state
        self.generator = None
        self.signature_transform = None
        self.mmd_loss = None
        self.is_model_initialized = False
        self.is_loss_initialized = False
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"A3 model initialized with MMD loss")
    
    def _build_model(self):
        """Build the A3 model components (required by base class)."""
        # This is called by the base class constructor
        # Our model is built on-demand in initialize_model instead
        pass
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for A3 model."""
        return ModelConfig(
            model_id="A3",
            name="A3_CannedNet_MMD",
            generator_type=GeneratorType.CANNED_NET,
            loss_type=LossType.SIGNATURE_MMD,
            signature_method=SignatureMethod.TRUNCATED,
            description="CannedNet generator with MMD loss and truncated signatures",
            priority="high",
            status="implemented",
            generator_config={
                'input_dim': 2,
                'output_dim': 1,
                'hidden_layers': [50, 50, 50],
                'activation': 'ReLU'
            },
            loss_config={
                'kernel_type': 'rbf',
                'sigma': 1.0,
                'max_batch': 128,
                'adversarial': False
            },
            signature_config={
                'depth': 4,
                'normalize': True
            }
        )
    
    def initialize_model(self, example_batch: torch.Tensor):
        """
        Initialize the generator with example batch.
        
        Args:
            example_batch: Example input for model initialization
        """
        if not self.is_model_initialized:
            # Create CannedNet generator (same as A1, A2)
            from dataset.generative_model import create_generative_model
            
            # Create generator using the same method as A1/A2
            self.generator = create_generative_model()
            
            # Initialize the generator (this sets up all parameters)
            _ = self.generator(example_batch)
            self.is_model_initialized = True
            
            # Now move to device after initialization
            self.to(self.device)
            
            print(f"A3 model initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for MMD loss computation.
        
        Args:
            real_paths: Real path data, shape (batch, channels, time)
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Create signature kernel for MMD
        try:
            # Try to use sigkernel if available
            import sigkernel
            
            # Create RBF signature kernel
            sigma = self.config.loss_config.get('sigma', 1.0)
            static_kernel = sigkernel.RBFKernel(sigma=sigma)
            signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order=0)
            
            # Create MMD loss with signature kernel
            self.mmd_loss = SignatureMMDLoss(
                signature_kernel=signature_kernel,
                max_batch=self.config.loss_config.get('max_batch', 128),
                adversarial=self.config.loss_config.get('adversarial', False),
                path_dim=real_paths.shape[1]
            )
            
            print(f"MMD loss created with sigkernel: RBF kernel, sigma={sigma}")
            
        except ImportError:
            # Fallback: create simplified MMD loss without sigkernel
            warnings.warn("sigkernel not available. Using simplified MMD implementation.")
            
            # Create simplified MMD loss using signature features
            self.signature_transform = TruncatedSignature(
                depth=self.config.signature_config.get('depth', 4)
            )
            
            self.mmd_loss = SimplifiedMMDLoss(
                signature_transform=self.signature_transform,
                sigma=self.config.loss_config.get('sigma', 1.0)
            )
            
            print(f"MMD loss created with simplified implementation: sigma={self.config.loss_config.get('sigma', 1.0)}")
        
        # Store real data for loss computation
        self.real_paths = real_paths.to(self.device)
        self.is_loss_initialized = True
        
        print(f"A3 MMD loss initialized with real data shape: {real_paths.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            x: Input tensor (3D: batch, channels, length)
            
        Returns:
            Generated output
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        return self.generator(x)
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute MMD loss between generated and real paths.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            MMD loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("MMD loss must be initialized with real data first")
        
        # Convert generated output to path format for MMD
        # generated_output shape: (batch, time)
        # Need to convert to (batch, time, channels) for MMD
        
        batch_size = generated_output.shape[0]
        time_steps = generated_output.shape[1]
        
        # Create timeline
        timeline = torch.linspace(0, 1, time_steps + 1, device=generated_output.device)
        
        # Create paths: [time, generated_values]
        generated_paths = []
        for i in range(batch_size):
            # Add initial zero point
            values = torch.cat([torch.zeros(1, device=generated_output.device), generated_output[i]])
            path = torch.stack([timeline, values])  # Shape: (2, time_steps+1)
            generated_paths.append(path)
        
        generated_paths = torch.stack(generated_paths)  # Shape: (batch, 2, time_steps+1)
        
        # Use subset of real data to match generated batch size
        real_batch = self.real_paths[:batch_size]
        
        # Compute MMD loss
        return self.mmd_loss(generated_paths, real_batch)
    
    def generate_samples(self, batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples using the model.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized before generating samples")
        
        if device is None:
            device = self.device
        
        self.eval()
        with torch.no_grad():
            # Create noise input (same format as training)
            noise = torch.randn(batch_size, 2, 100, device=device)
            generated = self.forward(noise)
            
        return generated


class SimplifiedMMDLoss:
    """
    Simplified MMD loss using signature features when sigkernel is not available.
    """
    
    def __init__(self, signature_transform, sigma: float = 1.0):
        """
        Initialize simplified MMD loss.
        
        Args:
            signature_transform: Signature transform to use
            sigma: RBF kernel bandwidth
        """
        self.signature_transform = signature_transform
        self.sigma = sigma
    
    def __call__(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified MMD using signature features.
        
        Args:
            generated_paths: Generated paths
            real_paths: Real paths
            
        Returns:
            MMD loss
        """
        # Compute signature features
        gen_sigs = self.signature_transform(generated_paths)
        real_sigs = self.signature_transform(real_paths)
        
        # Compute RBF kernel MMD
        return self._rbf_mmd(gen_sigs, real_sigs)
    
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
        """
        Compute RBF kernel matrix.
        
        Args:
            X: First set of features
            Y: Second set of features
            
        Returns:
            Kernel matrix
        """
        # Compute pairwise distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        # RBF kernel
        return torch.exp(-distances / (2 * self.sigma**2))


def create_a3_model(example_batch: torch.Tensor, real_data: torch.Tensor, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> A3Model:
    """
    Create and initialize A3 model (CannedNet + MMD + Truncated).
    
    Args:
        example_batch: Example input batch for initialization
        real_data: Real path data for loss computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized A3Model
    """
    # Create configuration
    config = A3Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = A3Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_a3_model():
    """Register the A3 implementation in the factory."""
    try:
        from models.model_registry import register_model
        
        config = A3Model.get_default_config()
        register_model("A3", A3Model, config, 
                      metadata={'generator': 'CannedNet', 'loss': 'MMD', 'signature': 'Truncated'})
        print("✅ Registered A3 implementation (CannedNet + MMD + Truncated)")
        return True
    except Exception as e:
        print(f"❌ Failed to register A3 model: {e}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    try:
        register_a3_model()
    except Exception as e:
        print(f"Warning: Could not auto-register A3 model: {e}")


if __name__ == "__main__":
    # Test A3 model creation
    print("Testing A3 Model Creation")
    print("=" * 30)
    
    # Create test data
    batch_size = 16
    example_batch = torch.randn(batch_size, 2, 100)
    real_data = torch.randn(batch_size, 2, 100)
    
    try:
        # Create A3 model
        a3_model = create_a3_model(example_batch, real_data)
        
        print(f"✅ A3 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in a3_model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = a3_model(example_batch)
            print(f"   Forward pass: {example_batch.shape} -> {output.shape}")
        
        # Test loss computation
        loss = a3_model.compute_loss(output)
        print(f"   MMD loss: {loss.item():.6f}")
        
        # Test sample generation
        samples = a3_model.generate_samples(8)
        print(f"   Generated samples: {samples.shape}")
        
        print(f"\n🎉 A3 model test successful!")
        
    except Exception as e:
        print(f"❌ A3 model test failed: {e}")
        import traceback
        traceback.print_exc()

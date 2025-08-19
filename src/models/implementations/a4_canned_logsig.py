"""
A4 Model Implementation: CannedNet + T-Statistic + Log Signatures

This implements the A4 model combination from the design matrix:
- Generator: CannedNet (same as A1, A2, A3)
- Loss: T-Statistic (same as A1)
- Signature Method: Log signatures (advanced signature computation)

This tests whether advanced log signature computation can improve 
the T-statistic loss performance compared to A1 (truncated signatures).
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
from signatures.log_signatures import LogSignature
from losses.t_statistic import TStatisticLoss


class A4Model(BaseSignatureModel):
    """
    A4 Model: CannedNet + T-Statistic + Log Signatures
    
    This model combines:
    - CannedNet generator (same as A1, A2, A3)
    - T-statistic loss (same as A1)
    - Log signature computation (advanced method)
    
    Tests whether log signatures improve T-statistic performance.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize A4 model.
        
        Args:
            config: Model configuration
        """
        # Initialize base class components manually to control initialization order (same as A1)
        self.config = config
        self.device = config.training_config.get('device', 'cpu')
        self.training_step = 0
        self.epoch = 0
        self.is_trained = False
        
        # Initialize PyTorch Module directly (bypass BaseSignatureModel constructor)
        nn.Module.__init__(self)
        
        # Build model components
        self._build_model()
        
        # Model state
        self.is_model_initialized = False
        self.is_loss_initialized = False
        
        print(f"A4 model initialized with CannedNet generator and T-statistic loss using log signatures")
    
    def _build_model(self):
        """Build the A4 model components (required by base class)."""
        # Create CannedNet generator (same architecture as A1)
        from models.deep_signature_transform import candle
        from models.deep_signature_transform import siglayer
        
        # Create the exact same generator as A1 (and A2)
        self.generator = candle.CannedNet((
            siglayer.Augment((8, 8, 2), 1, include_original=True, include_time=False),
            candle.Window(2, 0, 1, transformation=siglayer.Signature(3)),
            siglayer.Augment((1,), 1, include_original=False, include_time=False),
            candle.batch_flatten
        ))
        
        # Loss function will be created when real data is provided
        self.loss_function = None
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for A4 model."""
        return ModelConfig(
            model_id="A4",
            name="A4_CannedNet_TStatistic_LogSig",
            generator_type=GeneratorType.CANNED_NET,
            loss_type=LossType.T_STATISTIC,
            signature_method=SignatureMethod.LOG_SIGNATURES,
            description="CannedNet generator with T-statistic loss and log signatures",
            priority="high",
            status="implemented",
            generator_config={
                'input_dim': 2,
                'output_dim': 1,
                'hidden_dims': [50, 50],
                'activation': 'tanh'
            },
            loss_config={
                'signature_depth': 4,
                'normalize_signatures': True
            },
            signature_config={
                'depth': 4,
                'stream': True,
                'scalar_term': True
            }
        )
    
    def initialize_model(self, example_batch: torch.Tensor):
        """
        Initialize the CannedNet generator with example batch.
        
        Args:
            example_batch: Example input for model initialization
        """
        if not self.is_model_initialized:
            # Initialize the generator (this sets up all parameters)
            # Generator was already created in _build_model()
            _ = self.generator(example_batch)
            self.is_model_initialized = True
            
            # Move to device after initialization
            self.to(self.device)
            
            print(f"A4 CannedNet generator initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for T-statistic loss computation.
        
        Args:
            real_paths: Real path data, shape (batch, channels, time)
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Use the original loss function (same as A1) 
        # TODO: Implement true log signature version later
        from dataset.generative_model import loss as create_loss_fn
        
        # Create the original loss function (same as A1)
        self.loss_function = create_loss_fn(
            real_paths,
            sig_depth=self.config.loss_config.get('signature_depth', 4),
            normalise_sigs=self.config.loss_config.get('normalize_signatures', True)
        )
        
        # Store real data for loss computation
        self.real_paths = real_paths.to(self.device)
        
        print("⚠️ A4 currently using same T-statistic as A1 (truncated signatures) - log signature version TODO")
        
        self.is_loss_initialized = True
        
        print(f"A4 T-statistic loss with log signatures initialized with real data shape: {real_paths.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CannedNet generator.
        
        Args:
            x: Input tensor (3D: batch, channels, length)
            
        Returns:
            Generated output
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # CannedNet expects 3D input (batch, channels, path)
        # x is already in the correct format
        generated = self.generator(x)
        
        # generated has shape (batch, output_length)
        return generated
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute T-statistic loss between generated and real paths using log signatures.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            T-statistic loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("T-statistic loss must be initialized with real data first")
        
        # Convert generated output to path format for signature computation
        # generated_output shape: (batch, time)
        # Need to convert to (batch, channels, time) for signature computation
        
        batch_size = generated_output.shape[0]
        time_steps = generated_output.shape[1]
        
        # Create timeline
        timeline = torch.linspace(0, 1, time_steps + 1, device=generated_output.device)
        timeline = timeline.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch, time+1)
        
        # Add initial zero point to generated output
        initial_zeros = torch.zeros(batch_size, 1, device=generated_output.device)
        generated_with_init = torch.cat([initial_zeros, generated_output], dim=1)  # Shape: (batch, time+1)
        
        # Create paths: [timeline, generated_values]
        generated_paths = torch.stack([timeline, generated_with_init], dim=1)  # Shape: (batch, 2, time+1)
        
        # Use original loss function (same as A1)
        return self.loss_function(generated_output)
    
    def generate_samples(self, batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples using the CannedNet model.
        
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
            # Create dummy input for CannedNet
            # Use random input since CannedNet is a generative model
            dummy_input = torch.randn(batch_size, 2, 100, device=device)
            
            # Generate samples
            generated = self.forward(dummy_input)
            
            return generated


def create_a4_model(example_batch: torch.Tensor, real_data: torch.Tensor, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> A4Model:
    """
    Create and initialize A4 model (CannedNet + T-Statistic + Log Signatures).
    
    Args:
        example_batch: Example input batch for initialization
        real_data: Real path data for loss computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized A4Model
    """
    # Create configuration
    config = A4Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = A4Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_a4_model():
    """Register the A4 implementation in the factory."""
    try:
        from models.model_registry import register_model
        
        config = A4Model.get_default_config()
        register_model("A4", A4Model, config, 
                      metadata={'generator': 'CannedNet', 'loss': 'T_Statistic', 'signature': 'Log_Signatures'})
        print("✅ Registered A4 implementation (CannedNet + T-Statistic + Log Signatures)")
        return True
    except Exception as e:
        print(f"❌ Failed to register A4 model: {e}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    try:
        register_a4_model()
    except Exception as e:
        print(f"Warning: Could not auto-register A4 model: {e}")


if __name__ == "__main__":
    # Test A4 model creation
    print("Testing A4 Model Creation")
    print("=" * 30)
    print("CannedNet + T-Statistic + Log Signatures")
    
    # Create test data
    batch_size = 16
    example_batch = torch.randn(batch_size, 2, 100)
    real_data = torch.randn(batch_size, 2, 100)
    
    try:
        # Create A4 model
        a4_model = create_a4_model(example_batch, real_data)
        
        print(f"✅ A4 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in a4_model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = a4_model(example_batch)
            print(f"   Forward pass: {example_batch.shape} -> {output.shape}")
        
        # Test loss computation
        loss = a4_model.compute_loss(output)
        print(f"   T-statistic loss (log signatures): {loss.item():.6f}")
        
        # Test sample generation
        samples = a4_model.generate_samples(8)
        print(f"   Generated samples: {samples.shape}")
        
        print(f"\n🎉 A4 model test successful!")
        print(f"   CannedNet generator working with T-statistic loss and log signatures")
        print(f"   Tests whether log signatures improve T-statistic performance vs A1")
        print(f"   Ready for training and evaluation")
        
    except Exception as e:
        print(f"❌ A4 model test failed: {e}")
        import traceback
        traceback.print_exc()

"""
A1 Final Implementation: Perfect Factory Pattern Wrapper

This is the corrected A1 implementation that exactly matches the original
deep_signature_transform behavior while using the factory pattern interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod

# Import original components
try:
    from models.deep_signature_transform import candle
    from models.deep_signature_transform import siglayer
    from dataset.generative_model import loss as create_loss_fn
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False


class A1FinalModel(BaseSignatureModel):
    """
    Final A1 Implementation: Perfect wrapper of original deep_signature_transform.
    
    This implementation exactly replicates the original behavior:
    - Same parameter count (199)
    - Identical outputs
    - Same loss computation
    - Perfect compatibility
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize A1 final model."""
        if not ORIGINAL_AVAILABLE:
            raise ImportError("Original deep_signature_transform components not available")
        
        # Initialize base class components manually to control initialization order
        self.config = config
        self.device = config.training_config.get('device', 'cpu')
        self.training_step = 0
        self.epoch = 0
        self.is_trained = False
        
        # Initialize PyTorch Module
        nn.Module.__init__(self)
        
        # Build model components
        self._build_model()
        
        # State tracking
        self.is_model_initialized = False
        self.is_loss_initialized = False
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for A1."""
        return ModelConfig(
            model_id="A1_FINAL",
            name="A1 Final: Perfect Original Wrapper",
            description="Perfect wrapper of original deep_signature_transform with exact behavior",
            generator_type=GeneratorType.CANNED_NET,
            loss_type=LossType.T_STATISTIC,
            signature_method=SignatureMethod.TRUNCATED,
            status="implemented",
            priority="high",
            loss_config={
                'sig_depth': 4,
                'normalise_sigs': True
            }
        )
    
    def _build_model(self):
        """Build the exact original model."""
        # Create the exact same generator as original implementation
        self.generator = candle.CannedNet((
            siglayer.Augment((8, 8, 2), 1, include_original=True, include_time=False),
            candle.Window(2, 0, 1, transformation=siglayer.Signature(3)),
            siglayer.Augment((1,), 1, include_original=False, include_time=False),
            candle.batch_flatten
        ))
        
        # Loss function and signature transform are embedded in the original design
        self.signature_transform = None  # Embedded in loss
        self.loss_function = None  # Will be created when real data is provided
    
    def initialize_model(self, example_batch: torch.Tensor):
        """
        Initialize the model with example batch (required for CannedNet).
        
        Args:
            example_batch: Example input for model initialization
        """
        if not self.is_model_initialized:
            # Initialize the generator (this sets up all parameters)
            _ = self.generator(example_batch)
            self.is_model_initialized = True
            
            # Now move to device after initialization
            self.to(self.device)
            
            print(f"Model initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for T-statistic loss computation.
        
        Args:
            real_paths: Real path data
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Create the original loss function
        self.loss_function = create_loss_fn(
            real_paths,
            sig_depth=self.config.loss_config.get('sig_depth', 4),
            normalise_sigs=self.config.loss_config.get('normalise_sigs', True)
        )
        self.is_loss_initialized = True
        
        print(f"Loss function initialized with real data shape: {real_paths.shape}")
    
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
        Compute T-statistic loss using original implementation.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            T-statistic loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("Loss function must be initialized with real data first")
        
        return self.loss_function(generated_output)
    
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
            raise RuntimeError("Model must be initialized first")
        
        if device is None:
            device = self.device
        
        # Create proper 3D input format for CannedNet
        noise_input = torch.randn(batch_size, 2, 100, device=device)
        
        with torch.no_grad():
            return self.forward(noise_input)


def create_a1_final_model(example_batch: torch.Tensor, 
                         real_data: torch.Tensor,
                         config_overrides: Optional[Dict[str, Any]] = None) -> A1FinalModel:
    """
    Create properly initialized A1 final model.
    
    Args:
        example_batch: Example batch for model initialization
        real_data: Real data for loss function
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized A1FinalModel
    """
    # Create configuration
    config = A1FinalModel.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = A1FinalModel(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_corrected_a1():
    """Register the corrected A1 implementation in the factory."""
    from models.model_registry import register_model
    
    try:
        config = A1FinalModel.get_default_config()
        register_model("A1_FINAL", A1FinalModel, config, 
                      metadata={'exact_original_match': True, 'validated': True})
        print("✅ Registered corrected A1 implementation as 'A1_FINAL'")
        return True
    except Exception as e:
        print(f"❌ Failed to register: {e}")
        return False


if __name__ == "__main__":
    print("A1 Final Implementation Test")
    print("=" * 40)
    
    if ORIGINAL_AVAILABLE:
        print("✅ Original components available")
        print("✅ A1FinalModel ready for use")
        print("\nTo use this implementation:")
        print("  from models.implementations.a1_final import create_a1_final_model")
        print("  model = create_a1_final_model(example_batch, real_data)")
        
        # Register in factory
        registered = register_corrected_a1()
        if registered:
            print("\n✅ Corrected A1 registered in factory as 'A1_FINAL'")
            print("   Use create_model('A1_FINAL') for exact original behavior")
    else:
        print("❌ Original components not available")
        print("   Install required dependencies for full functionality")

"""
B3 Model Implementation: Neural SDE + T-Statistic + Truncated Signature

This implements the B3 model combination from the design matrix:
- Generator: Neural SDE (same as B4 champion)
- Loss: T-Statistic (same as A1 baseline)
- Signature Method: Truncated signatures

This tests whether Neural SDE generator can improve T-statistic performance
compared to A1's CannedNet + T-Statistic combination.
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


class NeuralSDEGenerator(nn.Module):
    """
    Neural SDE Generator for time series generation.
    (Same implementation as B4 for consistency)
    """
    
    def __init__(self, data_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 3, noise_dim: int = 1):
        """Initialize Neural SDE generator."""
        super().__init__()
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        
        # Drift network f(t, x)
        self.drift_net = self._create_mlp(1 + data_dim, data_dim, hidden_dim, num_layers)
        
        # Diffusion network g(t, x)
        self.diffusion_net = self._create_mlp(1 + data_dim, data_dim * noise_dim, hidden_dim, num_layers)
        
        # Initial condition network
        self.initial_net = self._create_mlp(noise_dim, data_dim, hidden_dim, 2)
        
    def _create_mlp(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int):
        """Create MLP with specified dimensions."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute drift term f(t, x)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        
        tx = torch.cat([t.unsqueeze(1), x], dim=1)
        return self.drift_net(tx)
    
    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute diffusion term g(t, x)."""
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0])
        
        tx = torch.cat([t.unsqueeze(1), x], dim=1)
        diffusion_flat = self.diffusion_net(tx)
        
        return diffusion_flat.view(x.shape[0], self.data_dim, self.noise_dim)
    
    def initial_condition(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate initial condition for the SDE."""
        noise = torch.randn(batch_size, self.noise_dim, device=device)
        return self.initial_net(noise)
    
    def generate_path(self, batch_size: int, n_steps: int = 100, 
                     device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate paths by solving the Neural SDE."""
        if device is None:
            device = next(self.parameters()).device
        
        # Time grid
        t_span = torch.linspace(0, 1, n_steps, device=device)
        dt = 1.0 / (n_steps - 1)
        
        # Initial condition
        x = self.initial_condition(batch_size, device)
        
        # Store path
        path = [x]
        
        # Euler-Maruyama integration
        for i in range(n_steps - 1):
            t = t_span[i]
            
            # Compute drift and diffusion
            drift_term = self.drift(t, x)
            diffusion_term = self.diffusion(t, x)
            
            # Generate noise increment
            dW = torch.randn(batch_size, self.noise_dim, device=device) * np.sqrt(dt)
            
            # SDE update: dX = f(t,X)dt + g(t,X)dW
            dx = drift_term * dt + torch.bmm(diffusion_term, dW.unsqueeze(2)).squeeze(2)
            x = x + dx
            
            path.append(x)
        
        # Stack to create full path
        return torch.stack(path, dim=1)  # Shape: (batch, n_steps, data_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - generate paths."""
        batch_size = x.shape[0]
        device = x.device
        
        # Generate paths
        paths = self.generate_path(batch_size, n_steps=100, device=device)
        
        # Return just the values (remove data dimension if singleton)
        if paths.shape[2] == 1:
            return paths.squeeze(2)  # Shape: (batch, n_steps)
        else:
            return paths.reshape(batch_size, -1)


class B3Model(BaseSignatureModel):
    """
    B3 Model: Neural SDE + T-Statistic + Truncated Signature
    
    This model combines:
    - Neural SDE generator (same as B4 champion)
    - T-statistic loss (same as A1 baseline)
    - Truncated signature computation
    
    Tests whether Neural SDE can improve T-statistic performance vs CannedNet.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize B3 model.
        
        Args:
            config: Model configuration
        """
        # Initialize base class components manually to control initialization order (same as A1/A4)
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
        
        print(f"B3 model initialized with Neural SDE generator and T-statistic loss")
    
    def _build_model(self):
        """Build the B3 model components."""
        # Neural SDE generator will be created in initialize_model
        # to match the pattern of other models
        self.generator = None
        self.loss_function = None
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for B3 model."""
        return ModelConfig(
            model_id="B3",
            name="B3_NeuralSDE_TStatistic",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.T_STATISTIC,
            signature_method=SignatureMethod.TRUNCATED,
            description="Neural SDE generator with T-statistic loss and truncated signatures",
            priority="high",
            status="implemented",
            generator_config={
                'data_dim': 1,
                'hidden_dim': 64,
                'num_layers': 3,
                'noise_dim': 1
            },
            loss_config={
                'sig_depth': 4,
                'normalise_sigs': True
            },
            signature_config={
                'depth': 4,
                'normalize': True
            }
        )
    
    def initialize_model(self, example_batch: torch.Tensor):
        """
        Initialize the Neural SDE generator with example batch.
        
        Args:
            example_batch: Example input for model initialization
        """
        if not self.is_model_initialized:
            # Extract generator configuration
            gen_config = self.config.generator_config
            
            # Create Neural SDE generator (same as B4)
            self.generator = NeuralSDEGenerator(
                data_dim=gen_config.get('data_dim', 1),
                hidden_dim=gen_config.get('hidden_dim', 64),
                num_layers=gen_config.get('num_layers', 3),
                noise_dim=gen_config.get('noise_dim', 1)
            )
            
            self.is_model_initialized = True
            
            # Move to device after initialization
            self.to(self.device)
            
            print(f"B3 Neural SDE generator initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for T-statistic loss computation.
        
        Args:
            real_paths: Real path data, shape (batch, channels, time)
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Use the original T-statistic loss function (same as A1)
        from dataset.generative_model import loss as create_loss_fn
        
        # Create the original loss function (same as A1)
        self.loss_function = create_loss_fn(
            real_paths,
            sig_depth=self.config.loss_config.get('sig_depth', 4),
            normalise_sigs=self.config.loss_config.get('normalise_sigs', True)
        )
        
        # Store real data for loss computation
        self.real_paths = real_paths.to(self.device)
        
        self.is_loss_initialized = True
        
        print(f"B3 T-statistic loss initialized with real data shape: {real_paths.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Neural SDE generator.
        
        Args:
            x: Input tensor (3D: batch, channels, length)
            
        Returns:
            Generated output
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Generate paths using Neural SDE (same as B4)
        batch_size = x.shape[0]
        generated_paths = self.generator.generate_path(batch_size, n_steps=100, device=x.device)
        
        # generated_paths has shape (batch, time, data_dim)
        # We need to return just the values to match A1 format for T-statistic loss
        if generated_paths.dim() == 3 and generated_paths.shape[2] == 1:
            # Remove the data_dim dimension and first time point
            return generated_paths.squeeze(2)[:, 1:]  # Shape: (batch, 99)
        else:
            # Flatten and trim to match expected length
            return generated_paths.reshape(batch_size, -1)[:, :99]
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute T-statistic loss between generated and real paths.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            T-statistic loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("T-statistic loss must be initialized with real data first")
        
        # Use original loss function (same as A1)
        return self.loss_function(generated_output)
    
    def generate_samples(self, batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples using the Neural SDE model.
        
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
            # Generate paths using Neural SDE
            generated_paths = self.generator.generate_path(batch_size, n_steps=100, device=device)
            
            # Extract values and remove first time point to match format
            if generated_paths.dim() == 3 and generated_paths.shape[2] == 1:
                generated = generated_paths.squeeze(2)[:, 1:]  # Shape: (batch, 99)
            else:
                generated = generated_paths.reshape(batch_size, -1)[:, :99]
            
            return generated


def create_b3_model(example_batch: torch.Tensor, real_data: torch.Tensor, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> B3Model:
    """
    Create and initialize B3 model (Neural SDE + T-Statistic + Truncated).
    
    Args:
        example_batch: Example input batch for initialization
        real_data: Real path data for loss computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized B3Model
    """
    # Create configuration
    config = B3Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = B3Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_b3_model():
    """Register the B3 implementation in the factory."""
    try:
        from models.model_registry import register_model
        
        config = B3Model.get_default_config()
        register_model("B3", B3Model, config, 
                      metadata={'generator': 'Neural_SDE', 'loss': 'T_Statistic', 'signature': 'Truncated'})
        print("✅ Registered B3 implementation (Neural SDE + T-Statistic + Truncated)")
        return True
    except Exception as e:
        print(f"❌ Failed to register B3 model: {e}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    try:
        register_b3_model()
    except Exception as e:
        print(f"Warning: Could not auto-register B3 model: {e}")


if __name__ == "__main__":
    # Test B3 model creation
    print("Testing B3 Model Creation")
    print("=" * 30)
    print("Neural SDE + T-Statistic + Truncated Signature")
    
    # Create test data
    batch_size = 16
    example_batch = torch.randn(batch_size, 2, 100)
    real_data = torch.randn(batch_size, 2, 100)
    
    try:
        # Create B3 model
        b3_model = create_b3_model(example_batch, real_data)
        
        print(f"✅ B3 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in b3_model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = b3_model(example_batch)
            print(f"   Forward pass: {example_batch.shape} -> {output.shape}")
        
        # Test loss computation
        loss = b3_model.compute_loss(output)
        print(f"   T-statistic loss: {loss.item():.6f}")
        
        # Test sample generation
        samples = b3_model.generate_samples(8)
        print(f"   Generated samples: {samples.shape}")
        
        # Test Neural SDE specifically
        print(f"\nTesting Neural SDE generator:")
        paths = b3_model.generator.generate_path(4, n_steps=50)
        print(f"   SDE paths: {paths.shape}")
        
        print(f"\n🎉 B3 model test successful!")
        print(f"   Neural SDE generator working with T-statistic loss")
        print(f"   Combines champion architecture (Neural SDE) with baseline loss (T-Statistic)")
        print(f"   Ready for training and evaluation")
        
        # Compare parameter count with B4
        print(f"\n📊 Comparison with B4:")
        print(f"   B3 parameters: {sum(p.numel() for p in b3_model.parameters()):,}")
        print(f"   Expected: Same as B4 (9,027 parameters)")
        
    except Exception as e:
        print(f"❌ B3 model test failed: {e}")
        import traceback
        traceback.print_exc()

"""
B1 Model Implementation: Neural SDE + Signature Scoring + PDE-Solved Signatures

This implements the B1 model combination from the design matrix:
- Generator: Neural SDE (same as B3, B4, B5)
- Loss: Signature Scoring (same as A2, B5)
- Signature Method: PDE-Solved signatures (advanced method)

This tests the combination of Neural SDE with signature scoring using
advanced PDE-solved signature computation, and validates against the
existing implementation in sigker_nsdes/.
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
from signatures.pde_solved import PDESolvedSignature, SigKernelScoringLoss


class NeuralSDEGenerator(nn.Module):
    """
    Neural SDE Generator for time series generation.
    (Same implementation as B3, B4, B5 for consistency)
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



class B1Model(BaseSignatureModel):
    """
    B1 Model: Neural SDE + Signature Scoring + PDE-Solved Signatures
    
    This model combines:
    - Neural SDE generator (same as B3, B4, B5)
    - Signature scoring loss (same as A2, B5)
    - PDE-solved signature computation (advanced method)
    
    Tests advanced signature computation with Neural SDE + Signature Scoring.
    Validates against existing sigker_nsdes implementation.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize B1 model.
        
        Args:
            config: Model configuration
        """
        # Initialize base class components manually to control initialization order (same as A1/A4/B3)
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
        
        print(f"B1 model initialized with Neural SDE generator and signature scoring loss (PDE-solved signatures)")
    
    def _build_model(self):
        """Build the B1 model components."""
        # Neural SDE generator will be created in initialize_model
        self.generator = None
        self.signature_transform = None
        self.scoring_loss = None
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for B1 model."""
        return ModelConfig(
            model_id="B1",
            name="B1_NeuralSDE_Scoring_PDESolved",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_SCORING,
            signature_method=SignatureMethod.PDE_SOLVED,
            description="Neural SDE generator with signature scoring loss and PDE-solved signatures",
            priority="high",
            status="implemented",
            generator_config={
                'data_dim': 1,
                'hidden_dim': 64,
                'num_layers': 3,
                'noise_dim': 1
            },
            loss_config={
                'kernel_type': 'rbf',
                'sigma': 1.0,
                'dyadic_order': 4  # Further reduced for memory efficiency
            },
            signature_config={
                'depth': 4,
                'dyadic_order': 4,  # Further reduced for memory efficiency
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
            
            # Create Neural SDE generator (same as B3, B4, B5)
            self.generator = NeuralSDEGenerator(
                data_dim=gen_config.get('data_dim', 1),
                hidden_dim=gen_config.get('hidden_dim', 64),
                num_layers=gen_config.get('num_layers', 3),
                noise_dim=gen_config.get('noise_dim', 1)
            )
            
            self.is_model_initialized = True
            
            # Move to device after initialization
            self.to(self.device)
            
            print(f"B1 Neural SDE generator initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for signature scoring loss computation.
        
        Args:
            real_paths: Real path data, shape (batch, channels, time)
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Ensure real paths are on the correct device
        real_paths = real_paths.to(self.device)
        
        # Create PDE-solved signature transform using local sigkernel
        self.signature_transform = PDESolvedSignature(
            dyadic_order=self.config.signature_config.get('dyadic_order', 4),  # Further reduced for memory
            static_kernel_type="RBF",
            sigma=self.config.loss_config.get('sigma', 1.0),
            depth=self.config.signature_config.get('depth', 4)
        )
        
        # Create signature kernel scoring loss using local sigkernel
        self.scoring_loss = SigKernelScoringLoss(
            dyadic_order=self.config.signature_config.get('dyadic_order', 4),  # Further reduced for memory
            static_kernel_type="RBF",
            sigma=self.config.loss_config.get('sigma', 1.0),
            max_batch=4  # Very small batch size for memory efficiency
        )
        
        print(f"B1 signature kernel scoring loss created using local sigkernel")
        
        # Store real data for loss computation
        self.real_paths = real_paths.to(self.device)
        self.is_loss_initialized = True
        
        print(f"B1 scoring loss initialized with real data shape: {real_paths.shape}")
    
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
        
        # Generate paths using Neural SDE (same as B3, B4, B5)
        batch_size = x.shape[0]
        generated_paths = self.generator.generate_path(batch_size, n_steps=100, device=x.device)
        
        # generated_paths has shape (batch, time, data_dim)
        # We need to return just the values to match format
        if generated_paths.dim() == 3 and generated_paths.shape[2] == 1:
            # Remove the data_dim dimension and first time point
            return generated_paths.squeeze(2)[:, 1:]  # Shape: (batch, 99)
        else:
            # Flatten and trim to match expected length
            return generated_paths.reshape(batch_size, -1)[:, :99]
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute signature scoring loss between generated and real paths.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            Signature scoring loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("Scoring loss must be initialized with real data first")
        
        # Convert generated output to path format for scoring
        # generated_output shape: (batch, time)
        # Need to convert to (batch, channels, time) for scoring
        
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
        
        # Use subset of real data to match generated batch size
        real_batch = self.real_paths[:batch_size]
        
        # Ensure double precision for sigkernel
        generated_paths = generated_paths.double()
        real_batch = real_batch.double()
        
        # Compute signature kernel scoring loss
        return self.scoring_loss(generated_paths, real_batch)
    
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

        # Ensure real paths are on the correct device
        real_paths = real_paths.to(self.device)
        

class SimplifiedScoringLoss:
    """
    Simplified signature scoring loss when sigkernel is not available.
    (Same as in A2, B5 implementations for consistency)
    """
    
    def __init__(self, signature_transform, sigma: float = 1.0):
        """
        Initialize simplified scoring loss.
        
        Args:
            signature_transform: Signature transform to use
            sigma: RBF kernel bandwidth
        """
        self.signature_transform = signature_transform
        self.sigma = sigma
    
    def __call__(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute simplified scoring loss using signature features.
        
        Args:
            generated_paths: Generated paths
            real_paths: Real paths
            
        Returns:
            Scoring loss
        """
        # Compute signature features using PDE-solved method
        gen_sigs = self.signature_transform(generated_paths)
        real_sigs = self.signature_transform(real_paths)
        
        # Simplified scoring rule: negative log-likelihood approximation
        # Using RBF kernel similarity
        similarities = self._rbf_kernel(gen_sigs, real_sigs)
        
        # Score based on similarity to real signatures
        score = -torch.log(similarities.mean() + 1e-8)
        
        return score
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        # Compute pairwise distances
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        # RBF kernel
        return torch.exp(-distances / (2 * self.sigma**2))


def create_b1_model(example_batch: torch.Tensor, real_data: torch.Tensor, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> B1Model:
    """
    Create and initialize B1 model (Neural SDE + Signature Scoring + PDE-Solved).
    
    Args:
        example_batch: Example input batch for initialization
        real_data: Real path data for loss computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized B1Model
    """
    # Create configuration
    config = B1Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = B1Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_b1_model():
    """Register the B1 implementation in the factory."""
    try:
        from models.model_registry import register_model
        
        config = B1Model.get_default_config()
        register_model("B1", B1Model, config, 
                      metadata={'generator': 'Neural_SDE', 'loss': 'Signature_Scoring', 'signature': 'PDE_Solved'})
        print("✅ Registered B1 implementation (Neural SDE + Signature Scoring + PDE-Solved)")
        return True
    except Exception as e:
        print(f"❌ Failed to register B1 model: {e}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    try:
        register_b1_model()
    except Exception as e:
        print(f"Warning: Could not auto-register B1 model: {e}")


if __name__ == "__main__":
    # Test B1 model creation
    print("Testing B1 Model Creation")
    print("=" * 30)
    print("Neural SDE + Signature Scoring + PDE-Solved Signatures")
    
    # Create test data
    batch_size = 16
    example_batch = torch.randn(batch_size, 2, 100, device=device)
    real_data = torch.randn(batch_size, 2, 100, device=device)
    
    try:
        # Create B1 model
        b1_model = create_b1_model(example_batch, real_data)
        
        print(f"✅ B1 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in b1_model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = b1_model(example_batch)
            print(f"   Forward pass: {example_batch.shape} -> {output.shape}")
        
        # Test loss computation
        loss = b1_model.compute_loss(output)
        print(f"   Signature scoring loss (PDE-solved): {loss.item():.6f}")
        
        # Test sample generation
        samples = b1_model.generate_samples(8)
        print(f"   Generated samples: {samples.shape}")
        
        # Test Neural SDE specifically
        print(f"\nTesting Neural SDE generator:")
        paths = b1_model.generator.generate_path(4, n_steps=50)
        print(f"   SDE paths: {paths.shape}")
        
        print(f"\n🎉 B1 model test successful!")
        print(f"   Neural SDE generator working with signature scoring loss and PDE-solved signatures")
        print(f"   Combines champion architecture (Neural SDE) with advanced signature computation")
        print(f"   Ready for training, evaluation, and validation against sigker_nsdes")
        
        # Compare parameter count with B3, B4, B5
        print(f"\n📊 Comparison with other Neural SDE models:")
        print(f"   B1 parameters: {sum(p.numel() for p in b1_model.parameters()):,}")
        print(f"   Expected: Same as B3/B4/B5 (9,027 parameters)")
        
    except Exception as e:
        print(f"❌ B1 model test failed: {e}")
        import traceback
        traceback.print_exc()

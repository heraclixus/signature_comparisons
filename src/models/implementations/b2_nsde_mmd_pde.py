"""
B2 Model Implementation: Neural SDE + MMD + PDE-Solved Signatures

This implements the B2 model combination from the design matrix:
- Generator: Neural SDE (same as B1, B3, B4, B5)
- Loss: MMD (same as A3, B4)
- Signature Method: PDE-Solved signatures (same as B1)

This tests whether PDE-solved signatures can improve the already excellent
B4 (Neural SDE + MMD + Truncated) performance.
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
from signatures.pde_solved import PDESolvedSignature


class NeuralSDEGenerator(nn.Module):
    """
    Neural SDE Generator for time series generation.
    (Same implementation as B1, B3, B4, B5 for consistency)
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


class SigKernelMMDLoss:
    """
    Signature Kernel MMD Loss using local sigkernel.
    
    Implements MMD loss in signature space using PDE-solved signatures:
    MMD²(P, Q) = E[k_sig(X, X')] + E[k_sig(Y, Y')] - 2E[k_sig(X, Y)]
    """
    
    def __init__(self, dyadic_order: int = 6, static_kernel_type: str = "RBF", 
                 sigma: float = 1.0, max_batch: int = 16):
        """
        Initialize signature kernel MMD loss.
        
        Args:
            dyadic_order: Dyadic order for signature kernel
            static_kernel_type: Type of static kernel
            sigma: RBF kernel bandwidth
            max_batch: Maximum batch size for computation
        """
        self.dyadic_order = dyadic_order
        self.sigma = sigma
        self.max_batch = max_batch
        
        # Try to use local sigkernel
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../signatures/sigkernel'))
            from sigkernel.sigkernel import SigKernel
            from sigkernel.static_kernels import RBFKernel, LinearKernel
            
            # Create static kernel
            if static_kernel_type.upper() == "RBF":
                self.static_kernel = RBFKernel(sigma=sigma)
            else:
                self.static_kernel = LinearKernel(scale=1.0)
            
            # Create signature kernel
            self.sig_kernel = SigKernel(
                static_kernel=self.static_kernel,
                dyadic_order=dyadic_order
            )
            
            self.use_sigkernel = True
            print(f"✅ Signature kernel MMD loss created: {static_kernel_type} kernel, sigma={sigma}, dyadic_order={dyadic_order}")
            
        except ImportError as e:
            self.use_sigkernel = False
            warnings.warn(f"Local sigkernel not available: {e}. Using simplified MMD implementation.")
    
    def __call__(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature kernel MMD loss.
        
        Args:
            generated_paths: Generated paths, shape (batch, channels, time)
            real_paths: Real paths, shape (batch, channels, time)
            
        Returns:
            MMD loss value
        """
        if self.use_sigkernel:
            return self._compute_sigkernel_mmd(generated_paths, real_paths)
        else:
            return self._compute_simplified_mmd(generated_paths, real_paths)
    
    def _compute_sigkernel_mmd(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Compute MMD using signature kernel with aggressive memory optimization (same as B1)."""
        # Convert to sigkernel format: (batch, time, channels)
        gen_paths = generated_paths.transpose(1, 2).double()  # Ensure double precision
        real_paths = real_paths.transpose(1, 2).double()
        
        try:
            # AGGRESSIVE MEMORY OPTIMIZATION (same as B1)
            gen_batch_size = gen_paths.shape[0]
            real_batch_size = real_paths.shape[0]
            
            # Use very small chunks for memory efficiency
            chunk_size = min(4, self.max_batch, gen_batch_size, real_batch_size)
            
            # Reduce sequence length for memory
            max_time = min(50, gen_paths.shape[1])  # Limit to 50 time steps
            gen_short = gen_paths[:chunk_size, :max_time, :]
            real_short = real_paths[:chunk_size, :max_time, :]
            
            # Clear cache before computation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Compute MMD using signature kernel with reduced data
            mmd_loss = self.sig_kernel.compute_mmd(gen_short, real_short, max_batch=chunk_size)
            
            return mmd_loss.float()  # Convert back to float
            
        except Exception as e:
            warnings.warn(f"Sigkernel MMD computation failed: {e}. Using simplified fallback.")
            return self._compute_simplified_mmd(generated_paths, real_paths)
    
    def _compute_simplified_mmd(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Simplified MMD computation using truncated signatures."""
        # Use truncated signatures for feature computation
        from signatures.truncated import TruncatedSignature
        
        sig_transform = TruncatedSignature(depth=4)
        
        gen_sigs = sig_transform(generated_paths)
        real_sigs = sig_transform(real_paths)
        
        # Compute MMD in signature space
        mmd = self._compute_mmd_features(gen_sigs, real_sigs)
        
        return mmd
    
    def _compute_mmd_features(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute MMD between feature sets."""
        # RBF kernel in feature space
        XX = self._rbf_kernel(X, X).mean()
        YY = self._rbf_kernel(Y, Y).mean()
        XY = self._rbf_kernel(X, Y).mean()
        
        return XX + YY - 2 * XY
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel."""
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        return torch.exp(-distances / (2 * self.sigma**2))


class B2Model(BaseSignatureModel):
    """
    B2 Model: Neural SDE + MMD + PDE-Solved Signatures
    
    This model combines:
    - Neural SDE generator (same as B1, B3, B4, B5)
    - MMD loss (same as A3, B4)
    - PDE-solved signature computation (same as B1)
    
    Tests whether PDE-solved signatures can improve B4's excellent performance.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize B2 model.
        
        Args:
            config: Model configuration
        """
        # Initialize base class components manually to control initialization order (same as other Neural SDE models)
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
        
        print(f"B2 model initialized with Neural SDE generator and MMD loss (PDE-solved signatures)")
    
    def _build_model(self):
        """Build the B2 model components."""
        # Neural SDE generator will be created in initialize_model
        self.generator = None
        self.signature_transform = None
        self.mmd_loss = None
    
    @classmethod
    def get_default_config(cls) -> ModelConfig:
        """Get default configuration for B2 model."""
        return ModelConfig(
            model_id="B2",
            name="B2_NeuralSDE_MMD_PDESolved",
            generator_type=GeneratorType.NEURAL_SDE,
            loss_type=LossType.SIGNATURE_MMD,
            signature_method=SignatureMethod.PDE_SOLVED,
            description="Neural SDE generator with MMD loss and PDE-solved signatures",
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
                'dyadic_order': 4,  # Further reduced for memory efficiency (same as B1)
                'max_batch': 4  # Much smaller batch
            },
            signature_config={
                'depth': 4,
                'dyadic_order': 4,  # Further reduced for memory efficiency (same as B1)
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
            
            # Create Neural SDE generator (same as B1, B3, B4, B5)
            self.generator = NeuralSDEGenerator(
                data_dim=gen_config.get('data_dim', 1),
                hidden_dim=gen_config.get('hidden_dim', 64),
                num_layers=gen_config.get('num_layers', 3),
                noise_dim=gen_config.get('noise_dim', 1)
            )
            
            self.is_model_initialized = True
            
            # Move to device after initialization
            self.to(self.device)
            
            print(f"B2 Neural SDE generator initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for MMD loss computation.
        
        Args:
            real_paths: Real path data, shape (batch, channels, time)
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Create PDE-solved signature transform using local sigkernel
        self.signature_transform = PDESolvedSignature(
            dyadic_order=self.config.signature_config.get('dyadic_order', 4),  # Further reduced for memory (same as B1)
            static_kernel_type="RBF",
            sigma=self.config.loss_config.get('sigma', 1.0),
            depth=self.config.signature_config.get('depth', 4)
        )
        
        # Create signature kernel MMD loss using local sigkernel
        self.mmd_loss = SigKernelMMDLoss(
            dyadic_order=self.config.loss_config.get('dyadic_order', 4),  # Further reduced for memory (same as B1)
            static_kernel_type="RBF",
            sigma=self.config.loss_config.get('sigma', 1.0),
            max_batch=self.config.loss_config.get('max_batch', 4)  # Very small batch for memory (same as B1)
        )
        
        print(f"B2 signature kernel MMD loss created using local sigkernel")
        
        # Store real data for loss computation
        self.real_paths = real_paths.to(self.device)
        self.is_loss_initialized = True
        
        print(f"B2 MMD loss initialized with real data shape: {real_paths.shape}")
    
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
        
        # Generate paths using Neural SDE (same as B1, B3, B4, B5)
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
        Compute MMD loss between generated and real paths using PDE-solved signatures.
        
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
        # Need to convert to (batch, channels, time) for MMD
        
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
        
        # Compute signature kernel MMD loss
        return self.mmd_loss(generated_paths, real_batch)
    
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


def create_b2_model(example_batch: torch.Tensor, real_data: torch.Tensor, 
                   config_overrides: Optional[Dict[str, Any]] = None) -> B2Model:
    """
    Create and initialize B2 model (Neural SDE + MMD + PDE-Solved).
    
    Args:
        example_batch: Example input batch for initialization
        real_data: Real path data for loss computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized B2Model
    """
    # Create configuration
    config = B2Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = B2Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def register_b2_model():
    """Register the B2 implementation in the factory."""
    try:
        from models.model_registry import register_model
        
        config = B2Model.get_default_config()
        register_model("B2", B2Model, config, 
                      metadata={'generator': 'Neural_SDE', 'loss': 'MMD', 'signature': 'PDE_Solved'})
        print("✅ Registered B2 implementation (Neural SDE + MMD + PDE-Solved)")
        return True
    except Exception as e:
        print(f"❌ Failed to register B2 model: {e}")
        return False


# Auto-register when imported
if __name__ != "__main__":
    try:
        register_b2_model()
    except Exception as e:
        print(f"Warning: Could not auto-register B2 model: {e}")


if __name__ == "__main__":
    # Test B2 model creation
    print("Testing B2 Model Creation")
    print("=" * 30)
    print("Neural SDE + MMD + PDE-Solved Signatures")
    
    # Create test data
    batch_size = 8  # Smaller for testing
    example_batch = torch.randn(batch_size, 2, 50)  # Shorter sequences
    real_data = torch.randn(batch_size, 2, 50)
    
    try:
        # Create B2 model
        b2_model = create_b2_model(example_batch, real_data)
        
        print(f"✅ B2 model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in b2_model.parameters()):,}")
        
        # Test forward pass
        with torch.no_grad():
            output = b2_model(example_batch[:4])  # Even smaller batch
            print(f"   Forward pass: {example_batch[:4].shape} -> {output.shape}")
        
        # Test loss computation
        loss = b2_model.compute_loss(output)
        print(f"   MMD loss (PDE-solved): {loss.item():.6f}")
        
        # Test sample generation
        samples = b2_model.generate_samples(4)
        print(f"   Generated samples: {samples.shape}")
        
        print(f"\n🎉 B2 model test successful!")
        print(f"   Neural SDE generator working with MMD loss and PDE-solved signatures")
        print(f"   Combines champion architecture (Neural SDE) with champion loss (MMD) and advanced signatures")
        print(f"   Tests whether PDE-solved signatures improve B4's performance")
        
        # Compare parameter count with B1, B3, B4, B5
        print(f"\n📊 Comparison with other Neural SDE models:")
        print(f"   B2 parameters: {sum(p.numel() for p in b2_model.parameters()):,}")
        print(f"   Expected: Same as B1/B3/B4/B5 (9,027 parameters)")
        
    except Exception as e:
        print(f"❌ B2 model test failed: {e}")
        import traceback
        traceback.print_exc()

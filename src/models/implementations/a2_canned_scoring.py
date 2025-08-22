"""
A2 Model Implementation: CannedNet + Signature Scoring + Truncated

This implements experiment A2 using the validated factory pattern approach.
Following the A1FinalModel template to ensure correct implementation.

A2 tests the hypothesis that signature-aware architectures (CannedNet)
work particularly well with signature-based losses (scoring rule).
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

# Import original components (same as A1)
try:
    from models.deep_signature_transform import candle
    from models.deep_signature_transform import siglayer
    ORIGINAL_COMPONENTS_AVAILABLE = True
except ImportError:
    ORIGINAL_COMPONENTS_AVAILABLE = False
    warnings.warn("Original deep_signature_transform components not available")

# Import signature scoring components
try:
    from signatures import TruncatedSignature
    from signatures.signature_kernels import SignatureKernel
    SIGNATURE_COMPONENTS_AVAILABLE = True
except ImportError:
    SIGNATURE_COMPONENTS_AVAILABLE = False
    warnings.warn("Signature components not available")


class A2Model(BaseSignatureModel):
    """
    A2: CannedNet + Signature Scoring + Truncated
    
    This model combines:
    - Generator: CannedNet (same signature-aware architecture as A1)
    - Loss: Signature Kernel Scoring Rule (proper scoring rule, non-adversarial)
    - Signature Method: Truncated signatures (efficient computation)
    
    Key hypothesis: Signature-aware architectures should work particularly
    well with signature-based losses.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize A2 model."""
        if not ORIGINAL_COMPONENTS_AVAILABLE:
            raise ImportError("Original deep_signature_transform components required for A2")
        
        # Initialize base class components manually (following A1FinalModel pattern)
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
        """Get default configuration for A2."""
        return ModelConfig(
            model_id="A2",
            name="CannedNet + Signature Scoring + Truncated",
            description="Test scoring rule with signature-aware architecture",
            generator_type=GeneratorType.CANNED_NET,
            loss_type=LossType.SIGNATURE_SCORING,
            signature_method=SignatureMethod.TRUNCATED,
            status="implemented",
            priority="high",
            # Signature scoring loss configuration
            loss_config={
                'kernel_type': 'rbf',
                'sigma': 1.0,
                'adversarial': False,  # Non-adversarial scoring
                'max_batch': 128,
                'path_dim': 2
            },
            # Truncated signature configuration
            signature_config={
                'depth': 4
            }
        )
    
    def _build_model(self):
        """Build A2 model components."""
        # Use the exact same generator architecture as A1 (signature-aware CannedNet)
        self.generator = candle.CannedNet((
            siglayer.Augment((8, 8, 2), 1, include_original=True, include_time=False),
            candle.Window(2, 0, 1, transformation=siglayer.Signature(3)),
            siglayer.Augment((1,), 1, include_original=False, include_time=False),
            candle.batch_flatten
        ))
        
        # Create signature transform for loss computation
        if SIGNATURE_COMPONENTS_AVAILABLE:
            self.signature_transform = TruncatedSignature(
                depth=self.config.signature_config.get('depth', 4)
            )
        else:
            # Fallback: use the signature computation from original components
            self.signature_transform = siglayer.Signature(
                depth=self.config.signature_config.get('depth', 4)
            )
        
        # Loss function will be created when real data is provided
        self.loss_function = None
    
    def initialize_model(self, example_batch: torch.Tensor):
        """
        Initialize the model with example batch (required for CannedNet).
        
        Args:
            example_batch: Example input for model initialization
        """
        if not self.is_model_initialized:
            # For CannedNet initialization, we need to use CPU tensors first
            # then move the entire model to device after initialization
            example_batch_cpu = example_batch.cpu()
            
            # Initialize the generator (this sets up all parameters on CPU)
            _ = self.generator(example_batch_cpu)
            self.is_model_initialized = True
            
            # Now move entire model to target device after initialization
            self.to(self.device)
            
            print(f"A2 model initialized: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def set_real_data(self, real_paths: torch.Tensor):
        """
        Set real data for signature scoring loss computation.
        
        Args:
            real_paths: Real path data for scoring rule
        """
        if not self.is_model_initialized:
            raise RuntimeError("Model must be initialized with example batch first")
        
        # Ensure real paths are on the correct device
        real_paths = real_paths.to(self.device)
        
        # Create signature scoring loss
        self.loss_function = SignatureScoringLoss(
            signature_transform=self.signature_transform,
            real_paths=real_paths,
            device=self.device,  # Pass device to loss function
            **self.config.loss_config
        )
        self.is_loss_initialized = True
        
        print(f"A2 scoring loss initialized with real data shape: {real_paths.shape}")
    
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
        
        return self.generator(x)
    
    def compute_loss(self, generated_output: torch.Tensor, 
                    real_paths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute signature scoring rule loss.
        
        Args:
            generated_output: Generated paths from model
            real_paths: Not used (loss uses pre-set real data)
            
        Returns:
            Signature scoring rule loss value
        """
        if not self.is_loss_initialized:
            raise RuntimeError("Loss function must be initialized with real data first")
        
        return self.loss_function(generated_output)
    
    def generate_samples(self, batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples using the A2 model.
        
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
        
        # Create proper 3D input format for CannedNet (same as A1)
        noise_input = torch.randn(batch_size, 2, 100, device=device)
        
        with torch.no_grad():
            return self.forward(noise_input)


class SignatureScoringLoss:
    """
    Signature Scoring Rule Loss for A2.
    
    Implements the proper scoring rule:
    S(P, Y) = E_P[k(X, X)] - 2 * E_P[k(X, Y)]
    
    where P is the generated distribution and Y is real data.
    """
    
    def __init__(self, signature_transform, real_paths: torch.Tensor,
                 kernel_type: str = 'rbf', sigma: float = 1.0,
                 adversarial: bool = False, max_batch: int = 128, 
                 path_dim: int = 2, device: torch.device = None, **kwargs):
        """
        Initialize signature scoring loss.
        
        Args:
            signature_transform: Signature computation method
            real_paths: Real path data for scoring rule
            kernel_type: Type of kernel ('rbf' or 'linear')
            sigma: RBF kernel bandwidth
            adversarial: Whether to use learnable scaling (False for A2)
            max_batch: Maximum batch size for computation
            path_dim: Path dimension for scaling parameters
            device: Device to place tensors on
        """
        self.signature_transform = signature_transform
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.adversarial = adversarial
        self.max_batch = max_batch
        self.device = device or torch.device('cpu')
        
        # Ensure real paths are on correct device before signature computation
        real_paths = real_paths.to(self.device)
        
        # Precompute real path signatures
        self.real_signatures = self._compute_signatures(real_paths)
        
        # Adversarial scaling parameters (not used in A2 since adversarial=False)
        if adversarial:
            self._sigma = nn.Parameter(torch.ones(path_dim, device=self.device), requires_grad=True)
        else:
            self._sigma = None
        
        # Store device for gradient computation
        self.real_signatures = self.real_signatures.detach()  # Detach real signatures from computation graph
        
        print(f"Signature scoring loss created: {kernel_type} kernel, sigma={sigma}")
    
    def _compute_signatures(self, paths: torch.Tensor) -> torch.Tensor:
        """Compute signatures for paths."""
        try:
            # Ensure paths are on correct device
            paths = paths.to(self.device)
            
            # Ensure proper format for signature computation
            if len(paths.shape) == 3:
                # Already in (batch, channels, length) format
                signatures = self.signature_transform(paths)
            elif len(paths.shape) == 2:
                # Add channel dimension
                paths_3d = paths.unsqueeze(1)
                signatures = self.signature_transform(paths_3d)
            else:
                raise ValueError(f"Unexpected path shape: {paths.shape}")
            
            # Ensure signatures are on correct device and flatten for kernel computation
            signatures = signatures.to(self.device)
            if len(signatures.shape) > 2:
                signatures = signatures.view(signatures.size(0), -1)
            
            return signatures
        except Exception as e:
            warnings.warn(f"Signature computation failed: {e}. Using path directly.")
            flattened = paths.view(paths.size(0), -1).to(self.device)
            # Ensure gradient flow for fallback
            if paths.requires_grad:
                flattened = flattened.requires_grad_(True)
            return flattened
    
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel between signature features."""
        # Ensure tensors are on correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Ensure 2D tensors
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        if len(y.shape) > 2:
            y = y.view(y.size(0), -1)
        
        # Compute pairwise squared distances
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        
        # RBF kernel: exp(-||x-y||^2 / (2σ^2))
        return torch.exp(-dist_sq / (2 * self.sigma**2))
    
    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel between signature features."""
        # Ensure tensors are on correct device
        x = x.to(self.device)
        y = y.to(self.device)
        
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        if len(y.shape) > 2:
            y = y.view(y.size(0), -1)
        
        return torch.mm(x, y.transpose(0, 1))
    
    def __call__(self, generated_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature scoring rule loss.
        
        Args:
            generated_paths: Generated paths from model
            
        Returns:
            Signature scoring rule loss
        """
        # Ensure generated paths are on correct device
        generated_paths = generated_paths.to(self.device)
        
        # Convert generated paths to proper format and compute signatures
        generated_paths_formatted = self._format_generated_paths(generated_paths)
        generated_signatures = self._compute_signatures(generated_paths_formatted)
        
        # Apply adversarial scaling if enabled (not used in A2)
        if self._sigma is not None:
            generated_signatures = generated_signatures * self._sigma.unsqueeze(0)
        
        # Compute kernel matrices
        if self.kernel_type == 'rbf':
            # K(X, X) - self-similarity of generated
            K_XX = self._rbf_kernel(generated_signatures, generated_signatures)
            # K(X, Y) - cross-similarity between generated and real
            K_XY = self._rbf_kernel(generated_signatures, self.real_signatures)
        else:  # linear kernel
            K_XX = self._linear_kernel(generated_signatures, generated_signatures)
            K_XY = self._linear_kernel(generated_signatures, self.real_signatures)
        
        # Scoring rule: E[k(X,X)] - 2*E[k(X,Y)]
        # Remove diagonal for unbiased E[k(X,X)] estimate
        K_XX_offdiag = K_XX - torch.diag(torch.diag(K_XX))
        E_K_XX = K_XX_offdiag.sum() / (K_XX.size(0) * (K_XX.size(0) - 1))
        E_K_XY = K_XY.mean()
        
        scoring_loss = E_K_XX - 2 * E_K_XY
        
        return scoring_loss
    
    def _format_generated_paths(self, generated_paths: torch.Tensor) -> torch.Tensor:
        """Format generated paths for signature computation."""
        # Ensure paths are on correct device
        generated_paths = generated_paths.to(self.device)
        
        if len(generated_paths.shape) == 2:
            # Add time dimension to match original format
            batch_size, path_length = generated_paths.shape
            
            # Create time coordinates on the correct device
            time_coords = torch.linspace(0, 1, path_length + 1, device=self.device)
            time_coords = time_coords.unsqueeze(0).expand(batch_size, -1)
            
            # Add initial point (0) and create full path
            initial_points = torch.zeros(batch_size, 1, device=self.device)
            full_values = torch.cat([initial_points, generated_paths], dim=1)
            
            # Stack time and value coordinates
            paths_3d = torch.stack([time_coords, full_values], dim=1)
            
            return paths_3d
        elif len(generated_paths.shape) == 3:
            # Already in proper format, ensure on correct device
            return generated_paths.to(self.device)
        else:
            raise ValueError(f"Unexpected generated paths shape: {generated_paths.shape}")
    
    def parameters(self):
        """Return learnable parameters (generator + any adversarial discriminator params)."""
        params = list(super().parameters())
        if self._sigma is not None:
            params.append(self._sigma)
        return params


def create_a2_model(example_batch: torch.Tensor, 
                   real_data: torch.Tensor,
                   config_overrides: Optional[Dict[str, Any]] = None) -> A2Model:
    """
    Create properly initialized A2 model.
    
    Args:
        example_batch: Example batch for model initialization
        real_data: Real data for scoring rule computation
        config_overrides: Optional configuration overrides
        
    Returns:
        Fully initialized A2Model
    """
    # Create configuration
    config = A2Model.get_default_config()
    
    if config_overrides:
        config_dict = config.to_dict()
        config_dict.update(config_overrides)
        config = ModelConfig.from_dict(config_dict)
    
    # Create and initialize model
    model = A2Model(config)
    model.initialize_model(example_batch)
    model.set_real_data(real_data)
    
    return model


def test_a2_implementation():
    """Test A2 implementation."""
    print("Testing A2 Implementation")
    print("=" * 40)
    
    if not ORIGINAL_COMPONENTS_AVAILABLE:
        print("❌ Original components not available")
        return False
    
    try:
        # Setup test data (same as A1 validation)
        from dataset import generative_model
        import torch.utils.data as torchdata
        
        torch.manual_seed(42)
        
        n_points = 100
        batch_size = 32
        
        train_dataset = generative_model.get_noise(n_points=n_points, num_samples=batch_size)
        signals = generative_model.get_signal(num_samples=batch_size, n_points=n_points).tensors[0]
        
        train_dataloader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        example_batch, _ = next(iter(train_dataloader))
        
        print(f"Test data: signals {signals.shape}, example {example_batch.shape}")
        
        # Create A2 model
        print("\nCreating A2 model...")
        a2_model = create_a2_model(example_batch, signals)
        
        a2_params = sum(p.numel() for p in a2_model.parameters())
        print(f"A2 parameters: {a2_params:,}")
        
        # Test forward pass
        test_input = example_batch[:4]
        with torch.no_grad():
            a2_output = a2_model(test_input)
        
        print(f"A2 output shape: {a2_output.shape}")
        
        # Test loss computation
        a2_loss = a2_model.compute_loss(a2_output)
        print(f"A2 loss: {a2_loss.item():.6f}")
        
        # Test sample generation
        samples = a2_model.generate_samples(8)
        print(f"A2 samples shape: {samples.shape}")
        
        # Compare with A1 for reference
        print(f"\nComparison with A1:")
        
        # Create A1 for comparison
        from models.implementations.a1_final import create_a1_final_model
        torch.manual_seed(42)  # Same initialization
        a1_model = create_a1_final_model(example_batch, signals)
        
        a1_params = sum(p.numel() for p in a1_model.parameters())
        
        with torch.no_grad():
            a1_output = a1_model(test_input)
        
        print(f"  A1 parameters: {a1_params:,}, A2 parameters: {a2_params:,}")
        print(f"  A1 output: {a1_output.shape}, A2 output: {a2_output.shape}")
        
        # Check if generators produce same output (they should, same architecture)
        if a1_output.shape == a2_output.shape:
            generator_mse = torch.nn.functional.mse_loss(a1_output, a2_output)
            print(f"  Generator output MSE: {generator_mse.item():.8f}")
            
            if generator_mse.item() < 1e-6:
                print("  ✅ Generators produce identical outputs (expected)")
            else:
                print("  ⚠️ Generators produce different outputs")
        
        # Test different loss functions
        a1_loss = a1_model.compute_loss(a1_output)
        print(f"  A1 loss (T-statistic): {a1_loss.item():.6f}")
        print(f"  A2 loss (Scoring): {a2_loss.item():.6f}")
        print(f"  Loss difference: {abs(a1_loss.item() - a2_loss.item()):.6f}")
        
        print(f"\n✅ A2 implementation successful!")
        print(f"   Same generator architecture as A1 (✅)")
        print(f"   Different loss function (Signature Scoring vs T-statistic)")
        print(f"   Ready for systematic comparison")
        
        return True
        
    except Exception as e:
        print(f"❌ A2 implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def register_a2_model():
    """Register A2 model in the factory."""
    from models.model_registry import register_model
    
    try:
        config = A2Model.get_default_config()
        register_model("A2", A2Model, config, 
                      metadata={'validated': True, 'uses_original_components': True})
        print("✅ Registered A2 implementation in factory")
        return True
    except Exception as e:
        print(f"❌ Failed to register A2: {e}")
        return False


if __name__ == "__main__":
    print("A2 Model Implementation Test")
    print("=" * 50)
    
    if ORIGINAL_COMPONENTS_AVAILABLE:
        print("✅ Original components available")
        
        # Test A2 implementation
        success = test_a2_implementation()
        
        if success:
            # Register in factory
            registered = register_a2_model()
            
            if registered:
                print("\n🎉 A2 IMPLEMENTATION COMPLETE!")
                print("   ✅ Uses same CannedNet architecture as validated A1")
                print("   ✅ Implements signature scoring rule loss")
                print("   ✅ Ready for systematic comparison with A1")
                print("   ✅ Registered in factory as 'A2'")
                
                print(f"\nUsage:")
                print(f"  from models.implementations.a2_canned_scoring import create_a2_model")
                print(f"  model = create_a2_model(example_batch, real_data)")
            else:
                print("\n⚠️ A2 works but registration failed")
        else:
            print("\n❌ A2 implementation needs debugging")
    else:
        print("❌ Original components not available")
        print("   Install deep_signature_transform dependencies")

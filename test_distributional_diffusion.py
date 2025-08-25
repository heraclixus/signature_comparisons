"""
Test script for Distributional Diffusion implementation.

This script tests the basic functionality of our distributional diffusion
implementation to ensure everything works correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_signature_scoring_loss():
    """Test the signature scoring loss implementation."""
    print("Testing SignatureScoringLoss...")
    
    try:
        from losses.signature_scoring_loss import SignatureScoringLoss, create_signature_scoring_loss
        
        # Create test data
        batch_size, population_size, dim, seq_len = 4, 6, 1, 32
        
        generated_samples = torch.randn(batch_size, population_size, dim, seq_len)
        real_sample = torch.randn(batch_size, dim, seq_len)
        
        # Test direct implementation
        try:
            loss_fn = create_signature_scoring_loss(method="direct", lambda_param=1.0)
            loss = loss_fn(generated_samples, real_sample)
            print(f"✓ Direct method loss: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ Direct method failed: {e}")
        
        # Test adapted implementation
        try:
            loss_fn = create_signature_scoring_loss(method="adapted", path_dim=dim, lambda_param=1.0)
            loss = loss_fn(generated_samples, real_sample)
            print(f"✓ Adapted method loss: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ Adapted method failed: {e}")
            
    except ImportError as e:
        print(f"✗ Could not import signature scoring loss: {e}")
    
    print()


def test_distributional_generator():
    """Test the distributional generator implementation."""
    print("Testing DistributionalGenerator...")
    
    try:
        from models.distributional_generator import create_distributional_generator
        
        # Test feedforward generator
        try:
            generator = create_distributional_generator(
                generator_type="feedforward",
                data_size=1,
                seq_len=32,
                hidden_size=64,
                num_layers=2
            )
            
            # Test forward pass
            batch_size = 4
            x_t = torch.randn(batch_size, 1, 32)
            t = torch.rand(batch_size)
            z = torch.randn(batch_size, 1, 32)
            
            output = generator(x_t, t, z)
            print(f"✓ Feedforward generator output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Feedforward generator failed: {e}")
        
        # Test SDE-based generator (if available)
        try:
            generator = create_distributional_generator(
                generator_type="sde_based",
                data_size=1,
                seq_len=32
            )
            print("✓ SDE-based generator created successfully")
        except Exception as e:
            print(f"✗ SDE-based generator failed: {e}")
            
    except ImportError as e:
        print(f"✗ Could not import distributional generator: {e}")
    
    print()


def test_distributional_diffusion():
    """Test the distributional diffusion model."""
    print("Testing DistributionalDiffusion...")
    
    try:
        from models.tsdiff.diffusion.distributional_diffusion import DistributionalDiffusion
        from models.distributional_generator import create_distributional_generator
        
        # Create model
        ddm = DistributionalDiffusion(
            dim=1,
            seq_len=32,
            gamma=1.0,
            population_size=4,
            lambda_param=1.0
        )
        
        # Create generator
        generator = create_distributional_generator(
            generator_type="feedforward",
            data_size=1,
            seq_len=32,
            hidden_size=32,
            num_layers=2
        )
        
        print(f"✓ Model created with {sum(p.numel() for p in ddm.parameters())} parameters")
        print(f"✓ Generator created with {sum(p.numel() for p in generator.parameters())} parameters")
        
        # Test forward diffusion
        batch_size = 4
        x0 = torch.randn(batch_size, 1, 32)
        t = torch.rand(batch_size)
        
        x_t = ddm.forward_diffusion(x0, t)
        print(f"✓ Forward diffusion output shape: {x_t.shape}")
        
        # Test loss computation
        loss = ddm.get_loss(generator, x0)
        print(f"✓ Loss computation: {loss.item():.6f}")
        
        # Test sampling
        samples = ddm.sample(generator, num_samples=8, num_coarse_steps=10)
        print(f"✓ Sampling output shape: {samples.shape}")
        
    except ImportError as e:
        print(f"✗ Could not import distributional diffusion: {e}")
    except Exception as e:
        print(f"✗ Distributional diffusion test failed: {e}")
    
    print()


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    try:
        from models.tsdiff.diffusion.distributional_diffusion import DistributionalDiffusion
        from models.distributional_generator import create_distributional_generator
        
        # Create model and generator
        ddm = DistributionalDiffusion(
            dim=1,
            seq_len=16,  # Smaller for faster testing
            population_size=4,
            lambda_param=1.0
        )
        
        generator = create_distributional_generator(
            generator_type="feedforward",
            data_size=1,
            seq_len=16,
            hidden_size=32,
            num_layers=2
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        
        # Create synthetic training data
        batch_size = 8
        x0 = torch.randn(batch_size, 1, 16).cumsum(dim=-1)  # Simple Brownian motion
        
        # Training step
        optimizer.zero_grad()
        loss = ddm.get_loss(generator, x0)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed, loss: {loss.item():.6f}")
        
        # Test multiple steps
        losses = []
        for step in range(5):
            optimizer.zero_grad()
            loss = ddm.get_loss(generator, x0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print(f"✓ Multiple training steps: {[f'{l:.4f}' for l in losses]}")
        
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
    
    print()


def test_configuration_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    try:
        import yaml
        
        config_path = "config/distributional_diffusion.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"✓ Configuration loaded successfully")
            print(f"  - Model dim: {config['model']['dim']}")
            print(f"  - Sequence length: {config['model']['seq_len']}")
            print(f"  - Population size: {config['model']['population_size']}")
            print(f"  - Lambda parameter: {config['model']['lambda_param']}")
        else:
            print(f"✗ Configuration file not found: {config_path}")
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
    
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Distributional Diffusion Implementation")
    print("=" * 60)
    print()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Run tests
    test_signature_scoring_loss()
    test_distributional_generator()
    test_distributional_diffusion()
    test_training_step()
    test_configuration_loading()
    
    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

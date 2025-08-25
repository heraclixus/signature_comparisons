"""
Test D2 Model Integration with Existing Pipeline

This script tests that our D2 distributional diffusion model
integrates properly with the existing signature comparison pipeline.
"""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_d2_model_creation():
    """Test D2 model creation through the factory."""
    print("Testing D2 model creation...")
    
    try:
        from models.model_factory import create_model, get_available_models, print_available_models
        
        # Check if D2 is registered
        available_models = get_available_models()
        print(f"Available models: {available_models}")
        
        if "D2" in available_models:
            print("✓ D2 is registered in the model factory")
            
            # Create D2 model
            d2_model = create_model("D2")
            print(f"✓ D2 model created: {type(d2_model)}")
            
            # Check model info
            model_info = d2_model.get_model_info()
            print(f"✓ Model info: {model_info}")
            
            return d2_model
        else:
            print("✗ D2 not found in available models")
            return None
            
    except Exception as e:
        print(f"✗ D2 model creation failed: {e}")
        return None


def test_d2_training_interface():
    """Test D2 training interface compatibility."""
    print("\nTesting D2 training interface...")
    
    try:
        from models.d2_distributional_diffusion import create_d2_model
        
        # Create model with small configuration for testing
        d2_model = create_d2_model(
            dim=1,
            seq_len=16,
            population_size=4,
            hidden_size=32,
            num_layers=2,
            device='cpu'
        )
        
        print(f"✓ D2 model created directly")
        
        # Test forward pass
        batch_size = 8
        noise_input = torch.randn(batch_size, 1, 16)
        
        output = d2_model.forward(noise_input)
        print(f"✓ Forward pass: input {noise_input.shape} -> output {output.shape}")
        
        # Test loss computation
        real_data = torch.randn(batch_size, 1, 16).cumsum(dim=-1)  # Brownian-like
        loss = d2_model.compute_loss(output, real_data)
        print(f"✓ Loss computation: {loss.item():.6f}")
        
        # Test sample generation
        samples = d2_model.generate_samples(num_samples=10)
        print(f"✓ Sample generation: {samples.shape}")
        
        return d2_model
        
    except Exception as e:
        print(f"✗ D2 training interface test failed: {e}")
        return None


def test_d2_training_loop():
    """Test D2 training loop."""
    print("\nTesting D2 training loop...")
    
    try:
        from models.d2_distributional_diffusion import create_d2_model
        
        # Create model
        d2_model = create_d2_model(
            dim=1,
            seq_len=16,
            population_size=4,
            hidden_size=32,
            num_layers=2,
            num_epochs=3,  # Very short training
            batch_size=16,
            learning_rate=1e-3,
            device='cpu'
        )
        
        # Generate synthetic training data
        num_samples = 64
        train_data = torch.randn(num_samples, 1, 16).cumsum(dim=-1)
        
        print(f"Training data shape: {train_data.shape}")
        
        # Train model
        history = d2_model.fit(train_data, num_epochs=3)
        
        print(f"✓ Training completed")
        print(f"  - Final loss: {history['train_loss'][-1]:.6f}")
        print(f"  - Loss history: {[f'{l:.4f}' for l in history['train_loss']]}")
        
        # Test generation after training
        samples = d2_model.generate_samples(num_samples=5)
        print(f"✓ Post-training generation: {samples.shape}")
        
        return d2_model
        
    except Exception as e:
        print(f"✗ D2 training loop test failed: {e}")
        return None


def test_d2_compatibility():
    """Test D2 compatibility with existing evaluation pipeline."""
    print("\nTesting D2 pipeline compatibility...")
    
    try:
        # Test if D2 can be used in existing training scripts
        from models.model_factory import create_model
        
        # Create D2 through factory
        d2_model = create_model("D2", config_overrides={
            'data_config': {'dim': 1, 'seq_len': 16},
            'training_config': {'device': 'cpu', 'batch_size': 16},
            'loss_config': {'population_size': 4}
        })
        
        print("✓ D2 created through factory with config overrides")
        
        # Test BaseSignatureModel interface methods
        batch_size = 8
        test_input = torch.randn(batch_size, 1, 16)
        test_target = torch.randn(batch_size, 1, 16).cumsum(dim=-1)
        
        # Test forward
        output = d2_model.forward(test_input)
        print(f"✓ BaseSignatureModel.forward: {output.shape}")
        
        # Test compute_loss
        loss = d2_model.compute_loss(output, test_target)
        print(f"✓ BaseSignatureModel.compute_loss: {loss.item():.6f}")
        
        # Test generate_samples
        samples = d2_model.generate_samples(10)
        print(f"✓ BaseSignatureModel.generate_samples: {samples.shape}")
        
        # Test model info
        info = d2_model.get_model_info()
        print(f"✓ Model info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ D2 compatibility test failed: {e}")
        return False


def test_cuda_support():
    """Test CUDA support if available."""
    print("\nTesting CUDA support...")
    
    if torch.cuda.is_available():
        try:
            from models.d2_distributional_diffusion import create_d2_model
            
            # Create model on CUDA
            d2_model = create_d2_model(
                dim=1,
                seq_len=16,
                population_size=4,
                device='cuda'
            )
            
            print("✓ D2 model created on CUDA")
            
            # Test forward pass on CUDA
            test_input = torch.randn(4, 1, 16, device='cuda')
            output = d2_model.forward(test_input)
            
            print(f"✓ CUDA forward pass: {output.device}")
            
            # Test loss computation on CUDA
            test_target = torch.randn(4, 1, 16, device='cuda').cumsum(dim=-1)
            loss = d2_model.compute_loss(output, test_target)
            
            print(f"✓ CUDA loss computation: {loss.item():.6f}")
            
            return True
            
        except Exception as e:
            print(f"✗ CUDA test failed: {e}")
            return False
    else:
        print("CUDA not available, skipping CUDA tests")
        return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Testing D2 Model Integration")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    results = []
    
    # Test 1: Model creation through factory
    d2_model = test_d2_model_creation()
    results.append(d2_model is not None)
    
    # Test 2: Training interface
    d2_trained = test_d2_training_interface()
    results.append(d2_trained is not None)
    
    # Test 3: Training loop
    d2_loop = test_d2_training_loop()
    results.append(d2_loop is not None)
    
    # Test 4: Pipeline compatibility
    compatibility = test_d2_compatibility()
    results.append(compatibility)
    
    # Test 5: CUDA support
    cuda_support = test_cuda_support()
    results.append(cuda_support)
    
    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)
    
    test_names = [
        "Model Factory Creation",
        "Training Interface", 
        "Training Loop",
        "Pipeline Compatibility",
        "CUDA Support"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_passed = sum(results)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n🎉 All integration tests passed! D2 is ready for the pipeline.")
    else:
        print(f"\n⚠️  {len(results) - total_passed} tests failed. Check implementation.")


if __name__ == "__main__":
    main()

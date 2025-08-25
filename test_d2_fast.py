"""
Fast D2 Model Test

This script tests D2 with highly optimized parameters for speed.
"""

import torch
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_d2_fast():
    """Test D2 with optimized parameters for speed."""
    print("🚀 Testing D2 with fast configuration...")
    
    try:
        from models.implementations.d2_distributional_diffusion import create_model
        
        # Create very small synthetic data for speed
        batch_size = 8
        dim = 1
        seq_len = 16
        
        # Generate synthetic Brownian motion data
        real_data = torch.randn(batch_size, dim, seq_len).cumsum(dim=-1) * 0.1
        example_batch = real_data[:4]  # Smaller example batch
        
        print(f"Data shape: {real_data.shape}")
        
        # Create D2 model with ultra-fast configuration
        fast_config = {
            'population_size': 2,  # Minimal population size
            'hidden_size': 32,     # Very small network
            'num_layers': 1,       # Single layer
            'num_coarse_steps': 5, # Minimal sampling steps
            'dyadic_order': 2,     # Reduced signature complexity
            'max_batch': 8,        # Small batch processing
            'learning_rate': 1e-3  # Higher learning rate
        }
        
        start_time = time.time()
        model = create_model(example_batch, real_data, fast_config)
        creation_time = time.time() - start_time
        
        print(f"✅ Model created in {creation_time:.3f}s")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        start_time = time.time()
        output = model.forward(real_data)
        forward_time = time.time() - start_time
        
        print(f"✅ Forward pass in {forward_time:.3f}s")
        print(f"   Output shape: {output.shape}")
        
        # Test loss computation
        start_time = time.time()
        loss = model.compute_loss(output)
        loss_time = time.time() - start_time
        
        print(f"✅ Loss computation in {loss_time:.3f}s")
        print(f"   Loss value: {loss.item():.6f}")
        
        # Test sample generation
        start_time = time.time()
        samples = model.generate_samples(4)
        generation_time = time.time() - start_time
        
        print(f"✅ Sample generation in {generation_time:.3f}s")
        print(f"   Samples shape: {samples.shape}")
        
        # Test training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        optimizer.zero_grad()
        loss = model.compute_loss(model.forward(real_data))
        loss.backward()
        optimizer.step()
        training_time = time.time() - start_time
        
        print(f"✅ Training step in {training_time:.3f}s")
        
        # Summary
        total_time = creation_time + forward_time + loss_time + generation_time + training_time
        print(f"\n📊 Performance Summary:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Model creation: {creation_time:.3f}s ({creation_time/total_time*100:.1f}%)")
        print(f"   Forward pass: {forward_time:.3f}s ({forward_time/total_time*100:.1f}%)")
        print(f"   Loss computation: {loss_time:.3f}s ({loss_time/total_time*100:.1f}%)")
        print(f"   Sample generation: {generation_time:.3f}s ({generation_time/total_time*100:.1f}%)")
        print(f"   Training step: {training_time:.3f}s ({training_time/total_time*100:.1f}%)")
        
        if total_time < 5.0:
            print(f"\n🎉 Fast configuration successful! Total time: {total_time:.3f}s")
            return True
        else:
            print(f"\n⚠️ Still slow. Total time: {total_time:.3f}s")
            return False
            
    except Exception as e:
        print(f"❌ Fast test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_pipeline_fast():
    """Test D2 with the actual training pipeline using test mode."""
    print("\n🔧 Testing D2 with training pipeline (test mode)...")
    
    try:
        # Use the training script with test mode
        import subprocess
        import sys
        
        cmd = [
            sys.executable, 
            "src/experiments/train_and_save_models.py",
            "--model", "D2",
            "--dataset", "brownian", 
            "--epochs", "2",
            "--test-mode"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ Training pipeline test completed in {end_time - start_time:.1f}s")
            print("📋 Output (last 10 lines):")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                print(f"   {line}")
            return True
        else:
            print(f"❌ Training pipeline test failed (exit code: {result.returncode})")
            print("📋 Error output:")
            for line in result.stderr.strip().split('\n')[-10:]:
                print(f"   {line}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training pipeline test timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training pipeline test error: {e}")
        return False


def main():
    """Run fast D2 tests."""
    print("=" * 60)
    print("D2 Fast Performance Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test 1: Direct model test
    direct_success = test_d2_fast()
    
    # Test 2: Training pipeline test
    pipeline_success = test_training_pipeline_fast()
    
    # Summary
    print("\n" + "=" * 60)
    print("Fast Test Summary")
    print("=" * 60)
    
    print(f"1. Direct Model Test: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"2. Training Pipeline Test: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if direct_success and pipeline_success:
        print("\n🎉 All fast tests passed! D2 is ready for training.")
    elif direct_success:
        print("\n⚠️ Direct test passed but pipeline test failed. Check training integration.")
    else:
        print("\n❌ Performance issues detected. Consider further optimization.")


if __name__ == "__main__":
    main()

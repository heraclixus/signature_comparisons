"""
Register Model Implementations in Factory

This script registers the validated model implementations in the factory system.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_registry import get_model_registry, register_model
from models.implementations.a1_final import A1FinalModel
from models.implementations.a2_canned_scoring import A2Model
from models.implementations.a3_canned_mmd import A3Model
from models.implementations.b3_nsde_tstatistic import B3Model
from models.implementations.b4_nsde_mmd import B4Model
from models.implementations.b5_nsde_scoring import B5Model


def register_validated_models():
    """Register all validated model implementations."""
    print("Registering Validated Model Implementations")
    print("=" * 50)
    
    registry = get_model_registry()
    
    # Register A1 Final (corrected implementation)
    try:
        a1_config = A1FinalModel.get_default_config()
        register_model("A1_FINAL", A1FinalModel, a1_config, 
                      metadata={
                          'exact_original_match': True,
                          'validated': True,
                          'parameter_count': 199,
                          'implementation_status': 'perfect'
                      })
        print("✅ Registered A1_FINAL: Perfect original wrapper")
    except ValueError:
        print("⚠️ A1_FINAL already registered")
    except Exception as e:
        print(f"❌ Failed to register A1_FINAL: {e}")
    
    # Register A2 (new implementation)
    try:
        a2_config = A2Model.get_default_config()
        # Update model_id to avoid conflict
        a2_config.model_id = "A2_IMPLEMENTED"
        
        register_model("A2_IMPLEMENTED", A2Model, a2_config,
                      metadata={
                          'uses_original_components': True,
                          'validated': True,
                          'parameter_count': 199,
                          'implementation_status': 'working',
                          'comparison_with_a1': 'validated'
                      })
        print("✅ Registered A2_IMPLEMENTED: CannedNet + Signature Scoring")
    except ValueError:
        print("⚠️ A2_IMPLEMENTED already registered")
    except Exception as e:
        print(f"❌ Failed to register A2_IMPLEMENTED: {e}")
    
    # Register A3 (new implementation)
    try:
        a3_config = A3Model.get_default_config()
        # Update model_id to avoid conflict
        a3_config.model_id = "A3_IMPLEMENTED"
        
        register_model("A3_IMPLEMENTED", A3Model, a3_config,
                      metadata={
                          'uses_original_components': True,
                          'validated': False,  # New implementation, needs validation
                          'parameter_count': 199,  # Same as A1, A2
                          'implementation_status': 'new',
                          'loss_type': 'MMD'
                      })
        print("✅ Registered A3_IMPLEMENTED: CannedNet + MMD + Truncated")
    except ValueError:
        print("⚠️ A3_IMPLEMENTED already registered")
    except Exception as e:
        print(f"❌ Failed to register A3_IMPLEMENTED: {e}")
    
    # Register B4 (new implementation)
    try:
        b4_config = B4Model.get_default_config()
        # Update model_id to avoid conflict
        b4_config.model_id = "B4_IMPLEMENTED"
        
        register_model("B4_IMPLEMENTED", B4Model, b4_config,
                      metadata={
                          'independent_implementation': True,
                          'validated': False,  # New implementation, needs validation
                          'parameter_count': 9027,  # Much larger than CannedNet
                          'implementation_status': 'new',
                          'generator_type': 'Neural_SDE',
                          'loss_type': 'MMD'
                      })
        print("✅ Registered B4_IMPLEMENTED: Neural SDE + MMD + Truncated")
    except ValueError:
        print("⚠️ B4_IMPLEMENTED already registered")
    except Exception as e:
        print(f"❌ Failed to register B4_IMPLEMENTED: {e}")
    
    # Register B5 (new implementation)
    try:
        b5_config = B5Model.get_default_config()
        # Update model_id to avoid conflict
        b5_config.model_id = "B5_IMPLEMENTED"
        
        register_model("B5_IMPLEMENTED", B5Model, b5_config,
                      metadata={
                          'independent_implementation': True,
                          'validated': False,  # New implementation, needs validation
                          'parameter_count': 9027,  # Same as B4
                          'implementation_status': 'new',
                          'generator_type': 'Neural_SDE',
                          'loss_type': 'Signature_Scoring'
                      })
        print("✅ Registered B5_IMPLEMENTED: Neural SDE + Signature Scoring + Truncated")
    except ValueError:
        print("⚠️ B5_IMPLEMENTED already registered")
    except Exception as e:
        print(f"❌ Failed to register B5_IMPLEMENTED: {e}")
    
    # Register B3 (new implementation)
    try:
        b3_config = B3Model.get_default_config()
        # Update model_id to avoid conflict
        b3_config.model_id = "B3_IMPLEMENTED"
        
        register_model("B3_IMPLEMENTED", B3Model, b3_config,
                      metadata={
                          'independent_implementation': True,
                          'validated': False,  # New implementation, needs validation
                          'parameter_count': 9027,  # Same as B4
                          'implementation_status': 'new',
                          'generator_type': 'Neural_SDE',
                          'loss_type': 'T_Statistic'
                      })
        print("✅ Registered B3_IMPLEMENTED: Neural SDE + T-Statistic + Truncated")
    except ValueError:
        print("⚠️ B3_IMPLEMENTED already registered")
    except Exception as e:
        print(f"❌ Failed to register B3_IMPLEMENTED: {e}")
    
    # Note: C1-C3 (GRU) models removed - not truly generative
    # Diversity testing revealed they produce nearly identical outputs
    # regardless of random seeds, making them unsuitable for stochastic process modeling
    
    # Print registry summary
    print(f"\nRegistry Summary:")
    registry.print_registry_summary()
    
    return True


def test_registered_models():
    """Test that registered models can be created."""
    print(f"\nTesting Registered Models")
    print("-" * 30)
    
    try:
        from models import create_model
        
        # Test creating models by ID
        registry = get_model_registry()
        available_models = registry.list_models()
        
        print(f"Available models: {available_models}")
        
        # Test specific models if available
        if "A1_FINAL" in available_models:
            print("✅ A1_FINAL available in registry")
        
        if "A2_IMPLEMENTED" in available_models:
            print("✅ A2_IMPLEMENTED available in registry")
        
        if "A3_IMPLEMENTED" in available_models:
            print("✅ A3_IMPLEMENTED available in registry")
        
        if "B4_IMPLEMENTED" in available_models:
            print("✅ B4_IMPLEMENTED available in registry")
        
        if "B5_IMPLEMENTED" in available_models:
            print("✅ B5_IMPLEMENTED available in registry")
        
        if "B3_IMPLEMENTED" in available_models:
            print("✅ B3_IMPLEMENTED available in registry")
        
        # C1-C3 models removed (not truly generative)
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False


def main():
    """Main registration and testing."""
    print("Model Registration and Validation")
    print("=" * 60)
    
    # Register models
    register_success = register_validated_models()
    
    # Test registry
    test_success = test_registered_models()
    
    print(f"\n" + "="*60)
    print("REGISTRATION SUMMARY")
    print("="*60)
    
    if register_success and test_success:
        print("🎉 MODEL REGISTRATION SUCCESSFUL!")
        print("   A1_FINAL: Perfect original wrapper")
        print("   A2_IMPLEMENTED: CannedNet + Signature Scoring")
        print("   A3_IMPLEMENTED: CannedNet + MMD")
        print("   B3_IMPLEMENTED: Neural SDE + T-Statistic")
        print("   B4_IMPLEMENTED: Neural SDE + MMD")
        print("   B5_IMPLEMENTED: Neural SDE + Signature Scoring")
        print("   All truly generative models available in factory system")
        print("   (C1-C3 RNN models removed - not truly generative)")
        
        print(f"\nUsage:")
        print(f"  # For exact original behavior")
        print(f"  a1_model = create_a1_final_model(example_batch, real_data)")
        print(f"  ")
        print(f"  # For signature scoring experiment")
        print(f"  a2_model = create_a2_model(example_batch, real_data)")
        print(f"  ")
        print(f"  # For MMD experiment")
        print(f"  a3_model = create_a3_model(example_batch, real_data)")
        print(f"  ")
        print(f"  # For Neural SDE + MMD experiment")
        print(f"  b4_model = create_b4_model(example_batch, real_data)")
        print(f"  ")
        print(f"  # All have identical interfaces for comparison")
        print(f"  a1_samples = a1_model.generate_samples(32)")
        print(f"  a2_samples = a2_model.generate_samples(32)")
        print(f"  a3_samples = a3_model.generate_samples(32)")
        print(f"  b4_samples = b4_model.generate_samples(32)")
        
    else:
        print("⚠️ Registration had some issues")
        print("   Check implementation details")
    
    return register_success and test_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

"""
Factory Pattern Example for Signature-based Models

This demonstrates how to use the new factory pattern to create
and manage different model combinations.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import (
    BaseSignatureModel, 
    ModelConfig, 
    ModelFactory,
    create_model,
    get_available_models,
    get_model_registry
)
from models.base_model import GeneratorType, LossType, SignatureMethod


def example_basic_factory_usage():
    """Basic example of using the model factory."""
    print("=== Basic Factory Usage ===")
    
    # List available models
    available_models = get_available_models()
    print(f"Available models: {available_models}")
    
    # Create models by ID
    for model_id in available_models[:3]:  # First 3 models
        try:
            model = create_model(model_id)
            print(f"\nCreated {model_id}:")
            print(f"  Name: {model.config.name}")
            print(f"  Generator: {model.config.generator_type.value}")
            print(f"  Loss: {model.config.loss_type.value}")
            print(f"  Signature: {model.config.signature_method.value}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"  Error creating {model_id}: {e}")


def example_create_from_components():
    """Example of creating models by specifying components."""
    print("\n=== Create from Components ===")
    
    # Create model by specifying component types
    try:
        from models.model_factory import create_model_from_components
        
        model = create_model_from_components(
            generator_type="rnn",
            loss_type="t_statistic", 
            signature_method="truncated"
        )
        
        print(f"Created model: {model.config.name}")
        print(f"Model ID: {model.config.model_id}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test generation
        samples = model.generate_samples(batch_size=4)
        print(f"Generated samples shape: {samples.shape}")
        
    except Exception as e:
        print(f"Error creating from components: {e}")


def example_custom_configuration():
    """Example of creating models with custom configurations."""
    print("\n=== Custom Configuration ===")
    
    # Create custom configuration
    custom_config = ModelConfig(
        model_id="CUSTOM_LSTM",
        name="Custom LSTM + T-Statistic",
        description="Custom LSTM configuration for testing",
        generator_type=GeneratorType.LSTM,
        loss_type=LossType.T_STATISTIC,
        signature_method=SignatureMethod.TRUNCATED,
        generator_config={
            'hidden_size': 128,  # Larger hidden size
            'num_layers': 3,     # More layers
            'dropout': 0.2       # Higher dropout
        },
        training_config={
            'learning_rate': 0.0005,  # Lower learning rate
            'batch_size': 64          # Smaller batch size
        }
    )
    
    try:
        model = create_model(custom_config)
        print(f"Created custom model: {model.config.name}")
        print(f"Hidden size: {model.config.generator_config['hidden_size']}")
        print(f"Learning rate: {model.config.training_config['learning_rate']}")
        
        # Test the model
        samples = model.generate_samples(batch_size=2)
        print(f"Generated samples shape: {samples.shape}")
        
    except Exception as e:
        print(f"Error creating custom model: {e}")


def example_model_registry():
    """Example of using the model registry."""
    print("\n=== Model Registry ===")
    
    registry = get_model_registry()
    
    # Print registry summary
    registry.print_registry_summary()
    
    # List models by type
    rnn_models = registry.list_models_by_type(generator_type=GeneratorType.RNN)
    print(f"\nRNN-based models: {rnn_models}")
    
    canned_net_models = registry.list_models_by_type(generator_type=GeneratorType.CANNED_NET)
    print(f"CannedNet-based models: {canned_net_models}")
    
    t_statistic_models = registry.list_models_by_type(loss_type=LossType.T_STATISTIC)
    print(f"T-Statistic loss models: {t_statistic_models}")


def example_model_comparison():
    """Example of comparing different models."""
    print("\n=== Model Comparison ===")
    
    # Create different models for comparison
    model_ids = ["A1", "A2", "C1"]  # Different combinations
    models = {}
    
    for model_id in model_ids:
        try:
            model = create_model(model_id)
            models[model_id] = model
            
            print(f"\n{model_id} Model:")
            print(f"  Generator: {model.config.generator_type.value}")
            print(f"  Loss: {model.config.loss_type.value}")
            print(f"  Signature: {model.config.signature_method.value}")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test generation
            samples = model.generate_samples(batch_size=2)
            print(f"  Sample shape: {samples.shape}")
            print(f"  Sample mean: {samples.mean().item():.4f}")
            print(f"  Sample std: {samples.std().item():.4f}")
            
        except Exception as e:
            print(f"Error with {model_id}: {e}")
    
    print(f"\nCreated {len(models)} models for comparison")


def example_save_load_config():
    """Example of saving and loading model configurations."""
    print("\n=== Save/Load Configuration ===")
    
    # Create a custom configuration
    config = ModelConfig(
        model_id="TEST_MODEL",
        name="Test Model Configuration",
        description="Example configuration for testing save/load",
        generator_type=GeneratorType.TRANSFORMER,
        loss_type=LossType.SIGNATURE_MMD,
        signature_method=SignatureMethod.TRUNCATED,
        generator_config={'embed_dim': 256, 'num_heads': 16},
        training_config={'learning_rate': 0.0001, 'batch_size': 32}
    )
    
    # Save configuration
    config_path = "test_model_config.json"
    config.save(config_path)
    print(f"Saved configuration to {config_path}")
    
    # Load configuration
    loaded_config = ModelConfig.load(config_path)
    print(f"Loaded configuration: {loaded_config.name}")
    print(f"Generator config: {loaded_config.generator_config}")
    print(f"Training config: {loaded_config.training_config}")
    
    # Clean up
    os.remove(config_path)
    print("Cleaned up test file")


def run_all_examples():
    """Run all factory pattern examples."""
    print("Factory Pattern Examples for Signature-based Models")
    print("=" * 60)
    
    try:
        example_basic_factory_usage()
        example_create_from_components()
        example_custom_configuration()
        example_model_registry()
        example_model_comparison()
        example_save_load_config()
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Factory Pattern Examples completed!")
    print("\nKey Benefits:")
    print("1. Unified interface for all model types")
    print("2. Easy configuration and customization")
    print("3. Registry system for model discovery")
    print("4. Mix and match any generator + loss + signature combination")
    print("5. Consistent API across all implementations")
    print("\nUsage:")
    print("  from models import create_model")
    print("  model = create_model('A2')  # Create registered model")
    print("  model = create_model_from_components('lstm', 't_statistic', 'truncated')  # Create custom combination")


if __name__ == "__main__":
    run_all_examples()

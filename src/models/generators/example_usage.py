"""
Example usage of the unified generators module.

This demonstrates how to use the different generator types
and the factory function for creating generators.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.generators import (
    GeneratorType,
    create_generator,
    get_default_config,
    CannedNet,
    NeuralSDEGenerator,
    RNNGenerator,
    TransformerGenerator
)


def example_canned_net():
    """Example of creating and using a CannedNet generator."""
    print("=== CannedNet Generator Example ===")
    
    # Get default configuration
    config = get_default_config(GeneratorType.CANNED_NET)
    print(f"Default config: {config}")
    
    try:
        # Create generator using factory
        generator = create_generator(GeneratorType.CANNED_NET, config)
        print(f"Created generator: {type(generator).__name__}")
        
        # Example input (would normally come from noise or data)
        batch_size = 4
        input_dim = 2
        sequence_length = 100
        
        # Create dummy input (this would be noise or initial conditions in practice)
        dummy_input = torch.randn(batch_size, input_dim, sequence_length)
        
        # Generate output
        with torch.no_grad():
            output = generator(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error with CannedNet: {e}")
        print("This is expected if deep_signature_transform dependencies are not available")


def example_neural_sde():
    """Example of creating and using a Neural SDE generator."""
    print("\n=== Neural SDE Generator Example ===")
    
    # Get default configuration
    config = get_default_config(GeneratorType.NEURAL_SDE)
    print(f"Default config: {config}")
    
    try:
        # Create generator using factory
        generator = create_generator(GeneratorType.NEURAL_SDE, config)
        print(f"Created generator: {type(generator).__name__}")
        
        # Example usage
        batch_size = 4
        time_steps = torch.linspace(0, 1, 100)
        
        # Generate paths
        with torch.no_grad():
            output = generator(time_steps, batch_size)
            print(f"Time steps shape: {time_steps.shape}")
            print(f"Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Error with Neural SDE: {e}")
        print("This is expected if torchsde dependencies are not available")


def example_rnn_generator():
    """Example of creating and using an RNN generator."""
    print("\n=== RNN Generator Example ===")
    
    # Get default configuration
    config = get_default_config(GeneratorType.LSTM)
    print(f"Default config: {config}")
    
    # Create generator using factory
    generator = create_generator(GeneratorType.LSTM, config)
    print(f"Created generator: {type(generator).__name__}")
    
    # Generate sequences
    batch_size = 4
    with torch.no_grad():
        output = generator.generate(batch_size)
        print(f"Generated sequences shape: {output.shape}")


def example_transformer_generator():
    """Example of creating and using a Transformer generator."""
    print("\n=== Transformer Generator Example ===")
    
    # Get default configuration
    config = get_default_config(GeneratorType.TRANSFORMER)
    print(f"Default config: {config}")
    
    # Create generator using factory
    generator = create_generator(GeneratorType.TRANSFORMER, config)
    print(f"Created generator: {type(generator).__name__}")
    
    # Generate sequences
    batch_size = 4
    with torch.no_grad():
        output = generator.generate(batch_size)
        print(f"Generated sequences shape: {output.shape}")


def example_factory_usage():
    """Example of using the factory with custom configurations."""
    print("\n=== Factory Usage with Custom Configs ===")
    
    # Custom RNN configuration
    custom_rnn_config = {
        'hidden_size': 128,
        'num_layers': 3,
        'sequence_length': 50,
        'dropout': 0.2
    }
    
    rnn_gen = create_generator(GeneratorType.LSTM, custom_rnn_config)
    print(f"Custom LSTM generator created with hidden_size={custom_rnn_config['hidden_size']}")
    
    # Custom Transformer configuration
    custom_transformer_config = {
        'embed_dim': 128,
        'num_heads': 16,
        'num_layers': 6,
        'sequence_length': 200
    }
    
    transformer_gen = create_generator(GeneratorType.TRANSFORMER, custom_transformer_config)
    print(f"Custom Transformer generator created with embed_dim={custom_transformer_config['embed_dim']}")


if __name__ == "__main__":
    print("Generator Module Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_canned_net()
    example_neural_sde()
    example_rnn_generator()
    example_transformer_generator()
    example_factory_usage()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nThe generators module provides:")
    print("1. CannedNet: Signature-aware deterministic architecture")
    print("2. Neural SDE: Continuous-time stochastic differential equations")
    print("3. RNN/LSTM/GRU: Standard recurrent architectures for baselines")
    print("4. Transformer: Attention-based sequence modeling")
    print("5. Factory functions: Unified interface for creating any generator type")
    print("\nUse create_generator(GeneratorType.X, config) to create any generator!")

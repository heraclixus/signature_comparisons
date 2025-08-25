"""
Demo training script for Distributional Diffusion.

This script demonstrates how to train the distributional diffusion model
on synthetic Brownian motion data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def generate_brownian_motion(num_samples: int, seq_len: int, dim: int = 1, dt: float = 0.01) -> torch.Tensor:
    """Generate synthetic Brownian motion data."""
    # Generate increments
    dW = torch.randn(num_samples, dim, seq_len) * np.sqrt(dt)
    
    # Cumulative sum to get Brownian paths
    paths = torch.cumsum(dW, dim=-1)
    
    # Start from zero
    paths = torch.cat([torch.zeros(num_samples, dim, 1), paths], dim=-1)
    
    return paths[:, :, :seq_len]  # Ensure correct length


def plot_samples(real_samples: torch.Tensor, generated_samples: torch.Tensor, title: str = "Samples"):
    """Plot real vs generated samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot real samples
    for i in range(min(10, real_samples.shape[0])):
        ax1.plot(real_samples[i, 0, :].numpy(), alpha=0.7)
    ax1.set_title("Real Samples")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    
    # Plot generated samples
    for i in range(min(10, generated_samples.shape[0])):
        ax2.plot(generated_samples[i, 0, :].detach().numpy(), alpha=0.7)
    ax2.set_title("Generated Samples")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def main():
    """Main demo function."""
    print("=" * 60)
    print("Distributional Diffusion Demo Training")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration - optimized for speed
    config = {
        'dim': 1,
        'seq_len': 16,          # Shorter sequences for faster computation
        'num_samples': 200,     # Much smaller dataset
        'batch_size': 100,      # Larger batch size for efficiency
        'num_epochs': 10,       # Fewer epochs
        'learning_rate': 5e-3,  # Higher learning rate for faster convergence
        'population_size': 4,   # Smaller population for faster signature computation
        'lambda_param': 1.0,
        'device': 'cpu'
    }
    
    device = config['device']
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    print()
    
    # Import components
    try:
        from models.tsdiff.diffusion.distributional_diffusion import DistributionalDiffusion
        from models.distributional_generator import create_distributional_generator
    except ImportError as e:
        print(f"Error importing components: {e}")
        return
    
    # Generate training data
    print("Generating training data...")
    train_data = generate_brownian_motion(
        num_samples=config['num_samples'],
        seq_len=config['seq_len'],
        dim=config['dim']
    )
    print(f"Training data shape: {train_data.shape}")
    
    # Create models
    print("Creating models...")
    ddm = DistributionalDiffusion(
        dim=config['dim'],
        seq_len=config['seq_len'],
        population_size=config['population_size'],
        lambda_param=config['lambda_param']
    ).to(device)
    
    generator = create_distributional_generator(
        generator_type="feedforward",
        data_size=config['dim'],
        seq_len=config['seq_len'],
        hidden_size=32,  # Smaller network for faster training
        num_layers=2     # Fewer layers
    ).to(device)
    
    print(f"DDM parameters: {sum(p.numel() for p in ddm.parameters())}")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    
    # Setup optimizer
    optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    losses = []
    
    for epoch in range(config['num_epochs']):
        epoch_losses = []
        
        # Create batches
        num_batches = len(train_data) // config['batch_size']
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * config['batch_size']
            end_idx = start_idx + config['batch_size']
            batch_data = train_data[start_idx:end_idx].to(device)
            
            # Training step
            optimizer.zero_grad()
            loss = ddm.get_loss(generator, batch_data)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 2 == 0 or epoch == 0:  # Print more frequently for shorter training
            print(f"Epoch {epoch+1:3d}/{config['num_epochs']}: Loss = {avg_loss:.6f}")
    
    print("Training completed!")
    
    # Generate samples
    print("\nGenerating samples...")
    generator.eval()
    
    with torch.no_grad():
        generated_samples = ddm.sample(
            generator=generator,
            num_samples=20,      # Fewer samples for faster generation
            num_coarse_steps=10, # Fewer coarse steps
            device=device
        )
    
    print(f"Generated samples shape: {generated_samples.shape}")
    
    # Plot results
    print("Creating plots...")
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    
    # Plot sample comparison
    plt.subplot(1, 2, 2)
    
    # Plot some real samples
    for i in range(5):
        plt.plot(train_data[i, 0, :].numpy(), 'b-', alpha=0.5, label='Real' if i == 0 else '')
    
    # Plot some generated samples
    for i in range(5):
        plt.plot(generated_samples[i, 0, :].cpu().numpy(), 'r--', alpha=0.5, label='Generated' if i == 0 else '')
    
    plt.title("Sample Comparison")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute some basic statistics
    print("\nStatistics:")
    print(f"Real data - Mean: {train_data.mean():.4f}, Std: {train_data.std():.4f}")
    print(f"Generated - Mean: {generated_samples.mean():.4f}, Std: {generated_samples.std():.4f}")
    
    # Compute final loss on test data
    test_data = generate_brownian_motion(50, config['seq_len'], config['dim']).to(device)
    
    with torch.no_grad():
        final_loss = ddm.get_loss(generator, test_data)
    
    print(f"Final test loss: {final_loss.item():.6f}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()

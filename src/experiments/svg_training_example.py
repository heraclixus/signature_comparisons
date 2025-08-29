#!/usr/bin/env python3
"""
SVG Training Example

This script shows how to train the SVG model on Moving MNIST
with proper configuration and monitoring.
"""

import subprocess
import sys
from pathlib import Path

def run_svg_training_example():
    """Run a complete SVG training example."""
    print("🎬 SVG TRAINING EXAMPLE")
    print("=" * 50)
    print("📚 Based on: Denton & Fergus - 'Stochastic Video Generation with a Learned Prior'")
    print("🔗 Original: https://github.com/edenton/svg/")
    print()
    
    # Recommended training configuration
    config = {
        'dataset': 'smmnist',
        'num_digits': 2,
        'g_dim': 128,         # Encoder/decoder feature dimension
        'z_dim': 10,          # Latent dimension
        'beta': 0.0001,       # KL divergence weight
        'data_root': '../data',
        'log_dir': '../results/svg_moving_mnist',
        'niter': 20,          # 20 epochs
        'epoch_size': 100,    # 100 batches per epoch  
        'batch_size': 16,     # Reasonable batch size
        'n_past': 5,          # Condition on 5 frames
        'n_future': 10,       # Predict 10 frames
        'n_eval': 25,         # Evaluate on 25 frames
        'image_width': 64,    # 64x64 images
        'rnn_size': 256,      # RNN hidden size
        'lr': 0.002,          # Learning rate
        'model': 'dcgan',     # Use DCGAN encoder/decoder
        'data_threads': 0     # No multiprocessing
    }
    
    print("🔧 Recommended Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print(f"\n📊 Expected Training:")
    print(f"   • Duration: ~30-60 minutes")
    print(f"   • Memory: ~2-4 GB")
    print(f"   • Output: Model + generated video samples")
    
    # Build command
    cmd = ['python', 'experiments/train_svg_lp.py']
    for key, value in config.items():
        cmd.extend([f'--{key}', str(value)])
    
    print(f"\n🚀 Training Command:")
    print(f"cd src")
    print(f"conda activate sig19")
    print(f"{' '.join(cmd)}")
    
    print(f"\n💡 Alternative Configurations:")
    print(f"   • Single digit: --num_digits 1")
    print(f"   • Longer sequences: --n_future 15 --n_eval 30")
    print(f"   • Higher resolution: --image_width 128 (requires dcgan_128)")
    print(f"   • VGG encoder: --model vgg")
    
    print(f"\n📁 Output Location:")
    print(f"   {config['log_dir']}/smmnist-{config['num_digits']}/[model_config]/")
    print(f"   ├── model.pth          # Trained model checkpoint")
    print(f"   ├── gen/")
    print(f"   │   ├── sample_*.png   # Generated vs ground truth")
    print(f"   │   ├── sample_*.gif   # Generated video sequences") 
    print(f"   │   └── rec_*.png      # Reconstruction quality")
    print(f"   └── plots/             # Additional visualizations")


if __name__ == "__main__":
    run_svg_training_example()

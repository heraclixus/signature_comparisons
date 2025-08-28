#!/usr/bin/env python3
"""
Generate Moving MNIST Video Datasets

This script generates and saves Moving MNIST video datasets to disk
in raw video format (seq_len, height, width, channels) suitable for
encoder/decoder architectures and latent space modeling.

Usage:
    python generate_moving_mnist_video_datasets.py [--standard]
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from dataset.multi_dataset import MultiDatasetManager
from utils.dataset_persistence import create_dataset_persistence


def generate_standard_video_datasets(force: bool = False):
    """Generate standard Moving MNIST video dataset configurations."""
    print("🎬 GENERATING STANDARD MOVING MNIST VIDEO DATASETS")
    print("=" * 70)
    print("📺 Format: Raw video sequences (seq_len, height, width, channels)")
    print("🎯 Purpose: Encoder/decoder architectures and latent space modeling")
    print()
    
    # Initialize managers
    manager = MultiDatasetManager(use_persistence=True)
    persistence = create_dataset_persistence()
    
    # Standard video configurations
    video_configs = [
        {
            'name': 'moving_mnist_video',
            'description': 'Standard Moving MNIST video (2 digits, 20 frames)',
            'params': {'num_samples': 1000, 'seq_len': 20, 'num_digits': 2}
        },
        {
            'name': 'moving_mnist_video_single',
            'description': 'Single digit Moving MNIST video (1 digit, 25 frames)',
            'params': {'num_samples': 1000, 'seq_len': 25, 'num_digits': 1}
        },
        {
            'name': 'moving_mnist_video_long',
            'description': 'Long sequence Moving MNIST video (2 digits, 50 frames)',
            'params': {'num_samples': 500, 'seq_len': 50, 'num_digits': 2}  # Fewer samples due to longer sequences
        }
    ]
    
    successful = 0
    failed = 0
    total_memory = 0
    
    for config in video_configs:
        dataset_name = config['name']
        description = config['description']
        params = config['params']
        
        print(f"\n{'='*50}")
        print(f"🎬 Generating {dataset_name}")
        print(f"{'='*50}")
        print(f"📊 {description}")
        print(f"   Samples: {params['num_samples']}")
        print(f"   Sequence length: {params['seq_len']}")
        print(f"   Digits per sequence: {params['num_digits']}")
        
        try:
            # Check if dataset already exists (if not forcing)
            if not force:
                # Create parameter hash for checking existence
                check_params = {
                    'num_samples': params['num_samples'],
                    'seq_len': params['seq_len'],
                    'num_digits': params['num_digits'],
                    'image_size': 64,
                    'deterministic': True
                }
                
                if persistence.dataset_exists(dataset_name, check_params):
                    print(f"✅ {dataset_name} already exists. Use --force to regenerate.")
                    successful += 1
                    continue
            
            # Generate dataset
            print(f"🏭 Generating {dataset_name}...")
            dataset = manager.get_dataset(dataset_name, **params)
            
            # Get sample to check format
            sample, label = dataset[0]
            memory_mb = sample.numel() * len(dataset) * 4 / (1024*1024)  # float32 = 4 bytes
            total_memory += memory_mb
            
            print(f"✅ {dataset_name.upper()} completed:")
            print(f"   📊 Samples: {len(dataset):,}")
            print(f"   📏 Sample shape: {sample.shape}")
            print(f"   📺 Format: (seq_len, height, width, channels)")
            print(f"   💾 Estimated size: {memory_mb:.1f} MB")
            print(f"   🎯 Perfect for encoder/decoder architectures!")
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed to generate {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("📋 GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(video_configs)}")
    print(f"💾 Total estimated storage: {total_memory:.1f} MB")
    
    if successful > 0:
        print(f"\n🎉 Successfully generated {successful} Moving MNIST video datasets!")
        print(f"   📁 Location: data/moving_mnist_video*/")
        print(f"   📺 Format: Raw video sequences (seq_len, height, width, channels)")
        print(f"   🎯 Ready for encoder/decoder architectures")
        print(f"   🧠 Perfect for latent space modeling")
        
        print(f"\n💡 Usage Examples:")
        print(f"   ```python")
        print(f"   from dataset.multi_dataset import MultiDatasetManager")
        print(f"   manager = MultiDatasetManager()")
        print(f"   dataset = manager.get_dataset('moving_mnist_video')")
        print(f"   sample, label = dataset[0]  # Shape: (20, 64, 64, 1)")
        print(f"   ```")
    
    if failed > 0:
        print(f"\n⚠️ {failed} dataset(s) failed to generate")
        return 1
    
    return 0


def generate_custom_video_dataset(dataset_name: str, num_samples: int, seq_len: int,
                                 num_digits: int, force: bool = False):
    """Generate a custom Moving MNIST video dataset."""
    print(f"🎬 GENERATING CUSTOM MOVING MNIST VIDEO DATASET")
    print(f"=" * 60)
    print(f"📺 Dataset: {dataset_name}")
    print(f"📊 Samples: {num_samples}")
    print(f"⏱️ Sequence length: {seq_len}")
    print(f"🔢 Digits per sequence: {num_digits}")
    
    try:
        # Create manager
        manager = MultiDatasetManager(use_persistence=True)
        
        # Generate dataset
        dataset = manager.get_dataset(
            'moving_mnist_video',  # Use the base video generator
            num_samples=num_samples,
            seq_len=seq_len,
            num_digits=num_digits
        )
        
        # Get sample info
        sample, label = dataset[0]
        memory_mb = sample.numel() * len(dataset) * 4 / (1024*1024)
        
        print(f"\n✅ Custom dataset generated successfully!")
        print(f"   📊 Samples: {len(dataset)}")
        print(f"   📏 Sample shape: {sample.shape}")
        print(f"   📺 Format: (seq_len, height, width, channels)")
        print(f"   💾 Size: {memory_mb:.1f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate custom dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Generate Moving MNIST Video Datasets")
    
    # Generation options
    parser.add_argument("--standard", action="store_true",
                       help="Generate all standard Moving MNIST video configurations")
    parser.add_argument("--force", action="store_true",
                       help="Force regeneration even if datasets exist")
    
    # Custom dataset options
    parser.add_argument("--dataset_name", type=str, default="custom_moving_mnist_video",
                       help="Name for custom dataset")
    parser.add_argument("--num_samples", type=int, default=500,
                       help="Number of video sequences")
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Length of each video sequence")
    parser.add_argument("--num_digits", type=int, default=2,
                       help="Number of digits per sequence")
    
    args = parser.parse_args()
    
    try:
        if args.standard:
            # Generate standard configurations
            return generate_standard_video_datasets(force=args.force)
        else:
            # Generate custom dataset
            return generate_custom_video_dataset(
                dataset_name=args.dataset_name,
                num_samples=args.num_samples,
                seq_len=args.seq_len,
                num_digits=args.num_digits,
                force=args.force
            )
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

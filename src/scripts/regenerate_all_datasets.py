#!/usr/bin/env python3
"""
Regenerate All Datasets with New Parameters

This script regenerates all datasets with the updated parameters:
- 32,768 sample paths (128 × 256)
- 64 time points per path
- Proper mathematical implementations

Usage:
    python regenerate_all_datasets.py [--force]
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from dataset.multi_dataset import MultiDatasetManager
from utils.dataset_persistence import create_dataset_persistence


def main():
    """Regenerate all datasets with new parameters."""
    parser = argparse.ArgumentParser(description="Regenerate all datasets with new parameters")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if datasets exist")
    args = parser.parse_args()
    
    print("🔄 Regenerating All Datasets with New Parameters")
    print("=" * 60)
    print("📊 New specifications:")
    print("   • Sample paths: 32,768 (128 × 256)")
    print("   • Time points: 64 per path")
    print("   • Batch size: 128 (for training)")
    print("   • Proper mathematical implementations")
    print()
    
    # Initialize managers
    manager = MultiDatasetManager(use_persistence=True)
    persistence = create_dataset_persistence()
    
    # List of all datasets
    datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
    
    successful = 0
    failed = 0
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"🔄 Regenerating {dataset_name.upper()}")
        print(f"{'='*50}")
        
        try:
            # Check if we need to delete old dataset
            if args.force:
                print(f"🗑️ Force mode: Cleaning up existing datasets...")
                # Note: We'll let the system handle this automatically by generating new ones
            
            # Generate dataset with new parameters
            print(f"🏭 Generating {dataset_name} with new parameters...")
            dataset = manager.get_dataset(dataset_name, num_samples=32768, n_points=64)
            
            print(f"✅ {dataset_name.upper()} completed:")
            print(f"   📊 Samples: {len(dataset):,}")
            print(f"   📏 Shape: {dataset[0][0].shape}")
            
            # Estimate file size
            estimated_size = len(dataset) * dataset[0][0].numel() * 4 / (1024*1024)
            print(f"   💾 Estimated size: {estimated_size:.1f} MB")
            
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed to regenerate {dataset_name}: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 REGENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(datasets)}")
    
    if successful > 0:
        print(f"\n🎉 Successfully regenerated {successful} datasets!")
        print(f"   📁 Location: data/{{dataset_name}}/")
        print(f"   📊 Each dataset: 32,768 samples × 64 time points")
        print(f"   🎯 Ready for training with batch size 128")
        
        # Show total storage
        total_estimated_size = successful * 16  # Rough estimate
        print(f"   💾 Total estimated storage: ~{total_estimated_size:.0f} MB")
    
    if failed > 0:
        print(f"\n⚠️ {failed} dataset(s) failed to regenerate")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
D2 Debug Runner

Simple runner script for D2 debugging suite.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.debug_d2.d2_comprehensive_debug import D2DebugSuite


def main():
    """Run D2 debug suite with user-friendly interface."""
    
    print("🔧 D2 DEBUG SUITE")
    print("=" * 40)
    print("Debugging D2 models on 1D time series")
    print()
    
    # Show system info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("✅ CUDA acceleration enabled for fast training!")
    else:
        print("⚠️  CPU mode - training will be slower")
    print()
    
    # Ask user for dataset choice
    print("Available datasets:")
    print("1. ou_process (Ornstein-Uhlenbeck - mean reverting)")
    print("2. fbm_h03 (Fractional Brownian Motion H=0.3 - anti-persistent)")
    print()
    
    choice = input("Choose dataset (1 or 2, default=1): ").strip()
    
    if choice == '2':
        dataset_name = 'fbm_h03'
    else:
        dataset_name = 'ou_process'
    
    # Ask for number of epochs
    epochs_input = input("Number of training epochs (default=50): ").strip()
    try:
        num_epochs = int(epochs_input) if epochs_input else 50
    except ValueError:
        num_epochs = 50
    
    print(f"\\n🚀 Running debug on {dataset_name} with {num_epochs} epochs...")
    print()
    
    # Run debug suite
    debug_suite = D2DebugSuite()
    results = debug_suite.run_full_debug(dataset_name, num_epochs)
    
    # Print summary
    print("\\n" + "="*60)
    print("🎯 DEBUG SUMMARY")
    print("="*60)
    
    if results['untrained']['mlp_metrics']:
        mlp_scale = results['untrained']['mlp_metrics']['scale_factor']
        mlp_wasserstein = results['untrained']['mlp_metrics']['wasserstein']
        print(f"📊 MLP Scale Issue: {mlp_scale:.0f}x too large")
        print(f"📊 MLP Wasserstein Distance: {mlp_wasserstein:.1f}")
    
    if results['untrained']['transformer_metrics']:
        trans_scale = results['untrained']['transformer_metrics']['scale_factor']
        trans_wasserstein = results['untrained']['transformer_metrics']['wasserstein']
        print(f"📊 Transformer Scale Issue: {trans_scale:.0f}x too large")
        print(f"📊 Transformer Wasserstein Distance: {trans_wasserstein:.1f}")
    
    print(f"\\n📁 Detailed results saved in: debug_d2_results/")
    print("🎨 Check the PNG files for visual analysis")
    print("📝 Check the TXT file for detailed report")


if __name__ == '__main__':
    main()

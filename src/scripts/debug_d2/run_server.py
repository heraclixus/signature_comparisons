#!/usr/bin/env python3
"""
D2 Debug Server Runner

Non-interactive script optimized for server/cluster execution.
Automatically runs comprehensive D2 debugging with CUDA acceleration.
"""

import sys
import os
import torch
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from scripts.debug_d2.d2_comprehensive_debug import D2DebugSuite


def main():
    """Run D2 debug suite optimized for server execution."""
    
    parser = argparse.ArgumentParser(description='D2 Debug Suite - Server Mode')
    parser.add_argument('--dataset', choices=['ou_process', 'fbm_h03'], 
                       default='ou_process', help='Dataset to use for debugging')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--output-dir', default='debug_d2_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("🔧 D2 DEBUG SUITE - SERVER MODE")
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Show system info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🔥 CUDA Cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
        print("✅ CUDA acceleration enabled!")
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
    else:
        print("⚠️  CPU mode - training will be slower")
        print("💡 Consider using a GPU server for faster results")
    print()
    
    # Run debug suite
    print(f"🚀 Starting comprehensive D2 debug on {args.dataset}...")
    print(f"⏱️  Estimated time: {args.epochs // 10} minutes on GPU, {args.epochs // 2} minutes on CPU")
    print()
    
    debug_suite = D2DebugSuite(args.output_dir)
    results = debug_suite.run_full_debug(args.dataset, args.epochs)
    
    # Print final summary
    print("\\n" + "="*60)
    print("🎯 SERVER DEBUG SUMMARY")
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
    
    # Training results summary
    if results.get('trained'):
        print(f"\\n🚀 Training Results:")
        if results['trained'].get('mlp_metrics'):
            trained_mlp_w = results['trained']['mlp_metrics']['wasserstein']
            print(f"📈 MLP After Training: {trained_mlp_w:.1f} Wasserstein")
        if results['trained'].get('transformer_metrics'):
            trained_trans_w = results['trained']['transformer_metrics']['wasserstein']
            print(f"📈 Transformer After Training: {trained_trans_w:.1f} Wasserstein")
    
    print(f"\\n📁 Results saved in: {args.output_dir}/")
    print("🎨 Check PNG files for visual analysis")
    print("📝 Check TXT file for detailed report")
    
    # GPU cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("\\n🧹 GPU memory cleaned up")


if __name__ == '__main__':
    main()

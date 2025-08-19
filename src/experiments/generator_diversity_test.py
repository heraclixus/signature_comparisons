"""
Generator Diversity Test

This script tests whether our generators are truly producing diverse random
sample paths or just deterministic outputs based on input patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.implementations.a1_final import create_a1_final_model
from models.implementations.b4_nsde_mmd import create_b4_model
from models.implementations.c1_gru_tstatistic import create_c1_model
from dataset import generative_model


def test_generator_diversity(model, model_name: str, n_tests: int = 5):
    """
    Test if a generator produces diverse outputs for the same input.
    
    Args:
        model: Model to test
        model_name: Name for reporting
        n_tests: Number of generation tests
        
    Returns:
        Dictionary with diversity metrics
    """
    print(f"\nTesting {model_name} Generator Diversity")
    print("-" * 40)
    
    model.eval()
    batch_size = 8
    
    # Test 1: Same input, multiple generations
    print(f"Test 1: Multiple generations with same random seed")
    
    all_outputs = []
    
    for test_i in range(n_tests):
        torch.manual_seed(42)  # Same seed each time
        np.random.seed(42)
        
        with torch.no_grad():
            output = model.generate_samples(batch_size)
        
        all_outputs.append(output.cpu().numpy())
        print(f"  Generation {test_i + 1}: Shape {output.shape}, Mean {output.mean().item():.4f}, Std {output.std().item():.4f}")
    
    # Check if outputs are identical (they should be with same seed)
    identical_outputs = True
    for i in range(1, len(all_outputs)):
        if not np.allclose(all_outputs[0], all_outputs[i], atol=1e-6):
            identical_outputs = False
            break
    
    print(f"  Same seed → Same output: {'✅' if identical_outputs else '❌'} ({'Expected' if identical_outputs else 'Unexpected'})")
    
    # Test 2: Different seeds, should produce different outputs
    print(f"\nTest 2: Multiple generations with different random seeds")
    
    diverse_outputs = []
    
    for test_i in range(n_tests):
        torch.manual_seed(42 + test_i)  # Different seed each time
        np.random.seed(42 + test_i)
        
        with torch.no_grad():
            output = model.generate_samples(batch_size)
        
        diverse_outputs.append(output.cpu().numpy())
        print(f"  Generation {test_i + 1}: Shape {output.shape}, Mean {output.mean().item():.4f}, Std {output.std().item():.4f}")
    
    # Check diversity
    diversity_scores = []
    for i in range(1, len(diverse_outputs)):
        mse_diff = np.mean((diverse_outputs[0] - diverse_outputs[i])**2)
        diversity_scores.append(mse_diff)
    
    mean_diversity = np.mean(diversity_scores)
    print(f"  Average MSE between different seeds: {mean_diversity:.6f}")
    print(f"  Diversity: {'✅ Good' if mean_diversity > 0.1 else '⚠️ Limited' if mean_diversity > 0.01 else '❌ Poor'}")
    
    # Test 3: Check if generator uses randomness internally
    print(f"\nTest 3: Internal randomness check")
    
    # Generate with same external seed but check internal variation
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate multiple batches
    batch_outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = model.generate_samples(batch_size)
        batch_outputs.append(output.cpu().numpy())
    
    # Check variation within and between batches
    within_batch_std = np.std(batch_outputs[0], axis=0).mean()
    between_batch_variation = np.std([np.mean(batch) for batch in batch_outputs])
    
    print(f"  Within-batch variation (std): {within_batch_std:.4f}")
    print(f"  Between-batch variation: {between_batch_variation:.4f}")
    
    # Test 4: Check specific samples from same batch
    print(f"\nTest 4: Sample diversity within batch")
    
    torch.manual_seed(42)
    with torch.no_grad():
        batch_output = model.generate_samples(8)
    
    # Check pairwise differences
    pairwise_diffs = []
    for i in range(batch_output.shape[0]):
        for j in range(i + 1, batch_output.shape[0]):
            diff = torch.mean((batch_output[i] - batch_output[j])**2).item()
            pairwise_diffs.append(diff)
    
    mean_pairwise_diff = np.mean(pairwise_diffs)
    print(f"  Mean pairwise MSE within batch: {mean_pairwise_diff:.6f}")
    print(f"  Intra-batch diversity: {'✅ Good' if mean_pairwise_diff > 0.1 else '⚠️ Limited' if mean_pairwise_diff > 0.01 else '❌ Poor'}")
    
    return {
        'model_name': model_name,
        'identical_with_same_seed': identical_outputs,
        'mean_diversity_different_seeds': mean_diversity,
        'within_batch_variation': within_batch_std,
        'between_batch_variation': between_batch_variation,
        'mean_pairwise_difference': mean_pairwise_diff,
        'diversity_assessment': 'good' if mean_diversity > 0.1 and mean_pairwise_diff > 0.1 else 'limited' if mean_diversity > 0.01 else 'poor'
    }


def analyze_generator_mechanisms():
    """Analyze how each generator type produces randomness."""
    
    print("GENERATOR MECHANISM ANALYSIS")
    print("=" * 50)
    print("Checking if generators are truly generative (produce diverse random paths)")
    
    # Setup test data
    batch_size = 32
    example_batch = torch.randn(batch_size, 2, 100)
    real_data = torch.randn(batch_size, 2, 100)
    
    results = []
    
    # Test A1 (CannedNet)
    print(f"\n{'='*60}")
    print("TESTING A1 (CannedNet + T-Statistic)")
    print(f"{'='*60}")
    
    try:
        torch.manual_seed(12345)
        a1_model = create_a1_final_model(example_batch, real_data)
        a1_results = test_generator_diversity(a1_model, "A1 (CannedNet)", n_tests=5)
        results.append(a1_results)
    except Exception as e:
        print(f"❌ A1 test failed: {e}")
    
    # Test B4 (Neural SDE)
    print(f"\n{'='*60}")
    print("TESTING B4 (Neural SDE + MMD)")
    print(f"{'='*60}")
    
    try:
        torch.manual_seed(12345)
        b4_model = create_b4_model(example_batch, real_data)
        b4_results = test_generator_diversity(b4_model, "B4 (Neural SDE)", n_tests=5)
        results.append(b4_results)
    except Exception as e:
        print(f"❌ B4 test failed: {e}")
    
    # Test C1 (GRU)
    print(f"\n{'='*60}")
    print("TESTING C1 (GRU + T-Statistic)")
    print(f"{'='*60}")
    
    try:
        torch.manual_seed(12345)
        c1_model = create_c1_model(example_batch, real_data)
        c1_results = test_generator_diversity(c1_model, "C1 (GRU)", n_tests=5)
        results.append(c1_results)
    except Exception as e:
        print(f"❌ C1 test failed: {e}")
    
    return results


def create_diversity_visualization(results: list, save_dir: str):
    """Create visualization of generator diversity."""
    
    if not results:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = [r['model_name'] for r in results]
    colors = ['blue', 'red', 'green', 'orange'][:len(models)]
    
    # 1. Diversity between different seeds
    ax = axes[0, 0]
    diversity_scores = [r['mean_diversity_different_seeds'] for r in results]
    bars = ax.bar(models, diversity_scores, color=colors, alpha=0.7)
    ax.set_ylabel('Mean MSE Between Different Seeds')
    ax.set_title('Generator Diversity\n(Higher = More Random)')
    ax.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, diversity_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Within-batch variation
    ax = axes[0, 1]
    within_batch_vars = [r['within_batch_variation'] for r in results]
    bars = ax.bar(models, within_batch_vars, color=colors, alpha=0.7)
    ax.set_ylabel('Within-batch Std Deviation')
    ax.set_title('Intra-batch Diversity')
    ax.grid(True, alpha=0.3)
    
    for bar, var in zip(bars, within_batch_vars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{var:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Pairwise differences within batch
    ax = axes[1, 0]
    pairwise_diffs = [r['mean_pairwise_difference'] for r in results]
    bars = ax.bar(models, pairwise_diffs, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Pairwise MSE')
    ax.set_title('Sample Diversity Within Batch')
    ax.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars, pairwise_diffs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary assessment
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "Generator Assessment\n" + "="*20 + "\n\n"
    
    for result in results:
        model = result['model_name'].split()[0]  # Get just the model ID
        assessment = result['diversity_assessment']
        
        status = "✅" if assessment == 'good' else "⚠️" if assessment == 'limited' else "❌"
        
        summary_text += f"{model}: {status} {assessment.title()}\n"
        summary_text += f"  Seed diversity: {result['mean_diversity_different_seeds']:.4f}\n"
        summary_text += f"  Batch diversity: {result['mean_pairwise_difference']:.4f}\n\n"
    
    # Overall assessment
    good_generators = sum(1 for r in results if r['diversity_assessment'] == 'good')
    
    if good_generators == len(results):
        summary_text += "Overall: ✅ All Good\n"
    elif good_generators > 0:
        summary_text += f"Overall: ⚠️ Mixed\n({good_generators}/{len(results)} good)\n"
    else:
        summary_text += "Overall: ❌ Issues\nNeed better randomness\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'generator_diversity_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Diversity analysis saved to: {save_dir}/generator_diversity_analysis.png")
    plt.close()


def check_current_generator_usage():
    """Check how we're currently using generators in our models."""
    
    print("CURRENT GENERATOR USAGE ANALYSIS")
    print("=" * 50)
    
    # Check CannedNet usage
    print(f"\n1. CannedNet (A1, A2, A3):")
    print(f"   - Input: Noise tensor (batch, channels, time)")
    print(f"   - Process: Feed-forward through signature layers")
    print(f"   - Output: Deterministic transformation of input")
    print(f"   - Randomness: Only from input noise")
    
    # Check Neural SDE usage
    print(f"\n2. Neural SDE (B4):")
    print(f"   - Input: Time grid + initial conditions")
    print(f"   - Process: Solve SDE with learned drift/diffusion")
    print(f"   - Output: Stochastic path from SDE solution")
    print(f"   - Randomness: From SDE noise process + initial conditions")
    
    # Check GRU usage
    print(f"\n3. GRU (C1, C2, C3):")
    print(f"   - Input: Time + noise sequence")
    print(f"   - Process: Sequential processing through RNN")
    print(f"   - Output: Deterministic transformation of input sequence")
    print(f"   - Randomness: Only from input noise sequence")
    
    print(f"\n🤔 POTENTIAL ISSUES:")
    print(f"   • CannedNet: May be too deterministic given input")
    print(f"   • GRU: May memorize patterns rather than generate randomly")
    print(f"   • Neural SDE: Should be most truly generative")
    
    print(f"\n🔍 WHAT TO CHECK:")
    print(f"   1. Do models produce different outputs with different random seeds?")
    print(f"   2. Do models produce diverse samples within the same batch?")
    print(f"   3. Are we using enough randomness in the generation process?")
    print(f"   4. Are models truly generative or just sophisticated regressors?")


def main():
    """Main diversity testing function."""
    print("Generator Diversity and Randomness Analysis")
    print("=" * 60)
    print("Testing if our generators produce truly diverse random sample paths")
    
    # Analyze current usage
    check_current_generator_usage()
    
    # Test actual diversity
    try:
        results = analyze_generator_mechanisms()
        
        if results:
            # Create visualization
            os.makedirs('results/evaluation', exist_ok=True)
            create_diversity_visualization(results, 'results/evaluation')
            
            # Summary assessment
            print(f"\n" + "="*60)
            print("GENERATOR DIVERSITY ASSESSMENT")
            print("="*60)
            
            for result in results:
                model = result['model_name']
                assessment = result['diversity_assessment']
                
                print(f"{model}:")
                print(f"  Assessment: {assessment.upper()}")
                print(f"  Seed-based diversity: {result['mean_diversity_different_seeds']:.6f}")
                print(f"  Batch diversity: {result['mean_pairwise_difference']:.6f}")
                
                if assessment == 'poor':
                    print(f"  ⚠️ May not be truly generative - investigate randomness sources")
                elif assessment == 'limited':
                    print(f"  ⚠️ Limited diversity - may need more randomness")
                else:
                    print(f"  ✅ Good diversity - appears truly generative")
            
            print(f"\n🎯 RECOMMENDATIONS:")
            print(f"   1. Ensure sufficient randomness in input generation")
            print(f"   2. Consider adding dropout or other stochastic elements during generation")
            print(f"   3. Verify that models are not just memorizing training patterns")
            print(f"   4. For RNNs, ensure input noise is varied enough to produce diversity")
            
            return True
        else:
            print(f"\n❌ No diversity tests completed")
            return False
            
    except Exception as e:
        print(f"\n❌ Diversity testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Generator diversity analysis complete")
        print(f"   Check results/evaluation/generator_diversity_analysis.png")
    else:
        print(f"\n❌ Analysis failed - check model implementations")

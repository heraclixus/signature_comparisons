"""
Evaluate Trained Models

This script evaluates trained models by loading them from checkpoints,
avoiding the need to retrain for evaluation. It automatically discovers
available trained models and compares them systematically.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from utils.model_checkpoint import create_checkpoint_manager


def compute_comprehensive_metrics(generated: torch.Tensor, ground_truth: torch.Tensor, 
                                model_id: str, checkpoint_info: Dict) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a trained model."""
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle shape mismatch
    if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
        gt_np = gt_np[:, 1, :]  # Extract value dimension
    
    # Match samples and length
    min_samples = min(gen_np.shape[0], gt_np.shape[0])
    gen_np = gen_np[:min_samples]
    gt_np = gt_np[:min_samples]
    
    if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:, :min_length]
        gt_np = gt_np[:, :min_length]
    
    # Compute metrics
    gen_flat = gen_np.flatten()
    gt_flat = gt_np.flatten()
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(gen_flat, gt_flat))
    mae = np.mean(np.abs(gen_flat - gt_flat))
    
    # Statistical tests
    ks_stat, ks_p = stats.ks_2samp(gen_flat, gt_flat)
    wasserstein_dist = stats.wasserstein_distance(gen_flat, gt_flat)
    
    # Moments
    gen_mean, gen_std = np.mean(gen_flat), np.std(gen_flat)
    gt_mean, gt_std = np.mean(gt_flat), np.std(gt_flat)
    gen_skew, gen_kurt = stats.skew(gen_flat), stats.kurtosis(gen_flat)
    gt_skew, gt_kurt = stats.skew(gt_flat), stats.kurtosis(gt_flat)
    
    # Path correlations
    correlations = []
    for i in range(min_samples):
        try:
            corr, _ = stats.pearsonr(gen_np[i], gt_np[i])
            if not np.isnan(corr):
                correlations.append(corr)
        except:
            pass
    
    mean_corr = np.mean(correlations) if correlations else 0.0
    std_corr = np.std(correlations) if correlations else 0.0
    
    return {
        # Model info
        'model_id': model_id,
        'training_epoch': checkpoint_info.get('epoch', 0),
        'training_loss': checkpoint_info.get('loss', 0.0),
        'total_parameters': checkpoint_info.get('total_parameters', 0),
        'model_class': checkpoint_info.get('model_class', 'Unknown'),
        
        # Generation quality
        'rmse': float(rmse),
        'mae': float(mae),
        
        # Distribution similarity
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'distributions_similar': ks_p > 0.05,
        'wasserstein_distance': float(wasserstein_dist),
        
        # Statistical moments
        'generated_mean': float(gen_mean),
        'generated_std': float(gen_std),
        'generated_skewness': float(gen_skew),
        'generated_kurtosis': float(gen_kurt),
        'ground_truth_mean': float(gt_mean),
        'ground_truth_std': float(gt_std),
        'ground_truth_skewness': float(gt_skew),
        'ground_truth_kurtosis': float(gt_kurt),
        
        # Moment differences
        'mean_abs_error': float(abs(gen_mean - gt_mean)),
        'std_abs_error': float(abs(gen_std - gt_std)),
        'mean_rel_error': float(abs(gen_mean - gt_mean) / (abs(gt_mean) + 1e-8)),
        'std_rel_error': float(abs(gen_std - gt_std) / (abs(gt_std) + 1e-8)),
        
        # Path correlations
        'mean_path_correlation': float(mean_corr),
        'std_path_correlation': float(std_corr),
        'n_valid_correlations': len(correlations),
        
        # Sample quality
        'generated_finite_ratio': float(np.sum(np.isfinite(gen_np)) / gen_np.size),
        'generated_range': float(np.max(gen_np) - np.min(gen_np)),
        'ground_truth_range': float(np.max(gt_np) - np.min(gt_np))
    }


def evaluate_model_performance(model, model_id: str, test_input: torch.Tensor, n_runs: int = 10):
    """Evaluate computational performance of a trained model."""
    model.eval()
    
    # Timing analysis
    forward_times = []
    
    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    # Timed runs
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        end_time = time.time()
        forward_times.append((end_time - start_time) * 1000)  # ms
    
    return {
        'mean_forward_time_ms': float(np.mean(forward_times)),
        'std_forward_time_ms': float(np.std(forward_times)),
        'min_forward_time_ms': float(np.min(forward_times)),
        'max_forward_time_ms': float(np.max(forward_times)),
        'output_shape': list(output.shape)
    }


def create_evaluation_visualization(results_df: pd.DataFrame, save_dir: str):
    """Create comprehensive evaluation visualization."""
    print(f"Creating evaluation visualization...")
    
    if results_df.empty:
        print("No results to visualize")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = results_df['model_id'].tolist()
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(models)]
    
    # 1. Generation Quality (RMSE)
    ax = axes[0, 0]
    rmse_vals = results_df['rmse'].tolist()
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Generation Quality (RMSE)\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribution Similarity
    ax = axes[0, 1]
    ks_vals = results_df['ks_statistic'].tolist()
    bars = ax.bar(models, ks_vals, color=colors, alpha=0.7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Distribution Similarity\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, ks_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Path Correlation
    ax = axes[0, 2]
    corr_vals = results_df['mean_path_correlation'].tolist()
    bars = ax.bar(models, corr_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Path Correlation')
    ax.set_title('Path-wise Correlation\n(Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, corr_vals):
        height = bar.get_height()
        y_pos = height + (0.01 if height >= 0 else -0.01)
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va=va, fontweight='bold')
    
    # 4. Training Performance
    ax = axes[1, 0]
    training_losses = results_df['training_loss'].tolist()
    training_epochs = results_df['training_epoch'].tolist()
    
    # Scatter plot: training loss vs epoch
    scatter = ax.scatter(training_epochs, training_losses, c=range(len(models)), 
                        cmap='viridis', s=100, alpha=0.7)
    
    for i, model in enumerate(models):
        ax.annotate(model, (training_epochs[i], training_losses[i]), 
                   xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax.set_xlabel('Training Epoch (Best)')
    ax.set_ylabel('Training Loss (Best)')
    ax.set_title('Training Performance')
    ax.grid(True, alpha=0.3)
    
    # 5. Computational Performance
    ax = axes[1, 1]
    if 'mean_forward_time_ms' in results_df.columns:
        time_vals = results_df['mean_forward_time_ms'].tolist()
        bars = ax.bar(models, time_vals, color=colors, alpha=0.7)
        ax.set_ylabel('Forward Time (ms)')
        ax.set_title('Computational Speed')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, time_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Performance data\nnot available', 
                ha='center', va='center', transform=ax.transAxes)
    
    # 6. Summary Rankings
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate rankings
    rmse_ranking = results_df.nsmallest(len(results_df), 'rmse')
    corr_ranking = results_df.nlargest(len(results_df), 'mean_path_correlation')
    
    summary_text = "Model Rankings\n" + "="*15 + "\n\n"
    
    summary_text += "By RMSE (↓ better):\n"
    for i, (_, row) in enumerate(rmse_ranking.iterrows(), 1):
        summary_text += f"  {i}. {row['model_id']}: {row['rmse']:.4f}\n"
    
    summary_text += f"\nBy Correlation (↑ better):\n"
    for i, (_, row) in enumerate(corr_ranking.iterrows(), 1):
        summary_text += f"  {i}. {row['model_id']}: {row['mean_path_correlation']:.4f}\n"
    
    # Overall winner
    best_rmse = rmse_ranking.iloc[0]['model_id']
    best_corr = corr_ranking.iloc[0]['model_id']
    
    summary_text += f"\n🏆 Best RMSE: {best_rmse}\n"
    summary_text += f"🏆 Best Correlation: {best_corr}\n\n"
    
    if best_rmse == best_corr:
        summary_text += f"Overall Winner: {best_rmse}"
    else:
        summary_text += f"Mixed Results:\nDifferent winners"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trained_models_evaluation.png'), dpi=300, bbox_inches='tight')
    print(f"Evaluation visualization saved to: {save_dir}/trained_models_evaluation.png")
    plt.close()


def evaluate_all_trained_models():
    """Evaluate all available trained models."""
    print("Evaluating All Trained Models")
    print("=" * 50)
    print("Loading models from checkpoints to avoid retraining")
    
    # Setup checkpoint manager
    checkpoint_manager = create_checkpoint_manager()
    
    # Check available models
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print("❌ No trained models found!")
        print("   Run train_and_save_models.py first to train models")
        return False
    
    print(f"\nFound {len(available_models)} trained models:")
    checkpoint_manager.print_available_models()
    
    # Setup evaluation data
    print(f"\nSetting up evaluation data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 64
    n_points = 100
    
    # Generate test data
    signals = generative_model.get_signal(num_samples=n_samples, n_points=n_points).tensors[0]
    train_dataset = generative_model.get_noise(n_points=n_points, num_samples=32)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    example_batch, _ = next(iter(train_loader))
    
    print(f"Evaluation data: {signals.shape}")
    
    # Evaluate each trained model
    results = []
    
    for model_id in available_models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_id}")
        print(f"{'='*50}")
        
        try:
            # Load model
            model = checkpoint_manager.load_model(model_id)
            if model is None:
                print(f"❌ Failed to load {model_id}")
                continue
            
            # Get checkpoint info
            checkpoint_info = checkpoint_manager.get_checkpoint_info(model_id)
            if checkpoint_info is None:
                checkpoint_info = {}
            
            print(f"Model loaded: {checkpoint_info.get('total_parameters', 0):,} parameters")
            print(f"Training: Epoch {checkpoint_info.get('epoch', 0)}, Loss {checkpoint_info.get('loss', 0.0):.6f}")
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                generated_samples = model.generate_samples(n_samples)
            
            print(f"Generated samples: {generated_samples.shape}")
            
            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(
                generated=generated_samples,
                ground_truth=signals,
                model_id=model_id,
                checkpoint_info=checkpoint_info
            )
            
            # Add performance metrics
            performance_metrics = evaluate_model_performance(model, model_id, example_batch[:8])
            metrics.update(performance_metrics)
            
            results.append(metrics)
            
            print(f"✅ {model_id} evaluation completed")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   Path Correlation: {metrics['mean_path_correlation']:.4f}")
            print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {model_id}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("❌ No models evaluated successfully")
        return False
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs('results/evaluation', exist_ok=True)
    results_path = 'results/evaluation/trained_models_evaluation.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n" + "="*60)
    print("TRAINED MODELS EVALUATION COMPLETE")
    print("="*60)
    
    print(f"Results saved to: {results_path}")
    print(f"Models evaluated: {len(results)}")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"{'Model':<10} {'RMSE':<8} {'Correlation':<12} {'KS Stat':<8} {'Training Loss':<12}")
    print("-" * 60)
    
    for _, row in results_df.iterrows():
        print(f"{row['model_id']:<10} {row['rmse']:<8.4f} {row['mean_path_correlation']:<12.4f} "
              f"{row['ks_statistic']:<8.4f} {row['training_loss']:<12.6f}")
    
    # Rankings
    print(f"\nRankings:")
    
    # Best RMSE
    best_rmse = results_df.nsmallest(1, 'rmse').iloc[0]
    print(f"🏆 Best RMSE: {best_rmse['model_id']} ({best_rmse['rmse']:.4f})")
    
    # Best correlation
    best_corr = results_df.nlargest(1, 'mean_path_correlation').iloc[0]
    print(f"🏆 Best Correlation: {best_corr['model_id']} ({best_corr['mean_path_correlation']:.4f})")
    
    # Best training
    best_training = results_df.nsmallest(1, 'training_loss').iloc[0]
    print(f"🏆 Best Training Loss: {best_training['model_id']} ({best_training['training_loss']:.6f})")
    
    # Create visualization
    create_evaluation_visualization(results_df, 'results/evaluation')
    
    print(f"\n✅ Evaluation complete! All trained models compared systematically.")
    
    return True


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained signature-based models")
    parser.add_argument("--list", action="store_true", help="List available trained models")
    parser.add_argument("--model", type=str, help="Evaluate specific model only")
    
    args = parser.parse_args()
    
    if args.list:
        checkpoint_manager = create_checkpoint_manager()
        checkpoint_manager.print_available_models()
        return
    
    if args.model:
        print(f"Evaluating specific model: {args.model}")
        # TODO: Implement single model evaluation
        return
    
    # Evaluate all trained models
    success = evaluate_all_trained_models()
    
    if success:
        print(f"\n🎉 SUCCESS: All trained models evaluated!")
        print(f"   Load results/evaluation/trained_models_evaluation.csv for analysis")
    else:
        print(f"\n❌ Evaluation failed or no trained models found")
        print(f"   Run train_and_save_models.py first to train models")


if __name__ == "__main__":
    main()

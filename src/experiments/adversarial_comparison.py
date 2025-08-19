"""
Adversarial vs Non-Adversarial Model Comparison

This script creates comprehensive comparisons between adversarial and non-adversarial
model variants, focusing on models that have both training types available.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_checkpoint import create_checkpoint_manager
from dataset.generative_model import get_signal
from experiments.adversarial_training import create_adversarial_model
from scipy import stats


def evaluate_single_model(model, eval_data: torch.Tensor, model_id: str) -> Dict[str, Any]:
    """Evaluate a single model and return metrics."""
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = model.generate_samples(64) if hasattr(model, 'generate_samples') else model(eval_data[:64])
        
        # Handle different output formats
        if samples.dim() == 2:
            # Add time dimension for compatibility
            batch_size, time_steps = samples.shape
            time_channel = torch.linspace(0, 1, time_steps, device=samples.device)
            time_channel = time_channel.unsqueeze(0).expand(batch_size, -1)
            samples = torch.stack([time_channel, samples], dim=1)
        
        # Compute metrics
        gen_np = samples[:, 1, :].detach().cpu().numpy()  # Extract values
        gt_np = eval_data[:len(gen_np), 1, :].detach().cpu().numpy()
        
        # Align shapes
        min_samples = min(gen_np.shape[0], gt_np.shape[0])
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:min_samples, :min_length]
        gt_np = gt_np[:min_samples, :min_length]
        
        # Compute metrics
        gen_flat = gen_np.flatten()
        gt_flat = gt_np.flatten()
        
        rmse = np.sqrt(np.mean((gen_flat - gt_flat) ** 2))
        ks_statistic, ks_p_value = stats.ks_2samp(gen_flat, gt_flat)
        wasserstein_dist = stats.wasserstein_distance(gen_flat, gt_flat)
        
        # Std analysis
        gen_std = np.std(gen_np, axis=0)
        gt_std = np.std(gt_np, axis=0)
        std_rmse = np.sqrt(np.mean((gen_std - gt_std) ** 2))
        
        return {
            'model_id': model_id,
            'rmse': rmse,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'wasserstein_distance': wasserstein_dist,
            'std_rmse': std_rmse
        }


def compare_adversarial_vs_non_adversarial_models(dataset_name: str = 'ou_process'):
    """Compare adversarial vs non-adversarial models for a specific dataset."""
    print(f"⚔️ Adversarial vs Non-Adversarial Comparison for {dataset_name.upper()}")
    print("=" * 70)
    
    # Setup evaluation data
    dataset = get_signal(num_samples=64)
    eval_data = torch.stack([dataset[i][0] for i in range(32)])
    
    # Get available models
    non_adv_manager = create_checkpoint_manager(f'results/{dataset_name}')
    adv_manager = create_checkpoint_manager(f'results/{dataset_name}_adversarial')
    
    non_adv_models = non_adv_manager.list_available_models()
    adv_models = adv_manager.list_available_models()
    
    print(f"Non-adversarial models: {non_adv_models}")
    print(f"Adversarial models: {adv_models}")
    
    # Find paired models (models that have both variants)
    adv_base_models = [model.replace('_ADV', '') for model in adv_models if model.endswith('_ADV')]
    paired_models = [model for model in non_adv_models if model in adv_base_models]
    
    print(f"Paired models (both variants available): {paired_models}")
    
    if not paired_models:
        print("❌ No models found with both adversarial and non-adversarial variants")
        return None
    
    # Evaluate paired models
    comparison_results = []
    
    for base_model_id in paired_models:
        print(f"\n📊 Comparing {base_model_id} vs {base_model_id}_ADV...")
        
        try:
            # Load non-adversarial model
            non_adv_model = non_adv_manager.load_model(base_model_id)
            if non_adv_model is None:
                print(f"❌ Failed to load non-adversarial {base_model_id}")
                continue
            
            # Evaluate non-adversarial
            non_adv_metrics = evaluate_single_model(non_adv_model, eval_data, base_model_id)
            non_adv_metrics['training_type'] = 'non_adversarial'
            
            # Create adversarial model and load weights
            example_batch = torch.randn(8, 2, 100)
            adversarial_model = create_adversarial_model(
                base_model_id=base_model_id,
                example_batch=example_batch,
                real_data=eval_data,
                adversarial=True,
                memory_efficient=True
            )
            
            # Load adversarial weights
            adv_model_path = f'results/{dataset_name}_adversarial/trained_models/{base_model_id}_ADV/model.pth'
            if os.path.exists(adv_model_path):
                state_dict = torch.load(adv_model_path, map_location='cpu')
                adversarial_model.load_state_dict(state_dict)
                print(f"✅ Loaded adversarial weights for {base_model_id}_ADV")
            
            # Evaluate adversarial
            adv_metrics = evaluate_single_model(adversarial_model, eval_data, f"{base_model_id}_ADV")
            adv_metrics['training_type'] = 'adversarial'
            adv_metrics['base_model_id'] = base_model_id
            
            comparison_results.extend([non_adv_metrics, adv_metrics])
            
            print(f"   Non-adversarial: RMSE={non_adv_metrics['rmse']:.4f}, KS={non_adv_metrics['ks_statistic']:.4f}")
            print(f"   Adversarial:     RMSE={adv_metrics['rmse']:.4f}, KS={adv_metrics['ks_statistic']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to compare {base_model_id}: {e}")
            continue
    
    if not comparison_results:
        print("❌ No successful comparisons")
        return None
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save results
    comparison_dir = f'results/{dataset_name}_comparison'
    os.makedirs(comparison_dir, exist_ok=True)
    comparison_path = os.path.join(comparison_dir, 'adversarial_vs_non_adversarial_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    # Create comparison visualization
    create_comparison_plots(comparison_df, comparison_dir, dataset_name)
    
    print(f"\n✅ Comparison complete! Results saved to: {comparison_dir}/")
    return comparison_df


def create_comparison_plots(comparison_df: pd.DataFrame, save_dir: str, dataset_name: str):
    """Create adversarial vs non-adversarial comparison plots."""
    print("🎨 Creating adversarial vs non-adversarial comparison plots...")
    
    # Get paired models
    non_adv_df = comparison_df[comparison_df['training_type'] == 'non_adversarial']
    adv_df = comparison_df[comparison_df['training_type'] == 'adversarial']
    
    if non_adv_df.empty or adv_df.empty:
        print("❌ Missing data for comparison")
        return
    
    models = sorted(non_adv_df['model_id'].unique())
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Adversarial vs Non-Adversarial Training Comparison: {dataset_name.upper()}', 
                 fontsize=18, fontweight='bold')
    
    metrics = [
        ('rmse', 'RMSE Comparison', 'RMSE (Lower = Better)'),
        ('ks_statistic', 'KS Statistic Comparison', 'KS Statistic (Lower = Better)'),
        ('wasserstein_distance', 'Wasserstein Distance Comparison', 'Wasserstein Distance (Lower = Better)'),
        ('std_rmse', 'Empirical Std RMSE Comparison', 'Std RMSE (Lower = Better)')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Get values for each model
        non_adv_values = []
        adv_values = []
        
        for model in models:
            non_adv_val = non_adv_df[non_adv_df['model_id'] == model][metric].iloc[0]
            adv_val = adv_df[adv_df['base_model_id'] == model][metric].iloc[0] if len(adv_df[adv_df['base_model_id'] == model]) > 0 else np.nan
            
            non_adv_values.append(non_adv_val)
            adv_values.append(adv_val)
        
        # Create paired bar plot
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, non_adv_values, width, label='Non-Adversarial', 
                       color='skyblue', alpha=0.8, edgecolor='navy')
        bars2 = ax.bar(x + width/2, adv_values, width, label='Adversarial', 
                       color='coral', alpha=0.8, edgecolor='darkred')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars, values in [(bars1, non_adv_values), (bars2, adv_values)]:
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + max(non_adv_values + adv_values) * 0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adversarial_vs_non_adversarial_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plots saved to: {save_dir}/adversarial_vs_non_adversarial_comparison.png")


def main():
    """Main comparison function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare adversarial vs non-adversarial models")
    parser.add_argument("--dataset", type=str, default='ou_process', 
                       help="Dataset to compare models on")
    
    args = parser.parse_args()
    
    # Run comparison
    comparison_df = compare_adversarial_vs_non_adversarial_models(args.dataset)
    
    if comparison_df is not None:
        print(f"\n🎉 Adversarial vs Non-Adversarial Comparison Complete!")
        print(f"   Dataset: {args.dataset}")
        print(f"   Models compared: {len(comparison_df) // 2}")
        print(f"   Results saved to: results/{args.dataset}_comparison/")


if __name__ == "__main__":
    main()

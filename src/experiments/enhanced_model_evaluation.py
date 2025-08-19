"""
Enhanced Model Evaluation with Trajectory Visualization and Empirical Std Analysis

This script provides comprehensive evaluation including:
1. 20 trajectory samples per model for visualization
2. Empirical standard deviation analysis at each time step
3. Clean, sorted visualizations without unnecessary metrics
4. Comparison plots for trajectory and std analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List, Tuple
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.generative_model import get_signal
from utils.model_checkpoint import create_checkpoint_manager


def compute_core_metrics(generated: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
    """Compute core evaluation metrics."""
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle shape alignment
    if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
        gt_np = gt_np[:, 1, :]  # Extract value dimension
    
    # Match dimensions
    min_samples = min(gen_np.shape[0], gt_np.shape[0])
    gen_np = gen_np[:min_samples]
    gt_np = gt_np[:min_samples]
    
    if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:, :min_length]
        gt_np = gt_np[:, :min_length]
    
    # Core metrics
    gen_flat = gen_np.flatten()
    gt_flat = gt_np.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((gen_flat - gt_flat) ** 2))
    
    # KS test for distribution similarity
    ks_statistic, ks_p_value = stats.ks_2samp(gen_flat, gt_flat)
    
    # Wasserstein distance
    wasserstein_dist = stats.wasserstein_distance(gen_flat, gt_flat)
    
    return {
        'rmse': rmse,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'wasserstein_distance': wasserstein_dist,
        'distributions_similar': ks_p_value > 0.05
    }


def compute_empirical_std_analysis(trajectories: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, Any]:
    """
    Compute empirical standard deviation analysis at each time step.
    
    Args:
        trajectories: Generated trajectories, shape (n_traj, time_steps)
        ground_truth: Ground truth data, shape (batch, channels, time_steps)
        
    Returns:
        Dictionary with std analysis results
    """
    traj_np = trajectories.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Extract ground truth values (remove time dimension)
    if len(gt_np.shape) == 3:
        gt_values = gt_np[:, 1, :]  # Shape: (batch, time_steps)
    else:
        gt_values = gt_np
    
    # Compute empirical std at each time step
    gen_std_per_timestep = np.std(traj_np, axis=0)  # Shape: (time_steps,)
    gt_std_per_timestep = np.std(gt_values, axis=0)  # Shape: (time_steps,)
    
    # Align lengths
    min_length = min(len(gen_std_per_timestep), len(gt_std_per_timestep))
    gen_std_per_timestep = gen_std_per_timestep[:min_length]
    gt_std_per_timestep = gt_std_per_timestep[:min_length]
    
    # Compute std matching metrics
    std_rmse = np.sqrt(np.mean((gen_std_per_timestep - gt_std_per_timestep) ** 2))
    std_correlation = np.corrcoef(gen_std_per_timestep, gt_std_per_timestep)[0, 1]
    std_mean_error = np.mean(np.abs(gen_std_per_timestep - gt_std_per_timestep))
    
    return {
        'empirical_std_generated': gen_std_per_timestep,
        'empirical_std_ground_truth': gt_std_per_timestep,
        'std_rmse': std_rmse,
        'std_correlation': std_correlation,
        'std_mean_absolute_error': std_mean_error,
        'time_steps': np.arange(min_length)
    }


def evaluate_all_models_enhanced():
    """Enhanced evaluation of all trained models."""
    print("🎯 Enhanced Model Evaluation with Trajectory Analysis")
    print("=" * 60)
    
    # Setup checkpoint manager
    checkpoint_manager = create_checkpoint_manager('results')
    
    # Get list of trained models
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print("❌ No trained models found")
        return
    
    print(f"Found {len(available_models)} trained models:")
    checkpoint_manager.print_available_models()
    
    # Setup evaluation data
    print("\nSetting up evaluation data...")
    dataset = get_signal(num_samples=64)
    eval_data = torch.stack([dataset[i][0] for i in range(32)])
    print(f"Evaluation data: {eval_data.shape}")
    
    # Evaluate each model
    results = []
    trajectory_data = {}  # Store trajectory data for visualization
    
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
            
            checkpoint_info = checkpoint_manager.get_checkpoint_info(model_id)
            print(f"✅ Model {model_id} loaded successfully")
            print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"Training: Epoch {checkpoint_info['epoch']}, Loss {checkpoint_info['loss']:.6f}")
            
            # Generate standard evaluation samples
            model.eval()
            with torch.no_grad():
                samples = model.generate_samples(64)
                print(f"Generated samples: {samples.shape}")
                
                # Generate 20 trajectories for visualization
                trajectories = model.generate_samples(20)
                print(f"Generated trajectories for visualization: {trajectories.shape}")
            
            # Compute core metrics
            metrics = compute_core_metrics(samples, eval_data)
            
            # Compute empirical std analysis
            std_analysis = compute_empirical_std_analysis(trajectories, eval_data)
            
            # Combine results
            result = {
                'model_id': model_id,
                'training_epoch': checkpoint_info['epoch'],
                'training_loss': checkpoint_info['loss'],
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'model_class': type(model).__name__,
                **metrics,
                'std_rmse': std_analysis['std_rmse'],
                'std_correlation': std_analysis['std_correlation'],
                'std_mean_absolute_error': std_analysis['std_mean_absolute_error']
            }
            
            results.append(result)
            
            # Store trajectory data for visualization
            trajectory_data[model_id] = {
                'trajectories': trajectories.detach().cpu().numpy(),
                'empirical_std_generated': std_analysis['empirical_std_generated'],
                'empirical_std_ground_truth': std_analysis['empirical_std_ground_truth'],
                'time_steps': std_analysis['time_steps']
            }
            
            print(f"✅ {model_id} evaluation completed")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
            print(f"   Std RMSE: {std_analysis['std_rmse']:.4f}")
            print(f"   Std Correlation: {std_analysis['std_correlation']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            continue
    
    if not results:
        print("❌ No models successfully evaluated")
        return
    
    # Save results
    results_df = pd.DataFrame(results)
    save_dir = "results/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'enhanced_models_evaluation.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print("ENHANCED EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Models evaluated: {len(results)}")
    
    # Create enhanced visualizations
    create_enhanced_visualizations(results_df, trajectory_data, save_dir)
    
    return results_df, trajectory_data


def create_enhanced_visualizations(results_df: pd.DataFrame, trajectory_data: Dict, save_dir: str):
    """Create enhanced visualizations with trajectories and empirical std analysis."""
    print("\nCreating enhanced visualizations...")
    
    # Sort models by KS statistic (best distribution matching first)
    sorted_results = results_df.sort_values('ks_statistic').reset_index(drop=True)
    models = sorted_results['model_id'].tolist()
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Model Evaluation: Trajectory and Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. RMSE comparison (sorted)
    ax1 = axes[0, 0]
    rmse_values = [sorted_results[sorted_results['model_id'] == model]['rmse'].iloc[0] for model in models]
    bars1 = ax1.bar(range(len(models)), rmse_values, color='skyblue', alpha=0.7)
    ax1.set_title('RMSE by Model (Sorted by Distribution Quality)', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + max(rmse_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. KS Statistic comparison (sorted)
    ax2 = axes[0, 1]
    ks_values = [sorted_results[sorted_results['model_id'] == model]['ks_statistic'].iloc[0] for model in models]
    bars2 = ax2.bar(range(len(models)), ks_values, color='lightcoral', alpha=0.7)
    ax2.set_title('KS Statistic by Model (Lower = Better Distribution)', fontweight='bold')
    ax2.set_ylabel('KS Statistic')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(ks_values):
        ax2.text(i, v + max(ks_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Wasserstein Distance comparison (sorted)
    ax3 = axes[1, 0]
    wasserstein_values = [sorted_results[sorted_results['model_id'] == model]['wasserstein_distance'].iloc[0] for model in models]
    bars3 = ax3.bar(range(len(models)), wasserstein_values, color='lightgreen', alpha=0.7)
    ax3.set_title('Wasserstein Distance by Model (Lower = Better)', fontweight='bold')
    ax3.set_ylabel('Wasserstein Distance')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(wasserstein_values):
        ax3.text(i, v + max(wasserstein_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Empirical Std RMSE comparison (sorted)
    ax4 = axes[1, 1]
    std_rmse_values = [sorted_results[sorted_results['model_id'] == model]['std_rmse'].iloc[0] for model in models]
    bars4 = ax4.bar(range(len(models)), std_rmse_values, color='gold', alpha=0.7)
    ax4.set_title('Empirical Std RMSE by Model (Lower = Better)', fontweight='bold')
    ax4.set_ylabel('Std RMSE')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(std_rmse_values):
        ax4.text(i, v + max(std_rmse_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Enhanced comparison visualization saved to: {os.path.join(save_dir, 'enhanced_model_comparison.png')}")
    
    # Create trajectory visualization
    create_trajectory_visualization(trajectory_data, save_dir)
    
    # Create empirical std comparison
    create_empirical_std_visualization(trajectory_data, save_dir)


def create_trajectory_visualization(trajectory_data: Dict, save_dir: str):
    """Create ultra-clear trajectory visualization with equal sample counts."""
    print("Creating ultra-clear trajectory visualization...")
    
    # Get ground truth for comparison (exactly 20 samples to match generated)
    dataset = get_signal(num_samples=64)
    ground_truth = torch.stack([dataset[i][0] for i in range(20)])  # Exactly 20 samples
    gt_values = ground_truth[:, 1, :].numpy()  # Extract value dimension
    
    print(f"Ground truth samples: {gt_values.shape[0]} (matching generated sample count)")
    
    # Create figure with subplots for each model
    n_models = len(trajectory_data)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.suptitle('Ultra-Clear Trajectory Analysis: Generated vs Ground Truth (20 samples each)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(trajectory_data.keys())
    time_steps = np.linspace(0, 1, gt_values.shape[1])
    
    for i, model_id in enumerate(model_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create statistical envelopes for better understanding
        gt_mean = np.mean(gt_values, axis=0)
        gt_std = np.std(gt_values, axis=0)
        
        # Plot ground truth envelope
        ax.fill_between(time_steps, gt_mean - gt_std, gt_mean + gt_std, 
                       color='red', alpha=0.15, label='GT ±1σ')
        
        # Plot ground truth mean (very visible)
        ax.plot(time_steps, gt_mean, color='darkred', linewidth=4, 
               label='GT Mean', alpha=0.9, linestyle='-')
        
        # Plot a few individual ground truth trajectories (clearly visible)
        for j in range(min(3, gt_values.shape[0])):
            ax.plot(time_steps, gt_values[j], color='red', alpha=0.7, 
                   linewidth=1.2, linestyle='--')
        
        # Plot generated trajectories
        trajectories = trajectory_data[model_id]['trajectories']
        time_steps_gen = np.linspace(0, 1, trajectories.shape[1])
        
        # Generated envelope
        gen_mean = np.mean(trajectories, axis=0)
        gen_std = np.std(trajectories, axis=0)
        ax.fill_between(time_steps_gen, gen_mean - gen_std, gen_mean + gen_std,
                       color='blue', alpha=0.1, label='Gen ±1σ')
        
        # Generated mean
        ax.plot(time_steps_gen, gen_mean, color='darkblue', linewidth=2.5,
               label='Gen Mean', alpha=0.8)
        
        # Individual generated trajectories (background)
        for j in range(trajectories.shape[0]):
            ax.plot(time_steps_gen, trajectories[j], color='lightblue', alpha=0.3, linewidth=0.6)
        
        ax.set_title(f'{model_id}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add comprehensive legend for first subplot
        if i == 0:
            ax.plot([], [], color='darkred', linewidth=4, alpha=0.9, label='GT Mean')
            ax.plot([], [], color='red', linewidth=1.2, alpha=0.7, linestyle='--', label='GT Samples')
            ax.plot([], [], color='darkblue', linewidth=2.5, alpha=0.8, label='Gen Mean')
            ax.plot([], [], color='lightblue', linewidth=0.6, alpha=0.3, label='Gen Samples')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Hide unused subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ultra_clear_trajectory_visualization.png'), dpi=300, bbox_inches='tight')
    print(f"Ultra-clear trajectory visualization saved to: {os.path.join(save_dir, 'ultra_clear_trajectory_visualization.png')}")
    print(f"   - Equal sample counts: 20 ground truth, 20 generated per model")
    print(f"   - Ground truth clearly visible with dark red colors")
    print(f"   - Statistical envelopes for better interpretation")


def create_empirical_std_visualization(trajectory_data: Dict, save_dir: str):
    """Create empirical standard deviation comparison visualization."""
    print("Creating empirical std comparison...")
    
    # Sort models by std RMSE (best first)
    std_rmse_scores = {model_id: np.sqrt(np.mean((data['empirical_std_generated'] - data['empirical_std_ground_truth']) ** 2))
                      for model_id, data in trajectory_data.items()}
    sorted_models = sorted(std_rmse_scores.keys(), key=lambda x: std_rmse_scores[x])
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Empirical Standard Deviation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Std evolution over time
    ax1 = axes[0]
    
    # Plot ground truth std (thick black line)
    gt_std = trajectory_data[list(trajectory_data.keys())[0]]['empirical_std_ground_truth']
    time_steps = trajectory_data[list(trajectory_data.keys())[0]]['time_steps']
    ax1.plot(time_steps, gt_std, color='black', linewidth=3, label='Ground Truth', alpha=0.8)
    
    # Plot each model's std
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
    for i, model_id in enumerate(sorted_models):
        data = trajectory_data[model_id]
        ax1.plot(data['time_steps'], data['empirical_std_generated'], 
                color=colors[i], linewidth=2, label=f'{model_id}', alpha=0.7)
    
    ax1.set_title('Empirical Standard Deviation Evolution Over Time', fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Empirical Std')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Std RMSE comparison (sorted)
    ax2 = axes[1]
    std_rmse_values = [std_rmse_scores[model_id] for model_id in sorted_models]
    bars = ax2.bar(range(len(sorted_models)), std_rmse_values, color='orange', alpha=0.7)
    ax2.set_title('Empirical Std RMSE by Model (Sorted by Performance)', fontweight='bold')
    ax2.set_ylabel('Std RMSE (Lower = Better)')
    ax2.set_xticks(range(len(sorted_models)))
    ax2.set_xticklabels(sorted_models, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(std_rmse_values):
        ax2.text(i, v + max(std_rmse_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'empirical_std_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Empirical std analysis saved to: {os.path.join(save_dir, 'empirical_std_analysis.png')}")


def main():
    """Main evaluation function."""
    # Run enhanced evaluation
    results_df, trajectory_data = evaluate_all_models_enhanced()
    
    if results_df is not None:
        print(f"\n🎉 ENHANCED EVALUATION COMPLETE!")
        print(f"   All models evaluated with trajectory and std analysis")
        print(f"   Clean visualizations created without unnecessary metrics")
        print(f"   20 trajectories per model for comprehensive analysis")
        print(f"   Empirical std matching analysis included")


if __name__ == "__main__":
    main()

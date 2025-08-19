"""
Multi-Dataset Evaluation System

Train and evaluate all models on multiple datasets:
1. Ornstein-Uhlenbeck Process (original)
2. Heston Stochastic Volatility Model
3. Simplified rBergomi Model
4. Standard Brownian Motion

This provides robust validation across different stochastic processes.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.multi_dataset import MultiDatasetManager
from utils.model_checkpoint import create_checkpoint_manager
from scipy import stats


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
    """Compute empirical standard deviation analysis."""
    traj_np = trajectories.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Extract ground truth values
    if len(gt_np.shape) == 3:
        gt_values = gt_np[:, 1, :]
    else:
        gt_values = gt_np
    
    # Compute empirical std at each time step
    gen_std_per_timestep = np.std(traj_np, axis=0)
    gt_std_per_timestep = np.std(gt_values, axis=0)
    
    # Align lengths
    min_length = min(len(gen_std_per_timestep), len(gt_std_per_timestep))
    gen_std_per_timestep = gen_std_per_timestep[:min_length]
    gt_std_per_timestep = gt_std_per_timestep[:min_length]
    
    # Compute std matching metrics
    std_rmse = np.sqrt(np.mean((gen_std_per_timestep - gt_std_per_timestep) ** 2))
    std_correlation = np.corrcoef(gen_std_per_timestep, gt_std_per_timestep)[0, 1]
    
    return {
        'std_rmse': std_rmse,
        'std_correlation': std_correlation
    }


def train_models_on_dataset(dataset_name: str, dataset_data: torch.utils.data.TensorDataset, 
                           models_to_train: List[str] = None, epochs: int = 20):
    """
    Train specified models on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_data: Dataset to train on
        models_to_train: List of model IDs to train (default: ['B3', 'B4', 'A2'])
        epochs: Number of training epochs
    """
    if models_to_train is None:
        models_to_train = ['B3', 'B4', 'A2']  # Top 3 performers
    
    print(f"🏗️ Training Models on {dataset_name.upper()} Dataset")
    print("=" * 60)
    
    # Setup data
    train_data = torch.stack([dataset_data[i][0] for i in range(min(128, len(dataset_data)))])
    print(f"Training data shape: {train_data.shape}")
    
    # Setup checkpoint manager for this dataset
    checkpoint_dir = f'results/{dataset_name}/trained_models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    
    results = []
    
    for model_id in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_id} on {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Import model creation function
            if model_id == 'B3':
                from models.implementations.b3_nsde_tstatistic import create_b3_model
                create_model_fn = create_b3_model
            elif model_id == 'B4':
                from models.implementations.b4_nsde_mmd import create_b4_model
                create_model_fn = create_b4_model
            elif model_id == 'A2':
                from models.implementations.a2_canned_scoring import create_a2_model
                create_model_fn = create_a2_model
            else:
                print(f"⚠️ Model {model_id} not supported in multi-dataset training")
                continue
            
            # Create model
            model = create_model_fn(train_data, train_data)
            print(f"✅ {model_id} created: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Simple training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            
            best_loss = float('inf')
            best_epoch = 0
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Get random batch
                batch_size = 16
                indices = torch.randperm(train_data.shape[0])[:batch_size]
                batch_data = train_data[indices]
                
                # Training step
                optimizer.zero_grad()
                generated_output = model(batch_data)
                loss = model.compute_loss(generated_output)
                loss.backward()
                optimizer.step()
                
                # Track best loss
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch + 1
                    
                    # Save best model
                    checkpoint_manager.save_model(
                        model=model,
                        model_id=model_id,
                        epoch=epoch + 1,
                        loss=loss.item(),
                        metrics={}
                    )
                
                epoch_time = time.time() - epoch_start
                
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1:2d}: Loss = {loss.item():.6f}, Best = {best_loss:.6f} (epoch {best_epoch}), Time = {epoch_time:.2f}s")
            
            results.append({
                'dataset': dataset_name,
                'model_id': model_id,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'total_epochs': epochs
            })
            
            print(f"✅ {model_id} training completed on {dataset_name}")
            print(f"   Best loss: {best_loss:.6f} at epoch {best_epoch}")
            
        except Exception as e:
            print(f"❌ Failed to train {model_id} on {dataset_name}: {e}")
            continue
    
    return results


def evaluate_models_on_dataset(dataset_name: str, dataset_data: torch.utils.data.TensorDataset):
    """
    Evaluate all trained models on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_data: Dataset to evaluate on
    """
    print(f"📊 Evaluating Models on {dataset_name.upper()} Dataset")
    print("=" * 60)
    
    # Setup evaluation data
    eval_data = torch.stack([dataset_data[i][0] for i in range(min(64, len(dataset_data)))])
    print(f"Evaluation data shape: {eval_data.shape}")
    
    # Setup checkpoint manager
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print(f"❌ No trained models found for {dataset_name}")
        return None
    
    print(f"Found {len(available_models)} trained models: {', '.join(available_models)}")
    
    results = []
    
    for model_id in available_models:
        print(f"\nEvaluating {model_id}...")
        
        try:
            # Load model
            model = checkpoint_manager.load_model(model_id)
            if model is None:
                print(f"❌ Failed to load {model_id}")
                continue
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = model.generate_samples(64)
                trajectories = model.generate_samples(20)
            
            # Compute metrics
            metrics = compute_core_metrics(samples, eval_data)
            std_analysis = compute_empirical_std_analysis(trajectories, eval_data)
            
            # Combine results
            result = {
                'dataset': dataset_name,
                'model_id': model_id,
                **metrics,
                'std_rmse': std_analysis['std_rmse'],
                'std_correlation': std_analysis['std_correlation']
            }
            
            results.append(result)
            
            print(f"✅ {model_id}: RMSE {metrics['rmse']:.4f}, KS {metrics['ks_statistic']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        eval_dir = f'results/{dataset_name}/evaluation'
        os.makedirs(eval_dir, exist_ok=True)
        
        results_path = os.path.join(eval_dir, f'{dataset_name}_evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to: {results_path}")
        
        return results_df
    
    return None


def run_multi_dataset_experiment():
    """Run complete multi-dataset experiment."""
    print("🚀 Multi-Dataset Signature Model Experiment")
    print("=" * 70)
    
    # Initialize dataset manager
    dataset_manager = MultiDatasetManager()
    dataset_manager.list_datasets()
    
    # Get all datasets
    datasets = dataset_manager.get_all_datasets(num_samples=256)
    
    if not datasets:
        print("❌ No datasets available")
        return
    
    print(f"\n✅ Generated {len(datasets)} datasets")
    
    # Models to test (top 3 performers from OU process)
    models_to_test = ['B3', 'B4', 'A2']
    
    all_results = []
    
    # Train and evaluate on each dataset
    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING {dataset_name.upper()} DATASET")
        print(f"{'='*70}")
        
        # Train models
        training_results = train_models_on_dataset(
            dataset_name, dataset_data, models_to_test, epochs=20
        )
        
        # Evaluate models
        eval_results = evaluate_models_on_dataset(dataset_name, dataset_data)
        
        if eval_results is not None:
            all_results.append(eval_results)
    
    # Create cross-dataset comparison
    if all_results:
        create_cross_dataset_comparison(all_results)
    
    print(f"\n🎉 MULTI-DATASET EXPERIMENT COMPLETE!")
    print(f"   Trained and evaluated models on {len(datasets)} datasets")
    print(f"   Results organized by dataset in results/ subdirectories")


def create_cross_dataset_comparison(all_results: List[pd.DataFrame]):
    """Create cross-dataset comparison visualization."""
    print("\n🎨 Creating Cross-Dataset Comparison...")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Create comparison plots
    datasets = combined_df['dataset'].unique()
    models = combined_df['model_id'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Dataset Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. RMSE by dataset
    ax1 = axes[0, 0]
    for model in models:
        model_data = combined_df[combined_df['model_id'] == model]
        rmse_values = [model_data[model_data['dataset'] == ds]['rmse'].iloc[0] if len(model_data[model_data['dataset'] == ds]) > 0 else np.nan for ds in datasets]
        ax1.plot(datasets, rmse_values, marker='o', linewidth=2, label=model, markersize=8)
    
    ax1.set_title('RMSE Across Datasets', fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. KS Statistic by dataset
    ax2 = axes[0, 1]
    for model in models:
        model_data = combined_df[combined_df['model_id'] == model]
        ks_values = [model_data[model_data['dataset'] == ds]['ks_statistic'].iloc[0] if len(model_data[model_data['dataset'] == ds]) > 0 else np.nan for ds in datasets]
        ax2.plot(datasets, ks_values, marker='s', linewidth=2, label=model, markersize=8)
    
    ax2.set_title('KS Statistic Across Datasets (Lower = Better)', fontweight='bold')
    ax2.set_ylabel('KS Statistic')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Std RMSE by dataset
    ax3 = axes[1, 0]
    for model in models:
        model_data = combined_df[combined_df['model_id'] == model]
        std_values = [model_data[model_data['dataset'] == ds]['std_rmse'].iloc[0] if len(model_data[model_data['dataset'] == ds]) > 0 else np.nan for ds in datasets]
        ax3.plot(datasets, std_values, marker='^', linewidth=2, label=model, markersize=8)
    
    ax3.set_title('Empirical Std RMSE Across Datasets', fontweight='bold')
    ax3.set_ylabel('Std RMSE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Overall ranking heatmap
    ax4 = axes[1, 1]
    
    # Create ranking matrix
    ranking_matrix = np.zeros((len(models), len(datasets)))
    
    for i, dataset in enumerate(datasets):
        dataset_results = combined_df[combined_df['dataset'] == dataset]
        if len(dataset_results) > 0:
            # Sort by KS statistic (lower is better)
            sorted_results = dataset_results.sort_values('ks_statistic')
            for j, model in enumerate(models):
                model_rank = sorted_results[sorted_results['model_id'] == model].index
                if len(model_rank) > 0:
                    rank = list(sorted_results['model_id']).index(model) + 1
                    ranking_matrix[j, i] = rank
    
    im = ax4.imshow(ranking_matrix, cmap='RdYlGn_r', aspect='auto')
    ax4.set_title('Model Rankings by Dataset (1=Best)', fontweight='bold')
    ax4.set_xticks(range(len(datasets)))
    ax4.set_xticklabels(datasets, rotation=45)
    ax4.set_yticks(range(len(models)))
    ax4.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(datasets)):
            if ranking_matrix[i, j] > 0:
                text = ax4.text(j, i, f'{int(ranking_matrix[i, j])}',
                               ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    # Save comparison
    comparison_dir = 'results/cross_dataset_comparison'
    os.makedirs(comparison_dir, exist_ok=True)
    plt.savefig(os.path.join(comparison_dir, 'cross_dataset_performance.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Cross-dataset comparison saved to: {comparison_dir}/cross_dataset_performance.png")
    
    # Save combined results
    combined_df.to_csv(os.path.join(comparison_dir, 'all_datasets_results.csv'), index=False)
    print(f"Combined results saved to: {comparison_dir}/all_datasets_results.csv")


def quick_multi_dataset_evaluation():
    """Quick evaluation of top models on all datasets (using existing OU-trained models)."""
    print("⚡ Quick Multi-Dataset Evaluation")
    print("=" * 50)
    
    # Initialize dataset manager
    dataset_manager = MultiDatasetManager()
    
    # Get all datasets (smaller for quick evaluation)
    datasets = dataset_manager.get_all_datasets(num_samples=64)
    
    # Load OU-trained models
    ou_checkpoint_manager = create_checkpoint_manager('results/ou_process')
    available_models = ou_checkpoint_manager.list_available_models()
    
    if not available_models:
        print("❌ No OU-trained models found")
        return
    
    # Test top 3 models on all datasets
    top_models = ['B3', 'B4', 'A2'] if all(m in available_models for m in ['B3', 'B4', 'A2']) else available_models[:3]
    
    all_results = []
    
    for dataset_name, dataset_data in datasets.items():
        print(f"\n📊 Evaluating on {dataset_name.upper()} dataset...")
        
        # Setup evaluation data
        eval_data = torch.stack([dataset_data[i][0] for i in range(32)])
        
        dataset_results = []
        
        for model_id in top_models:
            if model_id not in available_models:
                continue
                
            try:
                # Load OU-trained model
                model = ou_checkpoint_manager.load_model(model_id)
                if model is None:
                    continue
                
                # Evaluate on this dataset
                model.eval()
                with torch.no_grad():
                    samples = model.generate_samples(32)
                    trajectories = model.generate_samples(20)
                
                # Compute metrics
                metrics = compute_core_metrics(samples, eval_data)
                std_analysis = compute_empirical_std_analysis(trajectories, eval_data)
                
                result = {
                    'dataset': dataset_name,
                    'model_id': model_id,
                    'rmse': metrics['rmse'],
                    'ks_statistic': metrics['ks_statistic'],
                    'wasserstein_distance': metrics['wasserstein_distance'],
                    'std_rmse': std_analysis['std_rmse'],
                    'std_correlation': std_analysis['std_correlation']
                }
                
                dataset_results.append(result)
                print(f"   ✅ {model_id}: RMSE {metrics['rmse']:.4f}, KS {metrics['ks_statistic']:.4f}")
                
            except Exception as e:
                print(f"   ❌ {model_id} evaluation failed: {e}")
        
        all_results.extend(dataset_results)
    
    if all_results:
        # Save quick evaluation results
        results_df = pd.DataFrame(all_results)
        quick_dir = 'results/quick_multi_dataset'
        os.makedirs(quick_dir, exist_ok=True)
        
        results_path = os.path.join(quick_dir, 'quick_multi_dataset_results.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\n✅ Quick multi-dataset results saved to: {results_path}")
        
        # Create quick comparison visualization
        create_quick_comparison_viz(results_df, quick_dir)
        
        return results_df
    
    return None


def create_quick_comparison_viz(results_df: pd.DataFrame, save_dir: str):
    """Create quick comparison visualization."""
    print("Creating quick comparison visualization...")
    
    datasets = results_df['dataset'].unique()
    models = results_df['model_id'].unique()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Quick Multi-Dataset Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. RMSE comparison
    ax1 = axes[0]
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model in enumerate(models):
        rmse_values = []
        for dataset in datasets:
            model_data = results_df[(results_df['dataset'] == dataset) & (results_df['model_id'] == model)]
            rmse_values.append(model_data['rmse'].iloc[0] if len(model_data) > 0 else 0)
        
        ax1.bar(x + i * width, rmse_values, width, label=model, alpha=0.8)
    
    ax1.set_title('RMSE by Dataset and Model')
    ax1.set_ylabel('RMSE')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(datasets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. KS Statistic comparison
    ax2 = axes[1]
    
    for i, model in enumerate(models):
        ks_values = []
        for dataset in datasets:
            model_data = results_df[(results_df['dataset'] == dataset) & (results_df['model_id'] == model)]
            ks_values.append(model_data['ks_statistic'].iloc[0] if len(model_data) > 0 else 0)
        
        ax2.bar(x + i * width, ks_values, width, label=model, alpha=0.8)
    
    ax2.set_title('KS Statistic by Dataset and Model (Lower = Better)')
    ax2.set_ylabel('KS Statistic')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(datasets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'quick_multi_dataset_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Quick comparison saved to: {save_dir}/quick_multi_dataset_comparison.png")


def main():
    """Main function - run quick evaluation first."""
    print("Multi-Dataset Signature Model Analysis")
    print("=" * 70)
    
    # Run quick evaluation using existing OU-trained models
    print("Running quick evaluation on all datasets...")
    quick_results = quick_multi_dataset_evaluation()
    
    if quick_results is not None:
        print(f"\n📊 QUICK MULTI-DATASET SUMMARY:")
        print("=" * 50)
        
        # Show summary by dataset
        for dataset in quick_results['dataset'].unique():
            dataset_results = quick_results[quick_results['dataset'] == dataset]
            best_model = dataset_results.loc[dataset_results['ks_statistic'].idxmin()]
            
            print(f"{dataset.upper()}:")
            print(f"   Best model: {best_model['model_id']} (KS: {best_model['ks_statistic']:.4f})")
            print(f"   All models: {', '.join(dataset_results['model_id'].tolist())}")
        
        print(f"\n🎯 Cross-dataset insights available in results/quick_multi_dataset/")


if __name__ == "__main__":
    main()

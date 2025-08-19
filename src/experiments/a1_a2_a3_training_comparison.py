"""
A1 vs A2 vs A3 Training Comparison

This script trains all three CannedNet models with different loss functions:
- A1: CannedNet + T-Statistic + Truncated
- A2: CannedNet + Signature Scoring + Truncated  
- A3: CannedNet + MMD + Truncated

This provides a comprehensive comparison of loss function effects
on the same generator architecture.
"""

import torch
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model

# Import all three models
from models.implementations.a1_final import create_a1_final_model
from models.implementations.a2_canned_scoring import create_a2_model
from models.implementations.a3_canned_mmd import create_a3_model


def compute_evaluation_metrics(generated: torch.Tensor, ground_truth: torch.Tensor, model_name: str):
    """Compute comprehensive evaluation metrics."""
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle shape mismatch
    if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
        gt_np = gt_np[:, 1, :]  # Extract value dimension
    
    # Match samples
    min_samples = min(gen_np.shape[0], gt_np.shape[0])
    gen_np = gen_np[:min_samples]
    gt_np = gt_np[:min_samples]
    
    # Match length
    if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:, :min_length]
        gt_np = gt_np[:, :min_length]
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(gen_np.flatten(), gt_np.flatten()))
    
    gen_flat = gen_np.flatten()
    gt_flat = gt_np.flatten()
    ks_stat, ks_p = stats.ks_2samp(gen_flat, gt_flat)
    wasserstein = stats.wasserstein_distance(gen_flat, gt_flat)
    
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
    
    return {
        'model_name': model_name,
        'rmse': float(rmse),
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'wasserstein_distance': float(wasserstein),
        'mean_path_correlation': float(mean_corr),
        'generated_mean': float(np.mean(gen_flat)),
        'generated_std': float(np.std(gen_flat)),
        'ground_truth_mean': float(np.mean(gt_flat)),
        'ground_truth_std': float(np.std(gt_flat))
    }


def train_model(model, train_loader, optimizer, num_epochs, model_name):
    """Train a model and return training history."""
    print(f"\nTraining {model_name} for {num_epochs} epochs...")
    
    model.train()
    training_losses = []
    epoch_times = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        for batch_idx, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = model.compute_loss(output)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        epoch_time = time.time() - epoch_start
        epoch_loss = np.mean(epoch_losses)
        
        training_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}: Loss = {epoch_loss:.6f}, Time = {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"✅ {model_name} training completed in {total_time:.2f}s")
    print(f"   Final loss: {training_losses[-1]:.6f}")
    
    return {
        'losses': training_losses,
        'times': epoch_times,
        'final_loss': training_losses[-1],
        'total_time': total_time,
        'mean_epoch_time': np.mean(epoch_times)
    }


def create_training_visualization(histories: Dict, save_dir: str):
    """Create comprehensive training visualization for all models."""
    print(f"Creating training visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    models = list(histories.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(models)]
    epochs = range(1, len(histories[models[0]]['losses']) + 1)
    
    # 1. Loss curves
    ax = axes[0, 0]
    for i, (model, history) in enumerate(histories.items()):
        ax.plot(epochs, history['losses'], color=colors[i], label=model, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Loss convergence (last 20 epochs)
    ax = axes[0, 1]
    last_n = min(20, len(epochs))
    for i, (model, history) in enumerate(histories.items()):
        ax.plot(epochs[-last_n:], history['losses'][-last_n:], 
                color=colors[i], label=model, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Convergence (Last {last_n} Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training time per epoch
    ax = axes[0, 2]
    for i, (model, history) in enumerate(histories.items()):
        ax.plot(epochs, history['times'], color=colors[i], alpha=0.7, label=model)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Training Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final loss comparison
    ax = axes[1, 0]
    final_losses = [hist['final_loss'] for hist in histories.values()]
    bars = ax.bar(models, final_losses, color=colors, alpha=0.7)
    ax.set_ylabel('Final Loss')
    ax.set_title('Final Training Loss')
    ax.grid(True, alpha=0.3)
    
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Training efficiency
    ax = axes[1, 1]
    total_times = [hist['total_time'] for hist in histories.values()]
    bars = ax.bar(models, total_times, color=colors, alpha=0.7)
    ax.set_ylabel('Total Training Time (s)')
    ax.set_title('Training Efficiency')
    ax.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars, total_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Training Summary\n" + "="*16 + "\n\n"
    
    for model, history in histories.items():
        loss_type = model.split('_')[-1] if '_' in model else model
        summary_text += f"{loss_type}:\n"
        summary_text += f"  Final Loss: {history['final_loss']:.6f}\n"
        summary_text += f"  Time: {history['total_time']:.1f}s\n"
        summary_text += f"  Speed: {history['mean_epoch_time']:.2f}s/epoch\n\n"
    
    # Determine best performers
    best_loss_model = min(histories.keys(), key=lambda k: histories[k]['final_loss'])
    fastest_model = min(histories.keys(), key=lambda k: histories[k]['total_time'])
    
    summary_text += f"Best Loss: {best_loss_model}\n"
    summary_text += f"Fastest: {fastest_model}\n\n"
    summary_text += f"Status: ✅ TRAINED\n"
    summary_text += f"All {len(models)} models compared"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'a1_a2_a3_training_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Training visualization saved to: {save_dir}/a1_a2_a3_training_comparison.png")
    plt.close()


def create_performance_comparison(results: List[Dict], save_dir: str):
    """Create post-training performance comparison."""
    print(f"Creating performance comparison...")
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    models = df['experiment_id'].tolist()
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(models)]
    
    # 1. RMSE comparison
    ax = axes[0, 0]
    rmse_vals = df['rmse'].tolist()
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Generation Quality (RMSE)\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribution similarity
    ax = axes[0, 1]
    ks_vals = df['ks_statistic'].tolist()
    bars = ax.bar(models, ks_vals, color=colors, alpha=0.7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Distribution Similarity\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, ks_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Path correlation
    ax = axes[0, 2]
    corr_vals = df['mean_path_correlation'].tolist()
    bars = ax.bar(models, corr_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Path Correlation')
    ax.set_title('Path-wise Correlation\n(Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, corr_vals):
        height = bar.get_height()
        y_pos = height + (0.01 if height >= 0 else -0.01)
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}', ha='center', va=va, fontweight='bold')
    
    # 4. Wasserstein distance
    ax = axes[1, 0]
    wass_vals = df['wasserstein_distance'].tolist()
    bars = ax.bar(models, wass_vals, color=colors, alpha=0.7)
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Distribution Distance\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, wass_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Statistical moments
    ax = axes[1, 1]
    gen_means = df['generated_mean'].tolist()
    gt_means = df['ground_truth_mean'].tolist()
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gen_means, width, label='Generated', alpha=0.8, color=colors)
    bars2 = ax.bar(x + width/2, gt_means, width, label='Ground Truth', alpha=0.8, color='gray')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Mean Value')
    ax.set_title('Sample Mean Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Winner summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Determine winners for each metric
    rmse_winner = models[rmse_vals.index(min(rmse_vals))]
    ks_winner = models[ks_vals.index(min(ks_vals))]
    corr_winner = models[corr_vals.index(max(corr_vals))]
    wass_winner = models[wass_vals.index(min(wass_vals))]
    
    # Count wins for each model
    win_counts = {}
    for model in models:
        wins = sum([
            model == rmse_winner,
            model == ks_winner, 
            model == corr_winner,
            model == wass_winner
        ])
        win_counts[model] = wins
    
    overall_winner = max(win_counts.keys(), key=lambda k: win_counts[k])
    
    summary_text = f"""Post-Training Results
{'='*20}

Metric Winners:
  RMSE: {rmse_winner}
  KS Test: {ks_winner}
  Correlation: {corr_winner}
  Wasserstein: {wass_winner}

Win Counts:
"""
    
    for model in models:
        summary_text += f"  {model}: {win_counts[model]}/4\n"
    
    summary_text += f"""
🏆 Overall Winner: {overall_winner}

Key Insights:
  • Loss function matters!
  • Same architecture, different results
  • Training reveals distinctions

Status: ✅ COMPLETE
Ready for analysis!
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'a1_a2_a3_performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved to: {save_dir}/a1_a2_a3_performance_comparison.png")
    plt.close()


def run_comprehensive_training_experiment():
    """Run comprehensive training experiment with A1, A2, and A3."""
    print("A1 vs A2 vs A3 Comprehensive Training Comparison")
    print("=" * 60)
    print("Training all three CannedNet models with different loss functions")
    print("A1: T-Statistic, A2: Signature Scoring, A3: MMD")
    
    # Setup
    os.makedirs('results/training', exist_ok=True)
    
    # Data preparation
    print(f"\nPreparing data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 256
    n_points = 100
    batch_size = 32
    
    # Training data
    train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Test data
    signals = generative_model.get_signal(num_samples=64, n_points=n_points).tensors[0]
    example_batch, _ = next(iter(torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
    
    print(f"Training: {len(train_dataset)} samples, batch size {batch_size}")
    print(f"Test data: {signals.shape}")
    
    # Create models with same initialization
    print(f"\nCreating models...")
    torch.manual_seed(12345)
    a1_model = create_a1_final_model(example_batch, signals)
    
    torch.manual_seed(12345)
    a2_model = create_a2_model(example_batch, signals)
    
    torch.manual_seed(12345)
    a3_model = create_a3_model(example_batch, signals)
    
    print(f"A1 parameters: {sum(p.numel() for p in a1_model.parameters()):,}")
    print(f"A2 parameters: {sum(p.numel() for p in a2_model.parameters()):,}")
    print(f"A3 parameters: {sum(p.numel() for p in a3_model.parameters()):,}")
    
    # Setup optimizers
    lr = 0.001
    a1_optimizer = optim.Adam(a1_model.parameters(), lr=lr)
    a2_optimizer = optim.Adam(a2_model.parameters(), lr=lr)
    a3_optimizer = optim.Adam(a3_model.parameters(), lr=lr)
    
    # Training
    num_epochs = 80
    print(f"\nStarting training ({num_epochs} epochs, lr={lr})...")
    
    training_histories = {}
    training_histories['A1'] = train_model(a1_model, train_loader, a1_optimizer, num_epochs, "A1 (T-Statistic)")
    training_histories['A2'] = train_model(a2_model, train_loader, a2_optimizer, num_epochs, "A2 (Signature Scoring)")
    training_histories['A3'] = train_model(a3_model, train_loader, a3_optimizer, num_epochs, "A3 (MMD)")
    
    # Create training visualization
    create_training_visualization(training_histories, 'results/training')
    
    # Evaluate trained models
    print(f"\nEvaluating trained models...")
    
    for model in [a1_model, a2_model, a3_model]:
        model.eval()
    
    results = []
    
    # Evaluate each model
    models_info = [
        (a1_model, 'A1', 'T-Statistic'),
        (a2_model, 'A2', 'Signature_Scoring'),
        (a3_model, 'A3', 'MMD')
    ]
    
    for model, exp_id, loss_type in models_info:
        print(f"  Evaluating {exp_id}...")
        
        with torch.no_grad():
            samples = model.generate_samples(64)
        
        metrics = compute_evaluation_metrics(samples, signals, f"{exp_id}_Trained")
        metrics.update({
            'experiment_id': exp_id,
            'loss_type': loss_type,
            'final_training_loss': training_histories[exp_id]['final_loss'],
            'total_training_time': training_histories[exp_id]['total_time']
        })
        
        results.append(metrics)
    
    # Create performance comparison
    create_performance_comparison(results, 'results/training')
    
    # Save comprehensive results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/training/a1_a2_a3_training_results.csv', index=False)
    
    # Save training histories
    training_df_data = []
    for model_name, history in training_histories.items():
        for epoch, (loss, time_val) in enumerate(zip(history['losses'], history['times'])):
            training_df_data.append({
                'model': model_name,
                'epoch': epoch + 1,
                'loss': loss,
                'time': time_val
            })
    
    training_df = pd.DataFrame(training_df_data)
    training_df.to_csv('results/training/a1_a2_a3_training_dynamics.csv', index=False)
    
    # Print final summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE TRAINING COMPARISON COMPLETE")
    print("="*60)
    
    print(f"\nTraining Results:")
    for model_name, history in training_histories.items():
        print(f"  {model_name}: Final loss = {history['final_loss']:.6f}, Time = {history['total_time']:.1f}s")
    
    print(f"\nGeneration Quality (RMSE):")
    for result in results:
        exp_id = result['experiment_id']
        rmse = result['rmse']
        corr = result['mean_path_correlation']
        print(f"  {exp_id}: RMSE = {rmse:.4f}, Correlation = {corr:.4f}")
    
    # Overall assessment
    best_training = min(training_histories.keys(), key=lambda k: training_histories[k]['final_loss'])
    best_generation = min(results, key=lambda r: r['rmse'])
    best_correlation = max(results, key=lambda r: r['mean_path_correlation'])
    
    print(f"\nOverall Assessment:")
    print(f"  🏆 Best Training Loss: {best_training}")
    print(f"  🏆 Best Generation RMSE: {best_generation['experiment_id']}")
    print(f"  🏆 Best Path Correlation: {best_correlation['experiment_id']}")
    
    # Check for significant differences
    rmse_values = [r['rmse'] for r in results]
    rmse_range = max(rmse_values) - min(rmse_values)
    
    if rmse_range > 0.1:
        print(f"\n✅ SIGNIFICANT DIFFERENCES FOUND!")
        print(f"   RMSE range: {rmse_range:.4f}")
        print(f"   Loss functions show distinct effects!")
    else:
        print(f"\n⚠️ Similar performance across models")
        print(f"   RMSE range: {rmse_range:.4f}")
        print(f"   May need longer training or different hyperparameters")
    
    print(f"\nFiles generated:")
    print(f"  • results/training/a1_a2_a3_training_results.csv")
    print(f"  • results/training/a1_a2_a3_training_dynamics.csv")
    print(f"  • results/training/a1_a2_a3_training_comparison.png")
    print(f"  • results/training/a1_a2_a3_performance_comparison.png")
    
    return results_df, training_histories


if __name__ == "__main__":
    try:
        results_df, training_histories = run_comprehensive_training_experiment()
        print(f"\n🎉 Comprehensive training comparison successful!")
        print(f"   All three loss functions compared systematically!")
        print(f"   A3 (MMD) now part of the systematic evaluation!")
        
    except Exception as e:
        print(f"\n❌ Comprehensive training comparison failed: {e}")
        import traceback
        traceback.print_exc()

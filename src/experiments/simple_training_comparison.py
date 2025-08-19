"""
Simple Training Comparison: A1 vs A2

This script trains both A1 and A2 models and compares their performance
after optimization to see the real effects of different loss functions.
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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model

# Import models directly
from models.implementations.a1_final import create_a1_final_model
from models.implementations.a2_canned_scoring import create_a2_model


def compute_simple_metrics(generated: torch.Tensor, ground_truth: torch.Tensor, model_name: str):
    """Compute key metrics for comparison."""
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


def train_model_simple(model, train_loader, optimizer, num_epochs, model_name):
    """Simple training loop with progress tracking."""
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
        
        # Print progress every 20 epochs
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


def create_training_plots(a1_history, a2_history, save_dir):
    """Create training comparison plots."""
    print(f"Creating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(a1_history['losses']) + 1)
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, a1_history['losses'], 'b-', label='A1 (T-Statistic)', linewidth=2)
    ax.plot(epochs, a2_history['losses'], 'r-', label='A2 (Signature Scoring)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Last 20 epochs (convergence)
    ax = axes[0, 1]
    last_n = min(20, len(epochs))
    ax.plot(epochs[-last_n:], a1_history['losses'][-last_n:], 'b-', label='A1', linewidth=2)
    ax.plot(epochs[-last_n:], a2_history['losses'][-last_n:], 'r-', label='A2', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Convergence (Last {last_n} Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training time per epoch
    ax = axes[1, 0]
    ax.plot(epochs, a1_history['times'], 'b-', alpha=0.7, label='A1')
    ax.plot(epochs, a2_history['times'], 'r-', alpha=0.7, label='A2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Training Speed Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""Training Summary
{'='*16}

Final Loss:
  A1: {a1_history['final_loss']:.6f}
  A2: {a2_history['final_loss']:.6f}
  
Winner: {'A1' if a1_history['final_loss'] < a2_history['final_loss'] else 'A2'}
Difference: {abs(a1_history['final_loss'] - a2_history['final_loss']):.6f}

Training Time:
  A1: {a1_history['total_time']:.1f}s
  A2: {a2_history['total_time']:.1f}s
  
Speed:
  A1: {a1_history['mean_epoch_time']:.2f}s/epoch
  A2: {a2_history['mean_epoch_time']:.2f}s/epoch
  
Faster: {'A1' if a1_history['mean_epoch_time'] < a2_history['mean_epoch_time'] else 'A2'}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Training plots saved to: {save_dir}/training_comparison.png")
    plt.close()


def create_performance_comparison(results, save_dir):
    """Create post-training performance comparison."""
    print(f"Creating performance comparison...")
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = ['A1', 'A2']
    colors = ['blue', 'red']
    
    # Extract values
    rmse_vals = [df[df['model_name'].str.contains('A1')]['rmse'].iloc[0],
                 df[df['model_name'].str.contains('A2')]['rmse'].iloc[0]]
    
    ks_vals = [df[df['model_name'].str.contains('A1')]['ks_statistic'].iloc[0],
               df[df['model_name'].str.contains('A2')]['ks_statistic'].iloc[0]]
    
    corr_vals = [df[df['model_name'].str.contains('A1')]['mean_path_correlation'].iloc[0],
                 df[df['model_name'].str.contains('A2')]['mean_path_correlation'].iloc[0]]
    
    wass_vals = [df[df['model_name'].str.contains('A1')]['wasserstein_distance'].iloc[0],
                 df[df['model_name'].str.contains('A2')]['wasserstein_distance'].iloc[0]]
    
    # 1. RMSE
    ax = axes[0, 0]
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Square Error\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. KS Statistic
    ax = axes[0, 1]
    bars = ax.bar(models, ks_vals, color=colors, alpha=0.7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Distribution Similarity\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, ks_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Path Correlation
    ax = axes[0, 2]
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
    
    # 4. Wasserstein Distance
    ax = axes[1, 0]
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
    
    a1_data = df[df['model_name'].str.contains('A1')].iloc[0]
    a2_data = df[df['model_name'].str.contains('A2')].iloc[0]
    
    gen_means = [a1_data['generated_mean'], a2_data['generated_mean']]
    gt_means = [a1_data['ground_truth_mean'], a2_data['ground_truth_mean']]
    
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
    
    # Determine winners
    rmse_winner = "A1" if rmse_vals[0] < rmse_vals[1] else "A2"
    ks_winner = "A1" if ks_vals[0] < ks_vals[1] else "A2"
    corr_winner = "A1" if corr_vals[0] > corr_vals[1] else "A2"
    wass_winner = "A1" if wass_vals[0] < wass_vals[1] else "A2"
    
    # Count wins
    a1_wins = sum([rmse_winner == 'A1', ks_winner == 'A1', corr_winner == 'A1', wass_winner == 'A1'])
    a2_wins = 4 - a1_wins
    
    summary_text = f"""Post-Training Results
{'='*18}

Metric Winners:
  RMSE: {rmse_winner}
  KS Statistic: {ks_winner}
  Correlation: {corr_winner}
  Wasserstein: {wass_winner}

Overall Score:
  A1 wins: {a1_wins}/4
  A2 wins: {a2_wins}/4

Champion:
  {'🏆 A1 (T-Statistic)' if a1_wins > a2_wins else '🏆 A2 (Signature Scoring)' if a2_wins > a1_wins else '🤝 Tie'}

Key Differences:
  RMSE: {abs(rmse_vals[0] - rmse_vals[1]):.6f}
  Correlation: {abs(corr_vals[0] - corr_vals[1]):.6f}

Status: ✅ TRAINED
Loss effects revealed!
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Performance comparison saved to: {save_dir}/performance_comparison.png")
    plt.close()


def run_training_experiment():
    """Run the complete training experiment."""
    print("A1 vs A2 Training Comparison")
    print("=" * 40)
    print("Training both models to see loss function effects")
    
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
    
    # Create models
    print(f"\nCreating models...")
    torch.manual_seed(12345)  # Same initialization for fair comparison
    a1_model = create_a1_final_model(example_batch, signals)
    
    torch.manual_seed(12345)  # Same initialization
    a2_model = create_a2_model(example_batch, signals)
    
    print(f"A1 parameters: {sum(p.numel() for p in a1_model.parameters()):,}")
    print(f"A2 parameters: {sum(p.numel() for p in a2_model.parameters()):,}")
    
    # Setup optimizers
    lr = 0.001
    a1_optimizer = optim.Adam(a1_model.parameters(), lr=lr)
    a2_optimizer = optim.Adam(a2_model.parameters(), lr=lr)
    
    # Training
    num_epochs = 80
    print(f"\nStarting training ({num_epochs} epochs, lr={lr})...")
    
    a1_history = train_model_simple(a1_model, train_loader, a1_optimizer, num_epochs, "A1")
    a2_history = train_model_simple(a2_model, train_loader, a2_optimizer, num_epochs, "A2")
    
    # Create training plots
    create_training_plots(a1_history, a2_history, 'results/training')
    
    # Evaluate trained models
    print(f"\nEvaluating trained models...")
    
    a1_model.eval()
    a2_model.eval()
    
    with torch.no_grad():
        a1_samples = a1_model.generate_samples(64)
        a2_samples = a2_model.generate_samples(64)
    
    print(f"A1 generated: {a1_samples.shape}")
    print(f"A2 generated: {a2_samples.shape}")
    
    # Compute metrics
    a1_metrics = compute_simple_metrics(a1_samples, signals, "A1_Trained")
    a2_metrics = compute_simple_metrics(a2_samples, signals, "A2_Trained")
    
    # Add training info
    a1_metrics.update({
        'final_training_loss': a1_history['final_loss'],
        'total_training_time': a1_history['total_time'],
        'loss_type': 'T-Statistic'
    })
    
    a2_metrics.update({
        'final_training_loss': a2_history['final_loss'],
        'total_training_time': a2_history['total_time'],
        'loss_type': 'Signature_Scoring'
    })
    
    results = [a1_metrics, a2_metrics]
    
    # Create performance comparison
    create_performance_comparison(results, 'results/training')
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/training/training_experiment_results.csv', index=False)
    
    # Final summary
    print(f"\n" + "="*50)
    print("TRAINING EXPERIMENT COMPLETE")
    print("="*50)
    
    print(f"Training Results:")
    print(f"  A1 final loss: {a1_history['final_loss']:.6f}")
    print(f"  A2 final loss: {a2_history['final_loss']:.6f}")
    print(f"  Training winner: {'A1' if a1_history['final_loss'] < a2_history['final_loss'] else 'A2'}")
    
    print(f"\nGeneration Quality:")
    print(f"  A1 RMSE: {a1_metrics['rmse']:.4f}")
    print(f"  A2 RMSE: {a2_metrics['rmse']:.4f}")
    print(f"  Generation winner: {'A1' if a1_metrics['rmse'] < a2_metrics['rmse'] else 'A2'}")
    
    print(f"\nPath Correlation:")
    print(f"  A1: {a1_metrics['mean_path_correlation']:.4f}")
    print(f"  A2: {a2_metrics['mean_path_correlation']:.4f}")
    print(f"  Correlation winner: {'A1' if a1_metrics['mean_path_correlation'] > a2_metrics['mean_path_correlation'] else 'A2'}")
    
    # Overall assessment
    a1_wins = 0
    a2_wins = 0
    
    if a1_history['final_loss'] < a2_history['final_loss']:
        a1_wins += 1
    else:
        a2_wins += 1
        
    if a1_metrics['rmse'] < a2_metrics['rmse']:
        a1_wins += 1
    else:
        a2_wins += 1
        
    if a1_metrics['mean_path_correlation'] > a2_metrics['mean_path_correlation']:
        a1_wins += 1
    else:
        a2_wins += 1
    
    print(f"\nOverall Assessment:")
    print(f"  A1 wins: {a1_wins}/3 categories")
    print(f"  A2 wins: {a2_wins}/3 categories")
    
    if a1_wins > a2_wins:
        print(f"  🏆 A1 (T-Statistic) is the overall winner!")
    elif a2_wins > a1_wins:
        print(f"  🏆 A2 (Signature Scoring) is the overall winner!")
    else:
        print(f"  🤝 Both models perform similarly overall")
    
    # Key insight
    rmse_diff = abs(a1_metrics['rmse'] - a2_metrics['rmse'])
    loss_diff = abs(a1_history['final_loss'] - a2_history['final_loss'])
    
    if rmse_diff > 0.01 or loss_diff > 0.1:
        print(f"\n✅ SUCCESS: Different loss functions show distinct effects!")
        print(f"   RMSE difference: {rmse_diff:.4f}")
        print(f"   Training loss difference: {loss_diff:.4f}")
    else:
        print(f"\n⚠️ Similar performance - may need longer training or different hyperparameters")
    
    print(f"\nFiles generated:")
    print(f"  • results/training/training_experiment_results.csv")
    print(f"  • results/training/training_comparison.png")
    print(f"  • results/training/performance_comparison.png")
    
    return results_df, a1_history, a2_history


if __name__ == "__main__":
    try:
        results_df, a1_hist, a2_hist = run_training_experiment()
        print(f"\n🎉 Training comparison successful!")
        print(f"   Loss function effects revealed through training!")
        
    except Exception as e:
        print(f"\n❌ Training comparison failed: {e}")
        import traceback
        traceback.print_exc()

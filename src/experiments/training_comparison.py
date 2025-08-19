"""
Training Comparison: A1 vs A2 Post-Training Evaluation

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
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from experiments.simple_model_evaluation import compute_evaluation_metrics, evaluate_model_performance

# Import models
from models.implementations.a1_final import create_a1_final_model
from models.implementations.a2_canned_scoring import create_a2_model


class TrainingLogger:
    """Logger for tracking training progress."""
    
    def __init__(self):
        self.history = {
            'epoch': [],
            'loss': [],
            'time': []
        }
    
    def log(self, epoch: int, loss: float, time: float):
        """Log training step."""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['time'].append(time)
    
    def get_history(self) -> Dict[str, List]:
        """Get training history."""
        return self.history.copy()


def train_model(model, train_loader, optimizer, num_epochs: int = 50, 
                model_name: str = "Model") -> TrainingLogger:
    """
    Train a model and return training history.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        num_epochs: Number of training epochs
        model_name: Name for logging
        
    Returns:
        TrainingLogger with training history
    """
    print(f"\nTraining {model_name}...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    logger = TrainingLogger()
    model.train()
    
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
        
        logger.log(epoch + 1, epoch_loss, epoch_time)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs}: Loss = {epoch_loss:.6f}, Time = {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"✅ {model_name} training completed in {total_time:.2f}s")
    
    return logger


def compare_training_dynamics(a1_history: Dict, a2_history: Dict, save_dir: str):
    """Compare training dynamics between A1 and A2."""
    print(f"\nAnalyzing training dynamics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(a1_history['epoch'], a1_history['loss'], 'b-', label='A1 (T-Statistic)', linewidth=2)
    ax.plot(a2_history['epoch'], a2_history['loss'], 'r-', label='A2 (Signature Scoring)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for better visualization
    
    # 2. Loss convergence (last 20 epochs)
    ax = axes[0, 1]
    last_n = min(20, len(a1_history['epoch']))
    ax.plot(a1_history['epoch'][-last_n:], a1_history['loss'][-last_n:], 'b-', label='A1', linewidth=2)
    ax.plot(a2_history['epoch'][-last_n:], a2_history['loss'][-last_n:], 'r-', label='A2', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Convergence (Last {last_n} Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training time per epoch
    ax = axes[1, 0]
    ax.plot(a1_history['epoch'], a1_history['time'], 'b-', label='A1', alpha=0.7)
    ax.plot(a2_history['epoch'], a2_history['time'], 'r-', label='A2', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Training Time per Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary stats
    a1_final_loss = a1_history['loss'][-1]
    a2_final_loss = a2_history['loss'][-1]
    a1_mean_time = np.mean(a1_history['time'])
    a2_mean_time = np.mean(a2_history['time'])
    a1_total_time = sum(a1_history['time'])
    a2_total_time = sum(a2_history['time'])
    
    summary_text = f"""Training Summary
{'='*20}

Final Loss:
  A1: {a1_final_loss:.6f}
  A2: {a2_final_loss:.6f}
  
Convergence:
  A1 Better: {'✅' if a1_final_loss < a2_final_loss else '❌'}
  A2 Better: {'✅' if a2_final_loss < a1_final_loss else '❌'}
  Difference: {abs(a1_final_loss - a2_final_loss):.6f}

Training Speed:
  A1 avg: {a1_mean_time:.2f}s/epoch
  A2 avg: {a2_mean_time:.2f}s/epoch
  
Total Time:
  A1: {a1_total_time:.1f}s
  A2: {a2_total_time:.1f}s
  
Efficiency:
  Faster: {'A1' if a1_mean_time < a2_mean_time else 'A2'}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_dynamics_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Training dynamics plot saved to: {save_dir}/training_dynamics_comparison.png")
    plt.close()
    
    # Return summary stats
    return {
        'a1_final_loss': a1_final_loss,
        'a2_final_loss': a2_final_loss,
        'a1_mean_time_per_epoch': a1_mean_time,
        'a2_mean_time_per_epoch': a2_mean_time,
        'a1_total_training_time': a1_total_time,
        'a2_total_training_time': a2_total_time,
        'better_final_loss': 'A1' if a1_final_loss < a2_final_loss else 'A2',
        'loss_difference': abs(a1_final_loss - a2_final_loss)
    }


def evaluate_trained_models(a1_model, a2_model, test_data, ground_truth):
    """Evaluate both trained models and compare performance."""
    print(f"\nEvaluating trained models...")
    
    # Set models to eval mode
    a1_model.eval()
    a2_model.eval()
    
    results = []
    
    # Evaluate A1
    print(f"  Evaluating trained A1...")
    with torch.no_grad():
        a1_samples = a1_model.generate_samples(50)  # More samples for better statistics
    
    a1_metrics = compute_evaluation_metrics(a1_samples, ground_truth[:50], "A1_Trained")
    a1_performance = evaluate_model_performance(a1_model, "A1_Trained", test_data[:8])
    a1_metrics.update(a1_performance)
    a1_metrics.update({
        'model_type': 'A1_Trained',
        'generator_type': 'CannedNet',
        'loss_type': 'T-Statistic',
        'training_status': 'trained'
    })
    results.append(a1_metrics)
    
    # Evaluate A2
    print(f"  Evaluating trained A2...")
    with torch.no_grad():
        a2_samples = a2_model.generate_samples(50)
    
    a2_metrics = compute_evaluation_metrics(a2_samples, ground_truth[:50], "A2_Trained")
    a2_performance = evaluate_model_performance(a2_model, "A2_Trained", test_data[:8])
    a2_metrics.update(a2_performance)
    a2_metrics.update({
        'model_type': 'A2_Trained',
        'generator_type': 'CannedNet',
        'loss_type': 'Signature_Scoring',
        'training_status': 'trained'
    })
    results.append(a2_metrics)
    
    return results


def create_post_training_comparison(results: List[Dict], training_summary: Dict, save_dir: str):
    """Create comprehensive post-training comparison."""
    print(f"\nCreating post-training comparison...")
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = df['model_type'].tolist()
    colors = ['blue', 'red']
    
    # 1. RMSE Comparison
    ax = axes[0, 0]
    rmse_values = df['rmse'].tolist()
    bars = ax.bar(models, rmse_values, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Post-Training RMSE\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribution Similarity
    ax = axes[0, 1]
    ks_values = df['ks_statistic'].tolist()
    bars = ax.bar(models, ks_values, color=colors, alpha=0.7)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Distribution Similarity\n(Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, ks_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Path Correlation
    ax = axes[0, 2]
    corr_values = df['mean_path_correlation'].tolist()
    bars = ax.bar(models, corr_values, color=colors, alpha=0.7)
    ax.set_ylabel('Mean Path Correlation')
    ax.set_title('Path-wise Correlation\n(Higher is Better)')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, corr_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 4. Statistical Moments
    ax = axes[1, 0]
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
    ax.set_xticklabels(['A1', 'A2'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Training vs Generation Performance
    ax = axes[1, 1]
    
    # Training loss vs generation quality
    training_losses = [training_summary['a1_final_loss'], training_summary['a2_final_loss']]
    generation_rmse = rmse_values
    
    ax.scatter(training_losses, generation_rmse, c=colors, s=100, alpha=0.7)
    for i, model in enumerate(['A1', 'A2']):
        ax.annotate(model, (training_losses[i], generation_rmse[i]), 
                   xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Generation RMSE')
    ax.set_title('Training Loss vs Generation Quality')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Assessment
    ax = axes[1, 2]
    ax.axis('off')
    
    # Determine winner in each category
    a1_rmse, a2_rmse = rmse_values
    a1_corr, a2_corr = corr_values
    a1_ks, a2_ks = ks_values
    
    rmse_winner = "A1" if a1_rmse < a2_rmse else "A2"
    corr_winner = "A1" if a1_corr > a2_corr else "A2"
    ks_winner = "A1" if a1_ks < a2_ks else "A2"
    training_winner = training_summary['better_final_loss']
    
    summary_text = f"""Post-Training Results
{'='*20}

Generation Quality:
  RMSE Winner: {rmse_winner}
  Correlation Winner: {corr_winner}
  Distribution Winner: {ks_winner}

Training Performance:
  Loss Winner: {training_winner}
  Loss Diff: {training_summary['loss_difference']:.6f}

Overall Assessment:
  A1 Wins: {sum([rmse_winner == 'A1', corr_winner == 'A1', ks_winner == 'A1'])}/3
  A2 Wins: {sum([rmse_winner == 'A2', corr_winner == 'A2', ks_winner == 'A2'])}/3

Key Insight:
  {'Different losses show' if rmse_winner != corr_winner else 'Similar performance'}
  {'distinct effects!' if rmse_winner != corr_winner else 'across metrics.'}

Status: ✅ TRAINED
Ready for analysis!
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'post_training_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Post-training comparison saved to: {save_dir}/post_training_comparison.png")
    plt.close()


def run_training_comparison():
    """Run complete training comparison between A1 and A2."""
    print("Training Comparison: A1 vs A2")
    print("=" * 50)
    print("This will train both models and compare their post-training performance")
    print("Expected: Different loss functions will lead to different learned behaviors")
    
    # Setup
    os.makedirs('training_results', exist_ok=True)
    
    # Prepare data
    print(f"\nPreparing training data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 128
    n_points = 100
    batch_size = 32
    
    # Training data
    train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Test data
    signals = generative_model.get_signal(num_samples=n_samples, n_points=n_points).tensors[0]
    example_batch, _ = next(iter(torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
    
    print(f"Training data: {len(train_dataset)} samples, batch size {batch_size}")
    print(f"Test data: {signals.shape}")
    
    # Create models with different seeds to see training differences
    print(f"\nCreating models...")
    
    # Use same seed for fair comparison of architectures
    torch.manual_seed(12345)
    a1_model = create_a1_final_model(example_batch, signals)
    print(f"A1 model created: {sum(p.numel() for p in a1_model.parameters())} parameters")
    
    torch.manual_seed(12345)  # Same initialization
    a2_model = create_a2_model(example_batch, signals)
    print(f"A2 model created: {sum(p.numel() for p in a2_model.parameters())} parameters")
    
    # Setup optimizers
    learning_rate = 0.001
    a1_optimizer = optim.Adam(a1_model.parameters(), lr=learning_rate)
    a2_optimizer = optim.Adam(a2_model.parameters(), lr=learning_rate)
    
    print(f"Optimizers: Adam with lr={learning_rate}")
    
    # Train both models
    num_epochs = 100
    print(f"\nStarting training ({num_epochs} epochs)...")
    
    # Train A1
    a1_history = train_model(a1_model, train_loader, a1_optimizer, num_epochs, "A1 (T-Statistic)")
    
    # Train A2
    a2_history = train_model(a2_model, train_loader, a2_optimizer, num_epochs, "A2 (Signature Scoring)")
    
    # Compare training dynamics
    training_summary = compare_training_dynamics(
        a1_history.get_history(), 
        a2_history.get_history(), 
        'training_results'
    )
    
    # Evaluate trained models
    trained_results = evaluate_trained_models(a1_model, a2_model, example_batch, signals)
    
    # Create post-training comparison
    create_post_training_comparison(trained_results, training_summary, 'training_results')
    
    # Save results to CSV
    results_df = pd.DataFrame(trained_results)
    results_df.to_csv('training_results/trained_model_comparison.csv', index=False)
    
    # Print final summary
    print(f"\n" + "="*60)
    print("TRAINING COMPARISON COMPLETE")
    print("="*60)
    
    a1_data = results_df[results_df['model_type'] == 'A1_Trained'].iloc[0]
    a2_data = results_df[results_df['model_type'] == 'A2_Trained'].iloc[0]
    
    print(f"Final Results:")
    print(f"  A1 (T-Statistic):")
    print(f"    Training Loss: {training_summary['a1_final_loss']:.6f}")
    print(f"    Generation RMSE: {a1_data['rmse']:.4f}")
    print(f"    Path Correlation: {a1_data['mean_path_correlation']:.4f}")
    
    print(f"  A2 (Signature Scoring):")
    print(f"    Training Loss: {training_summary['a2_final_loss']:.6f}")
    print(f"    Generation RMSE: {a2_data['rmse']:.4f}")
    print(f"    Path Correlation: {a2_data['mean_path_correlation']:.4f}")
    
    # Determine overall winner
    a1_wins = 0
    a2_wins = 0
    
    if training_summary['a1_final_loss'] < training_summary['a2_final_loss']:
        a1_wins += 1
    else:
        a2_wins += 1
    
    if a1_data['rmse'] < a2_data['rmse']:
        a1_wins += 1
    else:
        a2_wins += 1
        
    if a1_data['mean_path_correlation'] > a2_data['mean_path_correlation']:
        a1_wins += 1
    else:
        a2_wins += 1
    
    print(f"\nOverall Assessment:")
    print(f"  A1 wins: {a1_wins}/3 metrics")
    print(f"  A2 wins: {a2_wins}/3 metrics")
    
    if a1_wins > a2_wins:
        print(f"  🏆 A1 (T-Statistic) performs better overall")
    elif a2_wins > a1_wins:
        print(f"  🏆 A2 (Signature Scoring) performs better overall")
    else:
        print(f"  🤝 Both models perform similarly")
    
    print(f"\nFiles Generated:")
    print(f"  • training_results/trained_model_comparison.csv")
    print(f"  • training_results/training_dynamics_comparison.png")
    print(f"  • training_results/post_training_comparison.png")
    
    print(f"\n✅ Training comparison complete!")
    print(f"   Now we can see the real effects of different loss functions!")
    
    return results_df, training_summary


if __name__ == "__main__":
    try:
        results_df, training_summary = run_training_comparison()
        print(f"\n🎉 SUCCESS: Training comparison reveals loss function effects!")
        
    except Exception as e:
        print(f"\n❌ Training comparison failed: {e}")
        import traceback
        traceback.print_exc()

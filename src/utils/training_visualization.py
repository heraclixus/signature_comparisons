"""
Training Visualization Utilities

Reusable visualization functions for model training progress and trajectory comparison.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy import stats
from typing import Optional, Tuple, Dict, Any
import os


class TrainingVisualizer:
    """Handles trajectory visualization during model training."""
    
    def __init__(self, output_dir: str, model_id: str):
        """
        Initialize training visualizer.
        
        Args:
            output_dir: Base directory for saving visualizations
            model_id: Model identifier for organizing files
        """
        self.visualization_dir = Path(output_dir) / "training_visualizations" / model_id
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        
    def create_trajectory_comparison(self, model, train_loader, epoch: int, device: torch.device) -> Optional[Dict[str, float]]:
        """
        Create trajectory visualization comparing ground truth vs generated.
        
        Args:
            model: Model to evaluate
            train_loader: Training data loader for ground truth
            epoch: Current epoch number
            device: Device for computation
            
        Returns:
            Dictionary with computed metrics or None if visualization failed
        """
        try:
            # Get a sample batch for ground truth
            sample_batch, _ = next(iter(train_loader))
            sample_batch = sample_batch.to(device)
            
            # Extract ground truth trajectories (limit to first 10 for clarity)
            gt_trajectories = sample_batch[:10].detach().cpu().numpy()
            
            # Generate samples from current model
            model.eval()
            try:
                with torch.no_grad():
                    generated_trajectories = self._generate_model_samples(model, sample_batch[:10])
            finally:
                model.train()
            
            # Create visualization plots
            metrics = self._create_comparison_plots(gt_trajectories, generated_trajectories, epoch)
            
            return metrics
            
        except Exception as e:
            print(f"    ⚠️ Trajectory visualization failed at epoch {epoch}: {e}")
            return None
    
    def _generate_model_samples(self, model, sample_batch: torch.Tensor) -> np.ndarray:
        """Generate samples from model with fallback for different model types."""
        if hasattr(model, 'generate_samples'):
            # For D2/D3/D4 models with generate_samples method
            generated = model.generate_samples(len(sample_batch))
        elif hasattr(model, 'sample'):
            # For models with sample method
            generated = model.sample(sample_batch)
        else:
            # Fallback: use forward pass
            generated = model(sample_batch)
        
        return generated.detach().cpu().numpy()
    
    def _create_comparison_plots(self, gt_trajectories: np.ndarray, generated_trajectories: np.ndarray, 
                               epoch: int) -> Dict[str, float]:
        """Create comparison plots and return metrics."""
        
        # Create time axis
        time_points = np.linspace(0, 1, gt_trajectories.shape[-1])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{self.model_id} Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Determine plot dimension for multi-dimensional data
        plot_dim = self._get_plot_dimension(gt_trajectories)
        
        # Plot 1: Original scale comparison
        self._plot_original_scale_comparison(axes[0], gt_trajectories, generated_trajectories, 
                                           time_points, plot_dim)
        
        # Plot 2: Normalized comparison or distribution analysis
        metrics = self._plot_normalized_or_distribution(axes[1], gt_trajectories, generated_trajectories,
                                                       time_points, plot_dim)
        
        plt.tight_layout()
        
        # Save visualization
        viz_filename = f'{self.model_id}_epoch_{epoch:03d}_trajectories.png'
        viz_path = self.visualization_dir / viz_filename
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print metrics
        print(f"    📊 Epoch {epoch} - Wasserstein: {metrics['wasserstein']:.3f}, Scale: {metrics['scale_factor']:.1f}x")
        print(f"    🎨 Visualization saved: {viz_filename}")
        
        return metrics
    
    def _get_plot_dimension(self, trajectories: np.ndarray) -> int:
        """Determine which dimension to plot for multi-dimensional data."""
        if len(trajectories.shape) == 3:  # [batch, dim, seq_len]
            # For 2D data, plot dimension 1 (values), not dimension 0 (time)
            return 1 if trajectories.shape[1] > 1 else 0
        else:  # [batch, seq_len] - 1D data
            return None
    
    def _plot_original_scale_comparison(self, ax, gt_trajectories: np.ndarray, generated_trajectories: np.ndarray,
                                      time_points: np.ndarray, plot_dim: Optional[int]):
        """Plot original scale comparison."""
        
        # Plot ground truth
        if plot_dim is not None:  # Multi-dimensional data
            for i in range(min(5, len(gt_trajectories))):
                ax.plot(time_points, gt_trajectories[i, plot_dim, :], 'r-', alpha=0.7, linewidth=1.5,
                       label='Ground Truth' if i == 0 else "")
            for i in range(min(5, len(generated_trajectories))):
                ax.plot(time_points, generated_trajectories[i, plot_dim, :], 'b-', alpha=0.6, linewidth=1,
                       label='Generated' if i == 0 else "")
        else:  # 1D data
            for i in range(min(5, len(gt_trajectories))):
                ax.plot(time_points, gt_trajectories[i, :], 'r-', alpha=0.7, linewidth=1.5,
                       label='Ground Truth' if i == 0 else "")
            for i in range(min(5, len(generated_trajectories))):
                ax.plot(time_points, generated_trajectories[i, :], 'b-', alpha=0.6, linewidth=1,
                       label='Generated' if i == 0 else "")
        
        ax.set_title('Trajectories (Original Scale)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_normalized_or_distribution(self, ax, gt_trajectories: np.ndarray, generated_trajectories: np.ndarray,
                                        time_points: np.ndarray, plot_dim: Optional[int]) -> Dict[str, float]:
        """Plot normalized comparison or distribution analysis and return metrics."""
        
        # Compute statistics
        if plot_dim is not None:  # Multi-dimensional data
            gt_values = gt_trajectories[:, plot_dim, :]
            gen_values = generated_trajectories[:, plot_dim, :]
        else:  # 1D data
            gt_values = gt_trajectories
            gen_values = generated_trajectories
        
        gt_std = gt_values.std()
        gen_std = gen_values.std()
        gt_mean = gt_values.mean()
        gen_mean = gen_values.mean()
        
        scale_factor = gen_std / gt_std if gt_std > 0 else 1.0
        
        # Compute Wasserstein distance
        wasserstein = stats.wasserstein_distance(gt_values.flatten(), gen_values.flatten())
        
        # If scale difference is large, show normalized version
        if scale_factor > 10 or scale_factor < 0.1:
            # Normalize generated to match ground truth scale
            gen_normalized = (gen_values - gen_mean) / gen_std * gt_std + gt_mean
            
            if plot_dim is not None:
                for i in range(min(5, len(gt_trajectories))):
                    ax.plot(time_points, gt_trajectories[i, plot_dim, :], 'r-', alpha=0.7, linewidth=1.5,
                           label='Ground Truth' if i == 0 else "")
                for i in range(min(5, len(gen_normalized))):
                    ax.plot(time_points, gen_normalized[i, :], 'b-', alpha=0.6, linewidth=1,
                           label='Generated (Normalized)' if i == 0 else "")
            else:
                for i in range(min(5, len(gt_trajectories))):
                    ax.plot(time_points, gt_trajectories[i, :], 'r-', alpha=0.7, linewidth=1.5,
                           label='Ground Truth' if i == 0 else "")
                for i in range(min(5, len(gen_normalized))):
                    ax.plot(time_points, gen_normalized[i, :], 'b-', alpha=0.6, linewidth=1,
                           label='Generated (Normalized)' if i == 0 else "")
            
            ax.set_title(f'Normalized Comparison\\n(Scale Factor: {scale_factor:.1f}x)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
        else:
            # Scales are similar, show distribution comparison
            if plot_dim is not None:
                gt_final = gt_trajectories[:, plot_dim, -1]
                gen_final = generated_trajectories[:, plot_dim, -1]
            else:
                gt_final = gt_trajectories[:, -1]
                gen_final = generated_trajectories[:, -1]
            
            ax.hist(gt_final, bins=15, alpha=0.7, label='Ground Truth', density=True, color='red')
            ax.hist(gen_final, bins=15, alpha=0.7, label='Generated', density=True, color='blue')
            ax.set_title('Final Value Distribution')
            ax.set_xlabel('Final Value')
            ax.set_ylabel('Density')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return {
            'wasserstein': wasserstein,
            'scale_factor': scale_factor,
            'gt_mean': gt_mean,
            'gt_std': gt_std,
            'gen_mean': gen_mean,
            'gen_std': gen_std
        }


def create_training_progress_plot(training_history: Dict[str, Any], output_dir: str, model_id: str):
    """
    Create training progress plot from training history.
    
    Args:
        training_history: Dictionary containing training metrics
        output_dir: Directory to save the plot
        model_id: Model identifier
    """
    if 'losses' not in training_history:
        return
    
    losses = training_history['losses']
    epochs = range(1, len(losses) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    
    # Mark best epoch
    if 'best_epoch' in training_history:
        best_epoch = training_history['best_epoch']
        best_loss = training_history['best_loss']
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax.plot(best_epoch, best_loss, 'ro', markersize=8, label=f'Best Loss: {best_loss:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_id} Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'{model_id}_training_progress.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training progress plot saved: {plot_path}")


def create_model_comparison_plot(models_history: Dict[str, Dict], output_dir: str, dataset_name: str):
    """
    Create comparison plot of multiple models' training progress.
    
    Args:
        models_history: Dictionary of model_id -> training_history
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset for the title
    """
    if not models_history:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Model Training Comparison: {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    # Plot 1: Training loss comparison
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_history)))
    
    for (model_id, history), color in zip(models_history.items(), colors):
        if 'losses' in history:
            epochs = range(1, len(history['losses']) + 1)
            ax1.plot(epochs, history['losses'], color=color, linewidth=2, label=model_id)
            
            # Mark best epoch
            if 'best_epoch' in history:
                ax1.plot(history['best_epoch'], history['best_loss'], 'o', color=color, markersize=6)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Best loss comparison
    ax2 = axes[1]
    
    model_names = []
    best_losses = []
    best_epochs = []
    
    for model_id, history in models_history.items():
        if 'best_loss' in history:
            model_names.append(model_id)
            best_losses.append(history['best_loss'])
            best_epochs.append(history.get('best_epoch', 0))
    
    if model_names:
        bars = ax2.bar(model_names, best_losses, alpha=0.7, color=colors[:len(model_names)])
        
        # Add value labels on bars
        for bar, loss, epoch in zip(bars, best_losses, best_epochs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.3f}\\n(ep {epoch})', ha='center', va='bottom', fontsize=9)
    
    ax2.set_ylabel('Best Loss')
    ax2.set_title('Best Loss Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f'model_comparison_{dataset_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison plot saved: {plot_path}")


def analyze_generation_quality(model, test_data: torch.Tensor, device: torch.device, 
                             model_id: str) -> Dict[str, float]:
    """
    Analyze the quality of model generation compared to test data.
    
    Args:
        model: Trained model to evaluate
        test_data: Ground truth test data
        device: Device for computation
        model_id: Model identifier for logging
        
    Returns:
        Dictionary with quality metrics
    """
    model.eval()
    
    try:
        with torch.no_grad():
            # Generate samples
            if hasattr(model, 'generate_samples'):
                generated = model.generate_samples(len(test_data))
            else:
                generated = model(test_data.to(device))
            
            # Convert to numpy
            gen_np = generated.detach().cpu().numpy()
            test_np = test_data.detach().cpu().numpy()
            
            # Compute metrics
            gen_values = gen_np.flatten()
            test_values = test_np.flatten()
            
            wasserstein = stats.wasserstein_distance(gen_values, test_values)
            ks_stat, ks_pvalue = stats.ks_2samp(gen_values, test_values)
            
            # Scale analysis
            scale_factor = gen_np.std() / test_np.std() if test_np.std() > 0 else 1.0
            
            # Statistical comparison
            mse = np.mean((gen_np.mean(axis=0) - test_np.mean(axis=0))**2)
            
            metrics = {
                'wasserstein_distance': wasserstein,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'scale_factor': scale_factor,
                'mse_means': mse,
                'gen_mean': gen_np.mean(),
                'gen_std': gen_np.std(),
                'test_mean': test_np.mean(),
                'test_std': test_np.std()
            }
            
            print(f"📊 {model_id} Generation Quality:")
            print(f"   Wasserstein Distance: {wasserstein:.3f}")
            print(f"   KS Statistic: {ks_stat:.3f} (p={ks_pvalue:.3f})")
            print(f"   Scale Factor: {scale_factor:.1f}x")
            
            return metrics
            
    except Exception as e:
        print(f"⚠️ Generation quality analysis failed for {model_id}: {e}")
        return {
            'wasserstein_distance': 9999.0,
            'ks_statistic': 1.0,
            'ks_pvalue': 0.0,
            'scale_factor': 9999.0,
            'mse_means': 9999.0,
            'gen_mean': 0.0,
            'gen_std': 0.0,
            'test_mean': test_data.mean().item(),
            'test_std': test_data.std().item()
        }
    finally:
        model.train()


def create_final_training_summary(models_history: Dict[str, Dict], output_dir: str, dataset_name: str):
    """
    Create comprehensive training summary with visualizations.
    
    Args:
        models_history: Dictionary of model_id -> training_history
        output_dir: Directory to save summary
        dataset_name: Dataset name for the summary
    """
    if not models_history:
        return
    
    # Create summary plot
    create_model_comparison_plot(models_history, output_dir, dataset_name)
    
    # Create individual progress plots
    for model_id, history in models_history.items():
        create_training_progress_plot(history, output_dir, model_id)
    
    # Save text summary
    summary_path = Path(output_dir) / f'training_summary_{dataset_name}.txt'
    
    with open(summary_path, 'w') as f:
        f.write(f"Training Summary: {dataset_name.upper()} Dataset\\n")
        f.write("=" * 50 + "\\n\\n")
        
        for model_id, history in models_history.items():
            f.write(f"{model_id}:\\n")
            f.write(f"  Best Loss: {history.get('best_loss', 'N/A')}\\n")
            f.write(f"  Best Epoch: {history.get('best_epoch', 'N/A')}\\n")
            f.write(f"  Final Loss: {history.get('final_loss', 'N/A')}\\n")
            f.write(f"  Total Time: {history.get('total_time', 'N/A')}s\\n")
            f.write(f"  Epochs Trained: {history.get('epochs_trained', 'N/A')}\\n\\n")
    
    print(f"✅ Training summary saved: {summary_path}")

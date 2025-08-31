#!/usr/bin/env python3
"""
D2 Proper Training Test

Tests D2 models with extensive training to see if they can learn proper distributions.
Uses one clean output directory and focuses on training dynamics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.d2_base_model import create_d2_model, create_d2_transformer_model
from dataset.multi_dataset import MultiDatasetManager


class D2ProperTraining:
    """Proper D2 training with extensive epochs and monitoring."""
    
    def __init__(self):
        # Use single clean directory
        self.output_dir = Path("d2_debug")
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training tracking
        self.training_log = []
        
    def train_d2_extensively(self, dataset_name: str = 'ou_process', 
                           num_epochs: int = 500, eval_every: int = 50,
                           num_samples: int = 100, seq_len: int = 64):
        """Train D2 models extensively to see if they can learn proper distributions."""
        
        print(f"🚀 D2 EXTENSIVE TRAINING: {dataset_name.upper()}")
        print("=" * 60)
        print(f"Training for {num_epochs} epochs, evaluating every {eval_every}")
        
        # Load dataset
        print(f"📊 Loading {dataset_name} dataset...")
        manager = MultiDatasetManager(use_persistence=True)
        dataset = manager.get_dataset(dataset_name, num_samples=num_samples, n_points=seq_len)
        
        # Extract 1D data
        data_list = []
        for i in range(len(dataset)):
            sample = dataset[i][0]  # [2, seq_len]
            values = sample[1, :].numpy()  # Take values, not time
            data_list.append(values)
        
        data = torch.tensor(np.array(data_list), dtype=torch.float32)
        data = data.unsqueeze(1)  # [batch, 1, seq_len]
        actual_seq_len = data.shape[2]
        
        print(f"   Data shape: {data.shape}")
        print(f"   Ground truth range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"   Ground truth std: {data.std():.3f}")
        
        # Create models
        print(f"\\n🏗️ Creating D2 models...")
        d2_mlp = create_d2_model(
            dim=1, seq_len=actual_seq_len, generator_type='feedforward',
            hidden_size=64, num_layers=3, population_size=5, test_mode=False
        )
        
        d2_transformer = create_d2_transformer_model(
            dim=1, seq_len=actual_seq_len, hidden_size=64, num_layers=3, num_heads=8,
            population_size=5
        )
        
        print(f"   D2-MLP: {sum(p.numel() for p in d2_mlp.parameters()):,} parameters")
        print(f"   D2-Transformer: {sum(p.numel() for p in d2_transformer.parameters()):,} parameters")
        
        # Setup optimizers with lower learning rate
        mlp_optimizer = torch.optim.Adam(d2_mlp.parameters(), lr=5e-4)
        transformer_optimizer = torch.optim.Adam(d2_transformer.parameters(), lr=5e-4)
        
        # Training loop
        print(f"\\n🚀 Starting extensive training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training step for both models
            d2_mlp.train()
            d2_transformer.train()
            
            # Simple training approach: try to minimize difference from data statistics
            mlp_loss = self._train_step(d2_mlp, mlp_optimizer, data)
            transformer_loss = self._train_step(d2_transformer, transformer_optimizer, data)
            
            # Evaluation and logging
            if epoch % eval_every == 0 or epoch == num_epochs - 1:
                elapsed = time.time() - start_time
                print(f"\\n📊 Epoch {epoch:3d} ({elapsed:.1f}s):")
                print(f"   MLP Loss: {mlp_loss:.4f}")
                print(f"   Transformer Loss: {transformer_loss:.4f}")
                
                # Evaluate generation quality
                mlp_metrics = self._evaluate_generation(d2_mlp, data, "MLP")
                transformer_metrics = self._evaluate_generation(d2_transformer, data, "Transformer")
                
                # Log metrics
                log_entry = {
                    'epoch': epoch,
                    'mlp_loss': mlp_loss,
                    'transformer_loss': transformer_loss,
                    'mlp_metrics': mlp_metrics,
                    'transformer_metrics': transformer_metrics
                }
                self.training_log.append(log_entry)
                
                print(f"   MLP - Wasserstein: {mlp_metrics['wasserstein']:.1f}, Scale: {mlp_metrics['scale_factor']:.1f}x")
                print(f"   Transformer - Wasserstein: {transformer_metrics['wasserstein']:.1f}, Scale: {transformer_metrics['scale_factor']:.1f}x")
                
                # Create progress plots
                if epoch > 0:
                    self._create_progress_plots()
        
        # Final evaluation and visualization
        print(f"\\n🎯 Final Evaluation...")
        final_mlp = self._evaluate_generation(d2_mlp, data, "MLP", detailed=True)
        final_transformer = self._evaluate_generation(d2_transformer, data, "Transformer", detailed=True)
        
        self._create_final_comparison(d2_mlp, d2_transformer, data, dataset_name)
        self._save_training_summary(dataset_name, final_mlp, final_transformer)
        
        return final_mlp, final_transformer
    
    def _train_step(self, model, optimizer, data):
        """Single training step."""
        optimizer.zero_grad()
        
        try:
            # Generate samples
            batch_size = min(32, data.shape[0])  # Use smaller batches
            generated = model.generate_samples(batch_size)
            target = data[:batch_size]
            
            # Simple MSE loss on statistics
            gen_mean = generated.mean()
            gen_std = generated.std()
            target_mean = target.mean()
            target_std = target.std()
            
            # Loss: try to match mean and std
            mean_loss = torch.nn.functional.mse_loss(gen_mean, target_mean)
            std_loss = torch.nn.functional.mse_loss(gen_std, target_std)
            
            # Also add direct MSE loss (smaller weight)
            direct_loss = torch.nn.functional.mse_loss(generated, target) * 0.1
            
            total_loss = mean_loss + std_loss + direct_loss
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            print(f"   Warning: Training step failed: {e}")
            return 1000.0
    
    def _evaluate_generation(self, model, data, model_name: str, detailed: bool = False):
        """Evaluate model generation quality."""
        model.eval()
        
        with torch.no_grad():
            try:
                generated = model.generate_samples(50)
                
                # Convert to numpy
                gen_np = generated.detach().cpu().numpy()
                data_np = data.detach().cpu().numpy()
                
                # Compute metrics
                gen_values = gen_np.flatten()
                data_values = data_np.flatten()
                
                wasserstein = stats.wasserstein_distance(gen_values, data_values)
                scale_factor = gen_np.std() / data_np.std()
                
                metrics = {
                    'wasserstein': wasserstein,
                    'scale_factor': scale_factor,
                    'gen_mean': gen_np.mean(),
                    'gen_std': gen_np.std(),
                    'data_mean': data_np.mean(),
                    'data_std': data_np.std()
                }
                
                if detailed:
                    metrics['generated_samples'] = gen_np
                
                return metrics
                
            except Exception as e:
                print(f"   Warning: Evaluation failed for {model_name}: {e}")
                return {
                    'wasserstein': 9999.0,
                    'scale_factor': 9999.0,
                    'gen_mean': 0.0,
                    'gen_std': 0.0,
                    'data_mean': data.mean().item(),
                    'data_std': data.std().item()
                }
    
    def _create_progress_plots(self):
        """Create training progress plots."""
        if len(self.training_log) < 2:
            return
            
        epochs = [entry['epoch'] for entry in self.training_log]
        mlp_losses = [entry['mlp_loss'] for entry in self.training_log]
        transformer_losses = [entry['transformer_loss'] for entry in self.training_log]
        mlp_wasserstein = [entry['mlp_metrics']['wasserstein'] for entry in self.training_log]
        transformer_wasserstein = [entry['transformer_metrics']['wasserstein'] for entry in self.training_log]
        mlp_scales = [entry['mlp_metrics']['scale_factor'] for entry in self.training_log]
        transformer_scales = [entry['transformer_metrics']['scale_factor'] for entry in self.training_log]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('D2 Training Progress', fontsize=16, fontweight='bold')
        
        # Loss plot
        ax1 = axes[0, 0]
        ax1.plot(epochs, mlp_losses, 'b-o', label='D2-MLP', markersize=3)
        ax1.plot(epochs, transformer_losses, 'g-o', label='D2-Transformer', markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Wasserstein distance plot
        ax2 = axes[0, 1]
        ax2.plot(epochs, mlp_wasserstein, 'b-o', label='D2-MLP', markersize=3)
        ax2.plot(epochs, transformer_wasserstein, 'g-o', label='D2-Transformer', markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Wasserstein Distance')
        ax2.set_title('Distribution Quality (Lower = Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Scale factor plot
        ax3 = axes[1, 0]
        ax3.plot(epochs, mlp_scales, 'b-o', label='D2-MLP', markersize=3)
        ax3.plot(epochs, transformer_scales, 'g-o', label='D2-Transformer', markersize=3)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Scale')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Scale Factor (Generated/Ground Truth)')
        ax3.set_title('Scale Learning (Target = 1.0)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Combined improvement plot
        ax4 = axes[1, 1]
        # Compute improvement scores (lower is better)
        mlp_scores = [w * s for w, s in zip(mlp_wasserstein, mlp_scales)]
        transformer_scores = [w * s for w, s in zip(transformer_wasserstein, transformer_scales)]
        
        ax4.plot(epochs, mlp_scores, 'b-o', label='D2-MLP', markersize=3)
        ax4.plot(epochs, transformer_scores, 'g-o', label='D2-Transformer', markersize=3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Combined Score (Wasserstein × Scale)')
        ax4.set_title('Overall Quality (Lower = Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save progress plot
        progress_path = self.output_dir / 'training_progress.png'
        plt.savefig(progress_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_final_comparison(self, d2_mlp, d2_transformer, data, dataset_name):
        """Create final trajectory comparison."""
        # Generate final samples
        d2_mlp.eval()
        d2_transformer.eval()
        
        with torch.no_grad():
            mlp_samples = d2_mlp.generate_samples(20)
            transformer_samples = d2_transformer.generate_samples(20)
        
        mlp_np = mlp_samples.detach().cpu().numpy()
        transformer_np = transformer_samples.detach().cpu().numpy()
        gt_np = data.detach().cpu().numpy()
        
        time_points = np.linspace(0, 1, gt_np.shape[2])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'D2 Final Results: {dataset_name.upper()}', fontsize=16, fontweight='bold')
        
        # Original scale comparison
        ax1 = axes[0, 0]
        for i in range(min(5, len(gt_np))):
            ax1.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2, 
                    label='Ground Truth' if i == 0 else "")
        for i in range(min(8, len(mlp_np))):
            ax1.plot(time_points, mlp_np[i, 0, :], 'b-', alpha=0.4, linewidth=1,
                    label='D2-MLP' if i == 0 else "")
        ax1.set_title('MLP vs Ground Truth (Original Scale)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for i in range(min(5, len(gt_np))):
            ax2.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2, 
                    label='Ground Truth' if i == 0 else "")
        for i in range(min(8, len(transformer_np))):
            ax2.plot(time_points, transformer_np[i, 0, :], 'g-', alpha=0.4, linewidth=1,
                    label='D2-Transformer' if i == 0 else "")
        ax2.set_title('Transformer vs Ground Truth (Original Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Normalized comparison
        mlp_norm = (mlp_np - mlp_np.mean()) / mlp_np.std() * gt_np.std() + gt_np.mean()
        transformer_norm = (transformer_np - transformer_np.mean()) / transformer_np.std() * gt_np.std() + gt_np.mean()
        
        ax3 = axes[1, 0]
        for i in range(min(5, len(gt_np))):
            ax3.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2, 
                    label='Ground Truth' if i == 0 else "")
        for i in range(min(8, len(mlp_norm))):
            ax3.plot(time_points, mlp_norm[i, 0, :], 'b-', alpha=0.6, linewidth=1,
                    label='D2-MLP (Normalized)' if i == 0 else "")
        ax3.set_title('MLP (Normalized Scale)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        for i in range(min(5, len(gt_np))):
            ax4.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2, 
                    label='Ground Truth' if i == 0 else "")
        for i in range(min(8, len(transformer_norm))):
            ax4.plot(time_points, transformer_norm[i, 0, :], 'g-', alpha=0.6, linewidth=1,
                    label='D2-Transformer (Normalized)' if i == 0 else "")
        ax4.set_title('Transformer (Normalized Scale)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        final_path = self.output_dir / 'final_comparison.png'
        plt.savefig(final_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Final comparison saved: {final_path}")
    
    def _save_training_summary(self, dataset_name, mlp_metrics, transformer_metrics):
        """Save training summary to file."""
        summary_path = self.output_dir / 'training_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write(f"D2 Extensive Training Results: {dataset_name}\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Training Configuration:\\n")
            f.write(f"  Dataset: {dataset_name}\\n")
            f.write(f"  Total Epochs: {self.training_log[-1]['epoch']}\\n")
            f.write(f"  Evaluation Points: {len(self.training_log)}\\n\\n")
            
            f.write(f"Final Results:\\n")
            f.write(f"  MLP Wasserstein Distance: {mlp_metrics['wasserstein']:.2f}\\n")
            f.write(f"  Transformer Wasserstein Distance: {transformer_metrics['wasserstein']:.2f}\\n")
            f.write(f"  MLP Scale Factor: {mlp_metrics['scale_factor']:.1f}x\\n")
            f.write(f"  Transformer Scale Factor: {transformer_metrics['scale_factor']:.1f}x\\n\\n")
            
            f.write(f"Ground Truth Statistics:\\n")
            f.write(f"  Mean: {mlp_metrics['data_mean']:.3f}\\n")
            f.write(f"  Std: {mlp_metrics['data_std']:.3f}\\n\\n")
            
            f.write(f"MLP Generated Statistics:\\n")
            f.write(f"  Mean: {mlp_metrics['gen_mean']:.1f}\\n")
            f.write(f"  Std: {mlp_metrics['gen_std']:.1f}\\n\\n")
            
            f.write(f"Transformer Generated Statistics:\\n")
            f.write(f"  Mean: {transformer_metrics['gen_mean']:.1f}\\n")
            f.write(f"  Std: {transformer_metrics['gen_std']:.1f}\\n\\n")
            
            # Training progression
            f.write(f"Training Progression:\\n")
            for i, entry in enumerate(self.training_log):
                if i == 0 or i == len(self.training_log) - 1 or i % 2 == 0:
                    f.write(f"  Epoch {entry['epoch']:3d}: ")
                    f.write(f"MLP W={entry['mlp_metrics']['wasserstein']:.1f} S={entry['mlp_metrics']['scale_factor']:.1f}x, ")
                    f.write(f"Trans W={entry['transformer_metrics']['wasserstein']:.1f} S={entry['transformer_metrics']['scale_factor']:.1f}x\\n")
        
        print(f"✅ Training summary saved: {summary_path}")


def main():
    """Run extensive D2 training."""
    trainer = D2ProperTraining()
    
    print("🚀 D2 EXTENSIVE TRAINING EXPERIMENT")
    print("=" * 60)
    print("Testing whether D2 models can learn proper distributions with extensive training")
    
    # Train on OU process with many epochs
    mlp_final, transformer_final = trainer.train_d2_extensively(
        dataset_name='ou_process',
        num_epochs=200,  # Extensive training
        eval_every=20,
        num_samples=100,
        seq_len=64
    )
    
    print(f"\\n🏆 TRAINING COMPLETE!")
    print(f"Final MLP Wasserstein: {mlp_final['wasserstein']:.1f}, Scale: {mlp_final['scale_factor']:.1f}x")
    print(f"Final Transformer Wasserstein: {transformer_final['wasserstein']:.1f}, Scale: {transformer_final['scale_factor']:.1f}x")
    print(f"\\n📁 All results saved in: d2_debug/")


if __name__ == '__main__':
    main()

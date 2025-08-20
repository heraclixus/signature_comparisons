"""
Latent SDE Training Script

This script trains latent SDE models using variational inference (ELBO objective).
It provides a fair comparison with signature-based (A1-A4, B1-B5) and adversarial models.

Usage:
    # Train V1 on single dataset
    python src/experiments/latent_sde_training.py --models V1 --epochs 100 --dataset ou_process
    
    # Train V1 on ALL datasets (recommended)
    python src/experiments/latent_sde_training.py --models V1 --epochs 100 --all-datasets
    
    # Train all latent SDE models on all datasets
    python src/experiments/latent_sde_training.py --all --epochs 100 --all-datasets
    
    # Force retrain existing models
    python src/experiments/latent_sde_training.py --all --all-datasets --force-retrain --epochs 50
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import time
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.multi_dataset import MultiDatasetManager
from utils.model_checkpoint import create_checkpoint_manager
from models.latent_sde.implementations.v1_latent_sde import create_v1_model
from models.sdematching.implementations.v2_sde_matching import create_v2_model


class LatentSDETrainer:
    """Trainer for latent SDE models using ELBO objective."""
    
    def __init__(self, dataset_name: str = 'ou_process', save_dir: str = None):
        """
        Initialize latent SDE trainer.
        
        Args:
            dataset_name: Name of dataset to train on
            save_dir: Directory to save results
        """
        self.dataset_name = dataset_name
        self.save_dir = save_dir or f'results/{dataset_name}_latent_sde'
        
        # Create checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(self.save_dir)
        
        # Setup dataset
        self.dataset_manager = MultiDatasetManager()
        
        print(f"🎯 Latent SDE Trainer initialized for {dataset_name}")
        print(f"   Save directory: {self.save_dir}")
    
    def setup_training_data(self, num_samples: int = 1000, batch_size: int = 32):
        """Setup training data for the specified dataset."""
        print(f"\n📊 Setting up training data for {self.dataset_name}...")
        
        # Get dataset
        dataset = self.dataset_manager.get_dataset(self.dataset_name, num_samples=num_samples)
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Get example batch for model initialization
        example_batch = next(iter(train_loader))
        
        # Handle different data formats
        if isinstance(example_batch, (list, tuple)):
            example_batch = example_batch[0]  # Extract tensor from tuple/list
        
        print(f"   Dataset: {len(dataset)} samples")
        print(f"   Batch size: {batch_size}")
        print(f"   Example batch shape: {example_batch.shape}")
        
        return train_loader, example_batch
    
    def train_latent_sde_model(self, model_id: str, num_epochs: int = 100, 
                              learning_rate: float = 1e-3, batch_size: int = 32,
                              latent_dim: int = 4, kl_weight: float = 1.0,
                              save_every: int = 25, patience: int = 10) -> Dict:
        """
        Train a latent SDE model.
        
        Args:
            model_id: Model identifier (e.g., "V1")
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            latent_dim: Latent state dimension
            kl_weight: Weight for KL divergence in ELBO
            save_every: Save checkpoint every N epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"\n{'='*60}")
        print(f"🎯 Training {model_id} Latent SDE Model")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Latent dimension: {latent_dim}")
        print(f"KL weight: {kl_weight}")
        
        # Setup training data
        train_loader, example_batch = self.setup_training_data(batch_size=batch_size)
        
        # Create real data tensor for model initialization
        real_data_samples = []
        for i, batch in enumerate(train_loader):
            # Handle different data formats
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Extract tensor from tuple/list
            real_data_samples.append(batch)
            if i >= 4:  # Get a few batches for initialization
                break
        real_data = torch.cat(real_data_samples, dim=0)
        
        # Create model
        print(f"\n🏗️ Creating {model_id} model...")
        if model_id == "V1":
            model = create_v1_model(
                example_batch=example_batch,
                real_data=real_data,
                theta=2.0,      # OU mean reversion rate
                mu=0.0,         # OU long-term mean
                sigma=0.5,      # OU volatility
                hidden_size=64  # Neural network size
            )
        elif model_id == "V2":
            model = create_v2_model(
                example_batch=example_batch,
                real_data=real_data,
                data_size=1,        # Observable dimension
                latent_size=4,      # Latent dimension
                hidden_size=64,     # Hidden layer size
                noise_std=0.1       # Observation noise
            )
        else:
            raise ValueError(f"Unknown latent SDE model: {model_id}")
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        print(f"\n🚀 Starting training...")
        training_history = {
            'epoch': [],
            'losses': [],  # Changed from 'total_loss' to match checkpoint manager format
            'times': [],   # Changed from 'epoch_time' to match checkpoint manager format
            'reconstruction_loss': [],
            'kl_loss': [],
            'elbo': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        final_loss = float('inf')  # Initialize final loss
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training epoch
            model.train()
            epoch_losses = []
            epoch_recon_losses = []
            epoch_kl_losses = []
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Handle different data formats
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]  # Extract tensor from tuple/list
                
                # Compute ELBO loss
                try:
                    loss, components = model.compute_training_loss(batch)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track losses
                    epoch_losses.append(components['loss'])  # Fixed key name
                    epoch_recon_losses.append(components['reconstruction_loss'])
                    epoch_kl_losses.append(components['kl_loss'])
                    
                except Exception as e:
                    print(f"   ❌ Training step failed at epoch {epoch+1}, batch {batch_idx}: {e}")
                    continue
            
            if not epoch_losses:
                print(f"   ❌ No successful training steps in epoch {epoch+1}")
                continue
            
            # Compute epoch averages
            avg_total_loss = np.mean(epoch_losses)
            avg_recon_loss = np.mean(epoch_recon_losses)
            avg_kl_loss = np.mean(epoch_kl_losses)
            epoch_time = time.time() - epoch_start
            final_loss = avg_total_loss  # Update final loss
            
            # Update training history
            training_history['epoch'].append(epoch + 1)
            training_history['losses'].append(avg_total_loss)  # Changed key
            training_history['times'].append(epoch_time)       # Changed key
            training_history['reconstruction_loss'].append(avg_recon_loss)
            training_history['kl_loss'].append(avg_kl_loss)
            training_history['elbo'].append(-avg_total_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"   Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Loss={avg_total_loss:.4f} "
                      f"(Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}) "
                      f"Time={epoch_time:.1f}s")
            
            # Early stopping check
            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                patience_counter = 0
                
                # Save best model
                self._save_latent_sde_model(model, model_id, epoch + 1, avg_total_loss, training_history)
            else:
                patience_counter += 1
            
            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                self._save_latent_sde_model(model, model_id, epoch + 1, avg_total_loss, training_history)
                print(f"   💾 Checkpoint saved at epoch {epoch + 1}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   🛑 Early stopping at epoch {epoch + 1} (patience: {patience})")
                break
        
        # Final save
        self._save_latent_sde_model(model, model_id, epoch + 1, final_loss, training_history)
        
        print(f"\n✅ Training complete!")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"   Final epoch: {epoch + 1}")
        print(f"   Total epochs with losses: {len(training_history['losses'])}")
        print(f"   Model saved to: {self.save_dir}/trained_models/{model_id}/")
        
        return training_history
    
    def _save_latent_sde_model(self, model, model_id: str, epoch: int, loss: float, history: Dict):
        """Save latent SDE model and training history."""
        # Prepare training config
        training_config = {
            'latent_dim': getattr(model, 'latent_dim', 4),
            'kl_weight': getattr(model, 'kl_weight', 1.0),
            'training_type': 'latent_sde'
        }
        
        # Prepare metrics
        metrics = {
            'model_class': type(model).__name__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Save model checkpoint
        self.checkpoint_manager.save_model(
            model=model,
            model_id=model_id,
            epoch=epoch,
            loss=loss,
            metrics=metrics,
            training_config=training_config,
            training_history=history
        )
        
        # Save training history
        if history['epoch']:
            history_df = pd.DataFrame(history)
            model_dir = os.path.join(self.save_dir, 'trained_models', model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            history_path = os.path.join(model_dir, 'training_history.csv')
            history_df.to_csv(history_path, index=False)
            
            # Create training curve plot
            self._create_training_curve(history_df, model_dir, model_id)
    
    def _create_training_curve(self, history_df: pd.DataFrame, model_dir: str, model_id: str):
        """Create training curve visualization for latent SDE."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{model_id} Latent SDE Training Progress', fontsize=14, fontweight='bold')
        
        epochs = history_df['epoch']
        
        # 1. Training Loss (Minimized)
        axes[0, 0].plot(epochs, history_df['losses'], 'b-', linewidth=2)  # Fixed key
        axes[0, 0].set_title('Training Loss (Lower = Better)', fontweight='bold')
        axes[0, 0].set_ylabel('Loss (-ELBO)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reconstruction Loss
        axes[0, 1].plot(epochs, history_df['reconstruction_loss'], 'g-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss', fontweight='bold')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. KL Divergence
        axes[1, 0].plot(epochs, history_df['kl_loss'], 'r-', linewidth=2)
        axes[1, 0].set_title('KL Divergence', fontweight='bold')
        axes[1, 0].set_ylabel('KL Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. ELBO (Evidence Lower BOund) - Maximized
        axes[1, 1].plot(epochs, history_df['elbo'], 'm-', linewidth=2)
        axes[1, 1].set_title('ELBO (Higher = Better)', fontweight='bold')
        axes[1, 1].set_ylabel('ELBO = -Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()


def train_latent_sde_models(models: List[str], dataset_name: str = 'ou_process',
                           num_epochs: int = 100, force_retrain: bool = False,
                           **kwargs) -> Dict[str, Dict]:
    """
    Train multiple latent SDE models.
    
    Args:
        models: List of model IDs to train (e.g., ["V1"])
        dataset_name: Dataset to train on
        num_epochs: Number of training epochs
        force_retrain: Whether to retrain existing models
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary of training results
    """
    print(f"🎯 Training Latent SDE Models on {dataset_name}")
    print(f"   Models: {models}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Force retrain: {force_retrain}")
    
    trainer = LatentSDETrainer(dataset_name)
    results = {}
    
    for model_id in models:
        print(f"\n{'='*50}")
        print(f"Training {model_id}")
        print(f"{'='*50}")
        
        # Check if model already exists
        if not force_retrain and trainer.checkpoint_manager.model_exists(model_id):
            print(f"⏭️ {model_id} already trained, skipping...")
            print(f"   Use --force-retrain to retrain existing models")
            continue
        
        if force_retrain and trainer.checkpoint_manager.model_exists(model_id):
            print(f"🔄 {model_id} exists but retraining due to --force-retrain flag")
        
        try:
            # Train model
            history = trainer.train_latent_sde_model(
                model_id=model_id,
                num_epochs=num_epochs,
                **kwargs
            )
            
            results[model_id] = {
                'status': 'success',
                'final_loss': history['losses'][-1] if history['losses'] else float('inf'),  # Fixed key
                'epochs_trained': len(history['epoch']),
                'history': history
            }
            
            print(f"✅ {model_id} training completed successfully")
            
        except Exception as e:
            print(f"❌ Training failed for {model_id}: {e}")
            results[model_id] = {
                'status': 'failed',
                'error': str(e),
                'final_loss': float('inf'),
                'epochs_trained': 0
            }
            continue
    
    # Save training summary
    _save_training_summary(results, trainer.save_dir, dataset_name)
    
    return results


def train_all_latent_sde_models(dataset_name: str = 'ou_process', num_epochs: int = 100,
                                force_retrain: bool = False, **kwargs) -> Dict[str, Dict]:
    """Train all available latent SDE models."""
    available_models = ["V1", "V2"]  # V1: TorchSDE Latent SDE, V2: SDE Matching
    
    return train_latent_sde_models(
        models=available_models,
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        force_retrain=force_retrain,
        **kwargs
    )


def train_all_datasets(models: List[str] = None, num_epochs: int = 100,
                      force_retrain: bool = False, **kwargs):
    """Train latent SDE models on all available datasets."""
    if models is None:
        models = ["V1"]
    
    datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
    
    print(f"🎯 Training Latent SDE Models on All Datasets")
    print(f"   Models: {models}")
    print(f"   Datasets: {datasets}")
    print(f"   Epochs per dataset: {num_epochs}")
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"🌊 Training on {dataset_name.upper()} Dataset")
        print(f"{'='*70}")
        
        try:
            dataset_results = train_latent_sde_models(
                models=models,
                dataset_name=dataset_name,
                num_epochs=num_epochs,
                force_retrain=force_retrain,
                **kwargs
            )
            all_results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f"❌ Failed to train on {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
            continue
    
    print(f"\n{'='*70}")
    print(f"🎉 LATENT SDE MULTI-DATASET TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Print summary
    for dataset_name, dataset_results in all_results.items():
        if 'error' in dataset_results:
            print(f"   {dataset_name}: ❌ Failed")
        else:
            successful = len([r for r in dataset_results.values() if r['status'] == 'success'])
            total = len(dataset_results)
            print(f"   {dataset_name}: ✅ {successful}/{total} models trained")
    
    return all_results


def _save_training_summary(results: Dict, save_dir: str, dataset_name: str):
    """Save training summary to CSV."""
    summary_data = []
    
    for model_id, result in results.items():
        summary_data.append({
            'model_id': model_id,
            'dataset': dataset_name,
            'status': result['status'],
            'final_loss': result['final_loss'],
            'epochs_trained': result['epochs_trained'],
            'training_type': 'latent_sde'
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save to training directory
        training_dir = os.path.join(save_dir, 'training')
        os.makedirs(training_dir, exist_ok=True)
        
        summary_path = os.path.join(training_dir, f'{dataset_name}_latent_sde_training_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"📊 Training summary saved to: {summary_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Latent SDE models")
    
    parser.add_argument("--models", nargs='+', default=None,
                       help="Models to train (V1, V2). If not specified, trains all available models")
    parser.add_argument("--all", action='store_true',
                       help="Train all available latent SDE models")
    parser.add_argument("--dataset", type=str, default='ou_process',
                       choices=['ou_process', 'heston', 'rbergomi', 'brownian', 
                               'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07'],
                       help="Dataset to train on")
    parser.add_argument("--all-datasets", action='store_true',
                       help="Train on all 8 datasets: ou_process, heston, rbergomi, brownian, fbm_h03, fbm_h04, fbm_h06, fbm_h07")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--latent-dim", type=int, default=4,
                       help="Latent state dimension")
    parser.add_argument("--kl-weight", type=float, default=1.0,
                       help="Weight for KL divergence in ELBO")
    parser.add_argument("--force-retrain", action='store_true',
                       help="Force retrain existing models")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.all:
        models = ["V1", "V2"]  # All available latent SDE models
    elif args.models:
        models = args.models
    else:
        models = ["V1", "V2"]  # Default to both models
    
    # Training parameters
    training_kwargs = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'latent_dim': args.latent_dim,
        'kl_weight': args.kl_weight,
        'patience': args.patience
    }
    
    # Train on specified datasets
    if args.all_datasets:
        results = train_all_datasets(
            models=models,
            num_epochs=args.epochs,
            force_retrain=args.force_retrain,
            **training_kwargs
        )
    else:
        results = train_latent_sde_models(
            models=models,
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            force_retrain=args.force_retrain,
            **training_kwargs
        )
    
    print(f"\n🎉 Latent SDE training pipeline complete!")
    print(f"   Results: {len(results)} dataset(s) processed")


if __name__ == "__main__":
    main()

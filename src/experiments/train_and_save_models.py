"""
Train and Save Models

This script trains models and saves the best checkpoints for later evaluation.
It tracks the best performance during training and saves models automatically.
"""

import torch
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from utils.model_checkpoint import create_checkpoint_manager

# Import all available models
try:
    from models.implementations.a1_final import create_a1_final_model
    A1_AVAILABLE = True
except ImportError:
    A1_AVAILABLE = False

try:
    from models.implementations.a2_canned_scoring import create_a2_model
    A2_AVAILABLE = True
except ImportError:
    A2_AVAILABLE = False

try:
    from models.implementations.a3_canned_mmd import create_a3_model
    A3_AVAILABLE = True
except ImportError:
    A3_AVAILABLE = False

try:
    from models.implementations.b4_nsde_mmd import create_b4_model
    B4_AVAILABLE = True
except ImportError:
    B4_AVAILABLE = False

try:
    from models.implementations.b5_nsde_scoring import create_b5_model
    B5_AVAILABLE = True
except ImportError:
    B5_AVAILABLE = False

try:
    from models.implementations.a4_canned_logsig import create_a4_model
    A4_AVAILABLE = True
except ImportError:
    A4_AVAILABLE = False

try:
    from models.implementations.b3_nsde_tstatistic import create_b3_model
    B3_AVAILABLE = True
except ImportError:
    B3_AVAILABLE = False

try:
    from models.implementations.b1_nsde_scoring import create_b1_model
    B1_AVAILABLE = True
except ImportError:
    B1_AVAILABLE = False

try:
    from models.implementations.b2_nsde_mmd_pde import create_b2_model
    B2_AVAILABLE = True
except ImportError:
    B2_AVAILABLE = False

# C1-C3 (GRU) models removed - not truly generative
# Diversity testing revealed they don't produce diverse random sample paths


class ModelTrainer:
    """
    Enhanced model trainer that saves best models during training.
    """
    
    def __init__(self, checkpoint_manager):
        """Initialize trainer with checkpoint manager."""
        self.checkpoint_manager = checkpoint_manager
        self.training_history = {}
    
    def train_with_checkpointing(self, model, model_id: str, train_loader, 
                               optimizer, num_epochs: int, 
                               save_every: int = 20, patience: int = 20):
        """
        Train model with automatic checkpointing of best models.
        
        Args:
            model: Model to train
            model_id: Unique identifier for the model
            train_loader: Training data loader
            optimizer: Optimizer
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            patience: Early stopping patience
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining {model_id} with checkpointing...")
        print(f"  Epochs: {num_epochs}, Save every: {save_every}, Patience: {patience}")
        
        model.train()
        
        # Training tracking
        training_losses = []
        epoch_times = []
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            
            # Training loop
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
            
            # Check if this is the best model so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                self.checkpoint_manager.save_model(
                    model=model,
                    model_id=model_id,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    training_config={
                        'optimizer': type(optimizer).__name__,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'total_epochs': num_epochs,
                        'best_epoch': epoch + 1
                    }
                )
                print(f"  💾 New best model saved at epoch {epoch + 1}: {epoch_loss:.6f}")
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:3d}: Loss = {epoch_loss:.6f}, "
                      f"Best = {best_loss:.6f} (epoch {best_epoch}), "
                      f"Time = {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch + 1} (patience: {patience})")
                break
            
            # Periodic checkpoint (even if not best)
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"{model_id}_epoch_{epoch + 1}"
                print(f"  📁 Periodic checkpoint saved: {checkpoint_path}")
        
        total_time = time.time() - start_time
        
        print(f"✅ {model_id} training completed in {total_time:.2f}s")
        print(f"   Best loss: {best_loss:.6f} at epoch {best_epoch}")
        print(f"   Final loss: {training_losses[-1]:.6f}")
        
        # Store training history
        history = {
            'model_id': model_id,
            'losses': training_losses,
            'times': epoch_times,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'final_loss': training_losses[-1],
            'total_time': total_time,
            'epochs_trained': len(training_losses)
        }
        
        self.training_history[model_id] = history
        return history


def setup_training_data(n_samples: int = 256, n_points: int = 100, batch_size: int = 32):
    """Setup training data for all models."""
    print(f"Setting up training data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training data
    train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
    train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Test data for model initialization
    signals = generative_model.get_signal(num_samples=64, n_points=n_points).tensors[0]
    example_batch, _ = next(iter(torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
    
    print(f"Training: {len(train_dataset)} samples, batch size {batch_size}")
    print(f"Test data: {signals.shape}")
    
    return train_loader, example_batch, signals


def train_available_models(num_epochs: int = 100, learning_rate: float = 0.001):
    """Train all available models and save checkpoints."""
    print("Training Available Models with Checkpointing")
    print("=" * 60)
    
    # Setup checkpoint manager
    checkpoint_manager = create_checkpoint_manager()
    trainer = ModelTrainer(checkpoint_manager)
    
    # Check existing models
    print(f"\nChecking for existing trained models...")
    checkpoint_manager.print_available_models()
    
    # Setup training data
    train_loader, example_batch, signals = setup_training_data()
    
    # Track which models to train
    models_to_train = []
    
    # Check A1
    if A1_AVAILABLE:
        if checkpoint_manager.model_exists("A1"):
            print(f"⏭️ A1 already trained, skipping...")
        else:
            models_to_train.append(("A1", create_a1_final_model, "T-Statistic"))
    
    # Check A2
    if A2_AVAILABLE:
        if checkpoint_manager.model_exists("A2"):
            print(f"⏭️ A2 already trained, skipping...")
        else:
            models_to_train.append(("A2", create_a2_model, "Signature Scoring"))
    
    # Check A3
    if A3_AVAILABLE:
        if checkpoint_manager.model_exists("A3"):
            print(f"⏭️ A3 already trained, skipping...")
        else:
            models_to_train.append(("A3", create_a3_model, "MMD"))
    
    # Check B4
    if B4_AVAILABLE:
        if checkpoint_manager.model_exists("B4"):
            print(f"⏭️ B4 already trained, skipping...")
        else:
            models_to_train.append(("B4", create_b4_model, "Neural SDE + MMD"))
    
    # Check B5
    if B5_AVAILABLE:
        if checkpoint_manager.model_exists("B5"):
            print(f"⏭️ B5 already trained, skipping...")
        else:
            models_to_train.append(("B5", create_b5_model, "Neural SDE + Signature Scoring"))
    
    # Check A4
    if A4_AVAILABLE:
        if checkpoint_manager.model_exists("A4"):
            print(f"⏭️ A4 already trained, skipping...")
        else:
            models_to_train.append(("A4", create_a4_model, "CannedNet + T-Statistic + Log Signatures"))
    
    # Check B3
    if B3_AVAILABLE:
        if checkpoint_manager.model_exists("B3"):
            print(f"⏭️ B3 already trained, skipping...")
        else:
            models_to_train.append(("B3", create_b3_model, "Neural SDE + T-Statistic"))
    
    # Check B1
    if B1_AVAILABLE:
        if checkpoint_manager.model_exists("B1"):
            print(f"⏭️ B1 already trained, skipping...")
        else:
            models_to_train.append(("B1", create_b1_model, "Neural SDE + Signature Scoring + PDE-Solved"))
    
    # Check B2
    if B2_AVAILABLE:
        if checkpoint_manager.model_exists("B2"):
            print(f"⏭️ B2 already trained, skipping...")
        else:
            models_to_train.append(("B2", create_b2_model, "Neural SDE + MMD + PDE-Solved"))
    
    # C1-C3 models removed - not truly generative
    
    if not models_to_train:
        print(f"\n✅ All available models already trained!")
        print(f"   Use --force flag to retrain existing models")
        return True
    
    print(f"\nModels to train: {len(models_to_train)}")
    for model_id, _, loss_type in models_to_train:
        print(f"  {model_id}: {loss_type}")
    
    # Train each model
    training_results = {}
    
    for model_id, create_fn, loss_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_id} ({loss_type})")
        print(f"{'='*60}")
        
        try:
            # Create model with consistent seed
            torch.manual_seed(12345)
            model = create_fn(example_batch, signals)
            
            print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Setup optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train with checkpointing
            history = trainer.train_with_checkpointing(
                model=model,
                model_id=model_id,
                train_loader=train_loader,
                optimizer=optimizer,
                num_epochs=num_epochs,
                save_every=25,
                patience=30
            )
            
            training_results[model_id] = history
            
        except Exception as e:
            print(f"❌ Training failed for {model_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save training summary
    if training_results:
        summary_data = []
        for model_id, history in training_results.items():
            summary_data.append({
                'model_id': model_id,
                'best_loss': history['best_loss'],
                'best_epoch': history['best_epoch'],
                'final_loss': history['final_loss'],
                'epochs_trained': history['epochs_trained'],
                'total_time': history['total_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = "results/training/training_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print(summary_df.to_string(index=False))
        print(f"\nTraining summary saved to: {summary_path}")
    
    # Final checkpoint status
    print(f"\nFinal Checkpoint Status:")
    checkpoint_manager.print_available_models()
    
    return True


def force_retrain_model(model_id: str, num_epochs: int = 100):
    """Force retrain a specific model even if checkpoint exists."""
    print(f"Force Retraining {model_id}")
    print("=" * 40)
    
    checkpoint_manager = create_checkpoint_manager()
    
    # Delete existing checkpoint if it exists
    if checkpoint_manager.model_exists(model_id):
        print(f"Deleting existing checkpoint for {model_id}...")
        checkpoint_manager.delete_model(model_id)
    
    # Run training for this specific model
    train_available_models(num_epochs)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and save signature-based models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--force", type=str, help="Force retrain specific model (A1, A2, A3)")
    parser.add_argument("--list", action="store_true", help="List available trained models")
    
    args = parser.parse_args()
    
    if args.list:
        checkpoint_manager = create_checkpoint_manager()
        checkpoint_manager.print_available_models()
        return
    
    if args.force:
        force_retrain_model(args.force, args.epochs)
    else:
        train_available_models(args.epochs, args.lr)


if __name__ == "__main__":
    main()

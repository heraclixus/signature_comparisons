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
from dataset.multi_dataset import MultiDatasetManager
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
                               save_every: int = 20, patience: int = 10):
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
                
                # Save best model with training history
                current_history = {
                    'model_id': model_id,
                    'losses': training_losses.copy(),
                    'times': epoch_times.copy(),
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'total_time': time.time() - start_time,
                    'epochs_trained': epoch + 1
                }
                
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
                    },
                    training_history=current_history
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


def setup_training_data(n_samples: int = 32768, n_points: int = 64, batch_size: int = 128, dataset_name: str = 'ou_process'):
    """Setup training data for all models using persistence-enabled dataset manager."""
    print(f"Setting up training data for {dataset_name.upper()}...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use MultiDatasetManager with persistence for consistent data loading
    dataset_manager = MultiDatasetManager(use_persistence=True)
    
    if dataset_name == 'ou_process':
        # For OU process, we need both noise (for training) and signal (for model initialization)
        # Training data (noise)
        train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
        train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Test data for model initialization (signal)  
        signals_dataset = dataset_manager.get_dataset('ou_process', num_samples=256, n_points=n_points)
        signals = torch.stack([signals_dataset[i][0] for i in range(min(256, len(signals_dataset)))])
        example_batch, _ = next(iter(torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
    else:
        # For other datasets, use the dataset manager
        full_dataset = dataset_manager.get_dataset(dataset_name, num_samples=n_samples, n_points=n_points)
        train_loader = torchdata.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Get signals for model initialization
        signals_dataset = dataset_manager.get_dataset(dataset_name, num_samples=256, n_points=n_points)
        signals = torch.stack([signals_dataset[i][0] for i in range(min(256, len(signals_dataset)))])
        example_batch, _ = next(iter(torchdata.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)))
    
    print(f"Training: {len(train_loader.dataset)} samples, batch size {batch_size}")
    print(f"Test data: {signals.shape}")
    
    return train_loader, example_batch, signals


def train_available_models(num_epochs: int = 100, learning_rate: float = 0.001, dataset_name: str = 'ou_process', 
                          memory_optimized: bool = False, retrain_all: bool = False):
    """Train all available models and save checkpoints."""
    print(f"Training Available Models with Checkpointing on {dataset_name.upper()}")
    if memory_optimized:
        print("🧠 Memory optimization enabled for B-type models")
    if retrain_all:
        print("🔄 Retrain all mode: Ignoring existing checkpoints")
    print("=" * 60)
    
    # Setup checkpoint manager for specific dataset
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    trainer = ModelTrainer(checkpoint_manager)
    
    # Check existing models
    print(f"\nChecking for existing trained models...")
    checkpoint_manager.print_available_models()
    
    # Setup training data
    train_loader, example_batch, signals = setup_training_data(dataset_name=dataset_name)
    
    # Track which models to train
    models_to_train = []
    
    # Check A1
    if A1_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("A1"):
            print(f"⏭️ A1 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("A1"):
                print(f"🔄 A1 exists but retraining due to --retrain-all flag")
            models_to_train.append(("A1", create_a1_final_model, "T-Statistic"))
    
    # Check A2
    if A2_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("A2"):
            print(f"⏭️ A2 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("A2"):
                print(f"🔄 A2 exists but retraining due to --retrain-all flag")
            models_to_train.append(("A2", create_a2_model, "Signature Scoring"))
    
    # Check A3
    if A3_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("A3"):
            print(f"⏭️ A3 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("A3"):
                print(f"🔄 A3 exists but retraining due to --retrain-all flag")
            models_to_train.append(("A3", create_a3_model, "MMD"))
    
    # Check B4
    if B4_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("B4"):
            print(f"⏭️ B4 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("B4"):
                print(f"🔄 B4 exists but retraining due to --retrain-all flag")
            models_to_train.append(("B4", create_b4_model, "Neural SDE + MMD"))
    
    # Check B5
    if B5_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("B5"):
            print(f"⏭️ B5 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("B5"):
                print(f"🔄 B5 exists but retraining due to --retrain-all flag")
            models_to_train.append(("B5", create_b5_model, "Neural SDE + Signature Scoring"))
    
    # Check A4
    if A4_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("A4"):
            print(f"⏭️ A4 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("A4"):
                print(f"🔄 A4 exists but retraining due to --retrain-all flag")
            models_to_train.append(("A4", create_a4_model, "CannedNet + T-Statistic + Log Signatures"))
    
    # Check B3
    if B3_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("B3"):
            print(f"⏭️ B3 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("B3"):
                print(f"🔄 B3 exists but retraining due to --retrain-all flag")
            models_to_train.append(("B3", create_b3_model, "Neural SDE + T-Statistic"))
    
    # Check B1
    if B1_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("B1"):
            print(f"⏭️ B1 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("B1"):
                print(f"🔄 B1 exists but retraining due to --retrain-all flag")
            models_to_train.append(("B1", create_b1_model, "Neural SDE + Signature Scoring + PDE-Solved"))
    
    # Check B2
    if B2_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("B2"):
            print(f"⏭️ B2 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("B2"):
                print(f"🔄 B2 exists but retraining due to --retrain-all flag")
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
            
            # Choose training method based on model type and memory optimization flag
            if memory_optimized and model_id.startswith('B'):
                print(f"  🧠 Using memory-optimized training for {model_id}")
                # Convert to tensor data for memory-optimized training
                train_data = torch.stack([signals[i] for i in range(min(256, len(signals)))])
                success, best_loss, best_epoch = train_model_memory_optimized(
                    model, model_id, checkpoint_manager, train_data, num_epochs
                )
                # Create history object for consistency
                history = {
                    'model_id': model_id,
                    'losses': [best_loss],  # Simplified history
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'final_loss': best_loss,
                    'total_time': 0,
                    'epochs_trained': best_epoch
                }
            else:
                # Standard training with checkpointing
                history = trainer.train_with_checkpointing(
                    model=model,
                    model_id=model_id,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    save_every=25,
                    patience=10
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
        summary_dir = f'results/{dataset_name}/training'
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f'{dataset_name}_training_summary.csv')
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


def train_all_datasets(epochs: int = 100, lr: float = 0.001, memory_optimized: bool = False, retrain_all: bool = False):
    """Train all models on all datasets."""
    print("🚀 Multi-Dataset Training Pipeline")
    if memory_optimized:
        print("🧠 Memory optimization enabled for B-type models")
    print("=" * 70)
    
    # Initialize dataset manager with persistence
    dataset_manager = MultiDatasetManager(use_persistence=True)
    
    # Get all datasets (they will be loaded from disk if available, generated otherwise)
    datasets = {
        'ou_process': None,  # Use existing OU data generation
        'heston': dataset_manager.get_dataset('heston', num_samples=32768),
        'rbergomi': dataset_manager.get_dataset('rbergomi', num_samples=32768),
        'brownian': dataset_manager.get_dataset('brownian', num_samples=32768),
        'fbm_h03': dataset_manager.get_dataset('fbm_h03', num_samples=32768),
        'fbm_h04': dataset_manager.get_dataset('fbm_h04', num_samples=32768),
        'fbm_h06': dataset_manager.get_dataset('fbm_h06', num_samples=32768),
        'fbm_h07': dataset_manager.get_dataset('fbm_h07', num_samples=32768)
    }
    
    print(f"Training on {len(datasets)} datasets:")
    for dataset_name in datasets.keys():
        print(f"   📊 {dataset_name.upper()}")
    
    for dataset_name, dataset_data in datasets.items():
        print(f"\n{'='*70}")
        print(f"TRAINING ON {dataset_name.upper()} DATASET")
        print(f"{'='*70}")
        
        if dataset_name == 'ou_process':
            # Use existing OU training function
            train_available_models(epochs, lr, dataset_name='ou_process', memory_optimized=memory_optimized, retrain_all=retrain_all)
        else:
            # Train on new dataset
            train_available_models_on_dataset(dataset_name, dataset_data, epochs, lr, memory_optimized=memory_optimized, retrain_all=retrain_all)


def train_available_models_on_dataset(dataset_name: str, dataset_data, epochs: int = 100, lr: float = 0.001, 
                                     memory_optimized: bool = False, retrain_all: bool = False):
    """Train all available models on a specific dataset."""
    print(f"Training Available Models on {dataset_name.upper()} Dataset")
    if memory_optimized:
        print("🧠 Memory optimization enabled for B-type models")
    if retrain_all:
        print("🔄 Retrain all mode: Ignoring existing checkpoints")
    print("=" * 60)
    
    # Setup training data
    if dataset_data is not None:
        train_data = torch.stack([dataset_data[i][0] for i in range(min(32768, len(dataset_data)))])
        test_data = torch.stack([dataset_data[i][0] for i in range(min(256, len(dataset_data)))])
    else:
        # Fallback to OU process
        dataset = generative_model.get_signal(num_samples=32768)
        train_data = torch.stack([dataset[i][0] for i in range(32768)])
        test_data = torch.stack([dataset[i][0] for i in range(256)])
    
    print(f"Training: {train_data.shape[0]} samples, batch size varies by model")
    print(f"Test data: {test_data.shape}")
    
    # Setup checkpoint manager for this dataset
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    
    # Check which models need training
    models_to_train = []
    
    # Check all available models
    model_configs = [
        ("A1", create_a1_final_model, "CannedNet + T-Statistic", A1_AVAILABLE),
        ("A2", create_a2_model, "CannedNet + Signature Scoring", A2_AVAILABLE),
        ("A3", create_a3_model, "CannedNet + MMD", A3_AVAILABLE),
        ("A4", create_a4_model, "CannedNet + T-Statistic + Log Signatures", A4_AVAILABLE),
        ("B1", create_b1_model, "Neural SDE + Signature Scoring + PDE-Solved", B1_AVAILABLE),
        ("B2", create_b2_model, "Neural SDE + MMD + PDE-Solved", B2_AVAILABLE),
        ("B3", create_b3_model, "Neural SDE + T-Statistic", B3_AVAILABLE),
        ("B4", create_b4_model, "Neural SDE + MMD", B4_AVAILABLE),
        ("B5", create_b5_model, "Neural SDE + Signature Scoring", B5_AVAILABLE)
    ]
    
    for model_id, create_fn, description, available in model_configs:
        if available:
            if not retrain_all and checkpoint_manager.model_exists(model_id):
                print(f"⏭️ {model_id} already trained on {dataset_name}, skipping...")
            else:
                if retrain_all and checkpoint_manager.model_exists(model_id):
                    print(f"🔄 {model_id} exists on {dataset_name} but retraining due to --retrain-all flag")
                models_to_train.append((model_id, create_fn, description))
    
    if not models_to_train:
        print(f"\n✅ All available models already trained on {dataset_name}!")
        return True
    
    print(f"\nModels to train on {dataset_name}: {len(models_to_train)}")
    for model_id, _, description in models_to_train:
        print(f"  {model_id}: {description}")
    
    # Train each model
    training_results = []
    
    for model_id, create_model_fn, description in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_id} ({description}) on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = create_model_fn(train_data, train_data)
            print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Choose training method based on model type and memory optimization flag
            if memory_optimized and model_id.startswith('B'):
                print(f"  🧠 Using memory-optimized training for {model_id}")
                success, best_loss, best_epoch = train_model_memory_optimized(
                    model, model_id, checkpoint_manager, train_data, epochs
                )
            else:
                success, best_loss, best_epoch = train_model_standard(
                    model, model_id, checkpoint_manager, train_data, epochs, lr
                )
            
            if success:
                training_results.append({
                    'dataset': dataset_name,
                    'model_id': model_id,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'total_epochs': epochs
                })
                print(f"✅ {model_id} training completed on {dataset_name}")
            else:
                print(f"❌ {model_id} training failed on {dataset_name}")
                
        except Exception as e:
            print(f"❌ Training failed for {model_id} on {dataset_name}: {e}")
            continue
    
    # Save training summary for this dataset
    if training_results:
        summary_df = pd.DataFrame(training_results)
        summary_dir = f'results/{dataset_name}/training'
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, f'{dataset_name}_training_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nTraining summary saved to: {summary_path}")
    
    # Final status
    print(f"\nFinal Checkpoint Status for {dataset_name}:")
    checkpoint_manager.print_available_models()
    
    return True


def train_model_memory_optimized(model, model_id: str, checkpoint_manager, train_data: torch.Tensor, epochs: int):
    """Memory-optimized training for B1, B2 models with sigkernel."""
    print(f"Training {model_id} with memory optimization...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 10
    
    # Track training history
    training_losses = []
    epoch_times = []
    start_time = time.time()
    
    # Very small batch training for memory efficiency
    mini_batch_size = 4
    accumulation_steps = 8
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(train_data.shape[0])
        
        # Process in mini-batches with gradient accumulation
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for step in range(0, len(indices), mini_batch_size):
            batch_indices = indices[step:step + mini_batch_size]
            if len(batch_indices) == 0:
                continue
            
            batch_data = train_data[batch_indices]
            
            # Forward pass
            generated_output = model(batch_data)
            loss = model.compute_loss(generated_output)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            num_batches += 1
            
            # Update weights every accumulation_steps
            if (step // mini_batch_size + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear memory
                del generated_output, loss, scaled_loss
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final optimizer step if needed
        if num_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate epoch loss
        if num_batches > 0:
            epoch_loss = accumulated_loss / num_batches
        
        # Track training history
        epoch_time = time.time() - epoch_start
        training_losses.append(epoch_loss)
        epoch_times.append(epoch_time)
        
        # Save best model with training history
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Create current training history
            current_history = {
                'model_id': model_id,
                'losses': training_losses.copy(),
                'times': epoch_times.copy(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'total_time': time.time() - start_time,
                'epochs_trained': epoch + 1
            }
            
            checkpoint_manager.save_model(
                model=model,
                model_id=model_id,
                epoch=epoch + 1,
                loss=epoch_loss,
                metrics={},
                training_history=current_history
            )
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:2d}: Loss = {epoch_loss:.6f}, Best = {best_loss:.6f} (epoch {best_epoch}), Time = {epoch_time:.2f}s")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  🛑 Early stopping at epoch {epoch + 1} (patience: {patience})")
            break
    
    return True, best_loss, best_epoch


def train_model_standard(model, model_id: str, checkpoint_manager, train_data: torch.Tensor, epochs: int, lr: float):
    """Standard training for regular models."""
    import torch.utils.data as torchdata
    
    # Create data loader
    batch_size = 128  # Use consistent batch size of 128 for all models
    dataset = torchdata.TensorDataset(train_data, torch.zeros(train_data.shape[0]))  # dummy labels
    train_loader = torchdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    trainer = ModelTrainer(checkpoint_manager)
    
    try:
        history = trainer.train_with_checkpointing(
            model, model_id, train_loader, optimizer, epochs
        )
        
        best_loss = min(history['losses'])
        best_epoch = history['losses'].index(best_loss) + 1
        
        # Save final training history if not already saved
        final_history = {
            'model_id': model_id,
            'losses': history['losses'],
            'times': history.get('times', []),
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'final_loss': history.get('final_loss', best_loss),
            'total_time': history.get('total_time', 0),
            'epochs_trained': len(history['losses'])
        }
        
        # Save final model with complete history
        checkpoint_manager.save_model(
            model=model,
            model_id=model_id,
            epoch=best_epoch,
            loss=best_loss,
            metrics={},
            training_history=final_history
        )
        
        return True, best_loss, best_epoch
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False, float('inf'), 0


def main():
    """Main training function with multi-dataset support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and save signature-based models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--force", type=str, help="Force retrain specific model")
    parser.add_argument("--retrain-all", action="store_true", help="Force retrain all models (ignores existing checkpoints)")
    parser.add_argument("--dataset", type=str, help="Train on specific dataset (ou_process, heston, rbergomi, brownian, fbm_h03, fbm_h04, fbm_h06, fbm_h07)")
    parser.add_argument("--list", action="store_true", help="List available trained models")
    parser.add_argument("--memory-opt", action="store_true", help="Enable memory optimization for B-type models (slower but uses less memory)")
    
    args = parser.parse_args()
    
    if args.list:
        # List models for all datasets
        datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
        for dataset_name in datasets:
            if os.path.exists(f'results/{dataset_name}'):
                print(f"\n{dataset_name.upper()} Dataset:")
                checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
                checkpoint_manager.print_available_models()
        return
    
    if args.force:
        # Force retrain on specific dataset or all datasets
        if args.dataset:
            force_retrain_model_on_dataset(args.force, args.dataset, args.epochs, args.memory_opt)
        else:
            force_retrain_model(args.force, args.epochs)
    elif args.retrain_all:
        # Retrain all models (ignoring existing checkpoints)
        if args.dataset:
            # Retrain all models on specific dataset
            if args.dataset == 'ou_process':
                train_available_models(args.epochs, args.lr, dataset_name='ou_process', memory_optimized=args.memory_opt, retrain_all=True)
            else:
                dataset_manager = MultiDatasetManager(use_persistence=True)
                dataset_data = dataset_manager.get_dataset(args.dataset, num_samples=32768)
                train_available_models_on_dataset(args.dataset, dataset_data, args.epochs, args.lr, memory_optimized=args.memory_opt, retrain_all=True)
        else:
            # Retrain all models on all datasets
            train_all_datasets(args.epochs, args.lr, args.memory_opt, retrain_all=True)
    elif args.dataset:
        # Train on specific dataset
        if args.dataset == 'ou_process':
            train_available_models(args.epochs, args.lr, memory_optimized=args.memory_opt)
        else:
            dataset_manager = MultiDatasetManager(use_persistence=True)
            dataset_data = dataset_manager.get_dataset(args.dataset, num_samples=32768)
            train_available_models_on_dataset(args.dataset, dataset_data, args.epochs, args.lr, memory_optimized=args.memory_opt)
    else:
        # Train on all datasets (default behavior)
        train_all_datasets(args.epochs, args.lr, args.memory_opt)


def force_retrain_model_on_dataset(model_id: str, dataset_name: str, epochs: int, memory_optimized: bool = False):
    """Force retrain a model on a specific dataset."""
    print(f"Force Retraining {model_id} on {dataset_name}")
    if memory_optimized:
        print("🧠 Memory optimization enabled for B-type models")
    print("=" * 50)
    
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    
    if checkpoint_manager.model_exists(model_id):
        print(f"Deleting existing checkpoint for {model_id} on {dataset_name}...")
        checkpoint_manager.delete_model(model_id)
    
    # Train on specific dataset
    if dataset_name == 'ou_process':
        train_available_models(epochs, dataset_name='ou_process', memory_optimized=memory_optimized)
    else:
        dataset_manager = MultiDatasetManager(use_persistence=True)
        dataset_data = dataset_manager.get_dataset(dataset_name, num_samples=32768)
        train_available_models_on_dataset(dataset_name, dataset_data, epochs, memory_optimized=memory_optimized)


if __name__ == "__main__":
    main()

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
import logging
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

try:
    from models.implementations.hybrid_latent_sde.c1_latent_sde_tstat import create_c1_model
    C1_AVAILABLE = True
except ImportError:
    C1_AVAILABLE = False

try:
    from models.implementations.hybrid_latent_sde.c2_latent_sde_scoring import create_c2_model
    C2_AVAILABLE = True
except ImportError:
    C2_AVAILABLE = False

try:
    from models.implementations.hybrid_latent_sde.c3_latent_sde_mmd import create_c3_model
    C3_AVAILABLE = True
except ImportError:
    C3_AVAILABLE = False

try:
    from models.implementations.hybrid_latent_sde.c4_sde_matching_tstat import create_c4_model
    C4_AVAILABLE = True
except ImportError:
    C4_AVAILABLE = False

try:
    from models.implementations.hybrid_latent_sde.c5_sde_matching_scoring import create_c5_model
    C5_AVAILABLE = True
except ImportError:
    C5_AVAILABLE = False

try:
    from models.implementations.hybrid_latent_sde.c6_sde_matching_mmd import create_c6_model
    C6_AVAILABLE = True
except ImportError:
    C6_AVAILABLE = False

try:
    from models.implementations.d1_diffusion import create_d1_model
    D1_AVAILABLE = True
except ImportError as e:
    D1_AVAILABLE = False
    print(f"❌ D1 model import failed: {e}")
    # For debugging server issues - show key error details
    if "tsdiff" in str(e):
        print("   → This appears to be a TSDiff import issue")
        print("   → Check that relative imports are used in tsdiff modules")
    else:
        import traceback
        traceback.print_exc()

try:
    from models.latent_sde.implementations.v1_latent_sde import create_v1_model
    V1_AVAILABLE = True
except ImportError:
    V1_AVAILABLE = False

try:
    from models.sdematching.implementations.v2_sde_matching import create_v2_model
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# C1-C3 (GRU) models removed - not truly generative
# Diversity testing revealed they don't produce diverse random sample paths

# Global variables for training
TRAINING_DEVICE = torch.device('cpu')  # Default, will be set in main()
TEST_MODE_PARAMS = None  # Will be set in main() based on --test-mode flag


def setup_logging(log_dir: str, model_id: str = None, dataset_name: str = None) -> logging.Logger:
    """
    Setup logging for training sessions.
    
    Args:
        log_dir: Directory to save log files
        model_id: Model identifier (e.g., "A1", "B2")
        dataset_name: Dataset name (e.g., "ou_process", "heston")
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename
    if model_id and dataset_name:
        log_filename = f"{model_id}_{dataset_name}_training.log"
    elif model_id:
        log_filename = f"{model_id}_training.log"
    elif dataset_name:
        log_filename = f"{dataset_name}_training.log"
    else:
        log_filename = "training.log"
    
    log_path = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(f"training_{model_id}_{dataset_name}")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log setup information
    logger.info(f"Logging initialized for training session")
    logger.info(f"Log file: {log_path}")
    if model_id:
        logger.info(f"Model: {model_id}")
    if dataset_name:
        logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Training device: {TRAINING_DEVICE}")
    
    return logger


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
                # Move data to device
                data = data.to(TRAINING_DEVICE)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                
                # Compute loss (handle different model types)
                if hasattr(model, 'compute_training_loss'):
                    # Check if it's a V1/V2 model (takes only data, no output)
                    if hasattr(model, 'latent_sde') or 'V1' in str(type(model)) or 'V2' in str(type(model)):
                        # V1/V2 models: compute_training_loss(data) -> returns (loss, components)
                        loss_result = model.compute_training_loss(data)
                        if isinstance(loss_result, tuple):
                            loss = loss_result[0]  # Extract loss from tuple
                        else:
                            loss = loss_result
                    else:
                        # D1 and other models with special training loss
                        loss = model.compute_training_loss(output, data)
                else:
                    # Standard models
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


class LoggingModelTrainer(ModelTrainer):
    """
    Enhanced model trainer with logging support.
    """
    
    def __init__(self, checkpoint_manager, logger: logging.Logger = None):
        """Initialize trainer with checkpoint manager and logger."""
        super().__init__(checkpoint_manager)
        self.logger = logger or logging.getLogger(__name__)
    
    def train_with_checkpointing(self, model, model_id: str, train_loader, 
                               optimizer, num_epochs: int, 
                               save_every: int = 20, patience: int = 10):
        """
        Train model with automatic checkpointing and logging.
        """
        self.logger.info(f"Starting training for {model_id}")
        self.logger.info(f"Configuration: epochs={num_epochs}, save_every={save_every}, patience={patience}")
        self.logger.info(f"Optimizer: {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        try:
            # Call parent method
            history = super().train_with_checkpointing(
                model, model_id, train_loader, optimizer, num_epochs, save_every, patience
            )
            
            self.logger.info(f"Training completed successfully for {model_id}")
            self.logger.info(f"Best loss: {history['best_loss']:.6f} at epoch {history['best_epoch']}")
            self.logger.info(f"Final loss: {history['final_loss']:.6f}")
            self.logger.info(f"Total training time: {history['total_time']:.2f}s")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_id}: {str(e)}")
            self.logger.error(f"Exception details:", exc_info=True)
            raise


def setup_training_data(n_samples: int = None, n_points: int = None, batch_size: int = None, dataset_name: str = 'ou_process'):
    """Setup training data for all models using persistence-enabled dataset manager."""
    # Use test mode parameters if available, otherwise use provided or defaults
    if TEST_MODE_PARAMS is not None:
        n_samples = n_samples or TEST_MODE_PARAMS['num_samples']
        n_points = n_points or TEST_MODE_PARAMS['n_points']
        batch_size = batch_size or TEST_MODE_PARAMS['batch_size']
        test_samples = TEST_MODE_PARAMS['test_samples']
    else:
        n_samples = n_samples or 32768
        n_points = n_points or 64
        batch_size = batch_size or 128
        test_samples = 256
    
    print(f"Setting up training data for {dataset_name.upper()}...")
    print(f"  Samples: {n_samples:,}, Points: {n_points}, Batch: {batch_size}")
    if TEST_MODE_PARAMS is not None:
        print(f"  🧪 Test mode: Using small datasets for fast prototyping")
    
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
        # Try to get from the same size dataset as training, or generate fresh if not available
        try:
            # First try to load a dataset with the same number of samples as training
            signals_dataset = dataset_manager.get_dataset('ou_process', num_samples=n_samples, n_points=n_points)
            # Take only the first test_samples for model initialization
            signals = torch.stack([signals_dataset[i][0] for i in range(min(test_samples, len(signals_dataset)))])
        except:
            # Fallback: generate fresh signal data for model initialization
            print(f"  📊 Generating fresh OU signal data for model initialization ({test_samples} samples)")
            signals_dataset = generative_model.get_signal(num_samples=test_samples, n_points=n_points)
            signals = torch.stack([signals_dataset[i][0] for i in range(test_samples)])
        
        example_batch, _ = next(iter(torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)))
    else:
        # For other datasets, use the dataset manager
        full_dataset = dataset_manager.get_dataset(dataset_name, num_samples=n_samples, n_points=n_points)
        train_loader = torchdata.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Get signals for model initialization
        signals_dataset = dataset_manager.get_dataset(dataset_name, num_samples=test_samples, n_points=n_points)
        signals = torch.stack([signals_dataset[i][0] for i in range(min(test_samples, len(signals_dataset)))])
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
    
    # Check C1 (Hybrid Latent SDE + T-Statistic)
    if C1_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C1"):
            print(f"⏭️ C1 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C1"):
                print(f"🔄 C1 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C1", create_c1_model, "Hybrid Latent SDE + T-Statistic"))
    
    # Check C2 (Hybrid Latent SDE + Signature Scoring)
    if C2_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C2"):
            print(f"⏭️ C2 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C2"):
                print(f"🔄 C2 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C2", create_c2_model, "Hybrid Latent SDE + Signature Scoring"))
    
    # Check C3 (Hybrid Latent SDE + Signature MMD)
    if C3_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C3"):
            print(f"⏭️ C3 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C3"):
                print(f"🔄 C3 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C3", create_c3_model, "Hybrid Latent SDE + Signature MMD"))
    
    # Check C4 (Hybrid SDE Matching + T-Statistic)
    if C4_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C4"):
            print(f"⏭️ C4 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C4"):
                print(f"🔄 C4 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C4", create_c4_model, "Hybrid SDE Matching + T-Statistic"))
    
    # Check C5 (Hybrid SDE Matching + Signature Scoring)
    if C5_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C5"):
            print(f"⏭️ C5 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C5"):
                print(f"🔄 C5 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C5", create_c5_model, "Hybrid SDE Matching + Signature Scoring"))
    
    # Check C6 (Hybrid SDE Matching + Signature MMD)
    if C6_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("C6"):
            print(f"⏭️ C6 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("C6"):
                print(f"🔄 C6 exists but retraining due to --retrain-all flag")
            models_to_train.append(("C6", create_c6_model, "Hybrid SDE Matching + Signature MMD"))
    
    # Check D1 (Time Series Diffusion Model)
    if D1_AVAILABLE:
        if not retrain_all and checkpoint_manager.model_exists("D1"):
            print(f"⏭️ D1 already trained, skipping...")
        else:
            if retrain_all and checkpoint_manager.model_exists("D1"):
                print(f"🔄 D1 exists but retraining due to --retrain-all flag")
            models_to_train.append(("D1", create_d1_model, "Time Series Diffusion Model"))
    
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
            
            # Move model to training device
            model = model.to(TRAINING_DEVICE)
            
            print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"Model device: {TRAINING_DEVICE}")
            
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
    
    # Use test mode parameters if available
    dataset_samples = TEST_MODE_PARAMS['num_samples'] if TEST_MODE_PARAMS else 32768
    
    # Get all datasets (they will be loaded from disk if available, generated otherwise)
    datasets = {
        'ou_process': None,  # Use existing OU data generation
        'heston': dataset_manager.get_dataset('heston', num_samples=dataset_samples),
        'rbergomi': dataset_manager.get_dataset('rbergomi', num_samples=dataset_samples),
        'brownian': dataset_manager.get_dataset('brownian', num_samples=dataset_samples),
        'fbm_h03': dataset_manager.get_dataset('fbm_h03', num_samples=dataset_samples),
        'fbm_h04': dataset_manager.get_dataset('fbm_h04', num_samples=dataset_samples),
        'fbm_h06': dataset_manager.get_dataset('fbm_h06', num_samples=dataset_samples),
        'fbm_h07': dataset_manager.get_dataset('fbm_h07', num_samples=dataset_samples)
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
        max_train_samples = TEST_MODE_PARAMS['num_samples'] if TEST_MODE_PARAMS else 32768
        max_test_samples = TEST_MODE_PARAMS['test_samples'] if TEST_MODE_PARAMS else 256
        
        train_data = torch.stack([dataset_data[i][0] for i in range(min(max_train_samples, len(dataset_data)))])
        test_data = torch.stack([dataset_data[i][0] for i in range(min(max_test_samples, len(dataset_data)))])
    else:
        # Fallback to OU process
        max_train_samples = TEST_MODE_PARAMS['num_samples'] if TEST_MODE_PARAMS else 32768
        max_test_samples = TEST_MODE_PARAMS['test_samples'] if TEST_MODE_PARAMS else 256
        
        dataset = generative_model.get_signal(num_samples=max_train_samples)
        train_data = torch.stack([dataset[i][0] for i in range(max_train_samples)])
        test_data = torch.stack([dataset[i][0] for i in range(max_test_samples)])
    
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
        ("B5", create_b5_model, "Neural SDE + Signature Scoring", B5_AVAILABLE),
        ("C1", create_c1_model, "Hybrid Latent SDE + T-Statistic", C1_AVAILABLE),
        ("C2", create_c2_model, "Hybrid Latent SDE + Signature Scoring", C2_AVAILABLE),
        ("C3", create_c3_model, "Hybrid Latent SDE + Signature MMD", C3_AVAILABLE),
        ("C4", create_c4_model, "Hybrid SDE Matching + T-Statistic", C4_AVAILABLE),
        ("C5", create_c5_model, "Hybrid SDE Matching + Signature Scoring", C5_AVAILABLE),
        ("C6", create_c6_model, "Hybrid SDE Matching + Signature MMD", C6_AVAILABLE),
        ("D1", create_d1_model, "Time Series Diffusion Model", D1_AVAILABLE),
        ("V1", create_v1_model, "Latent SDE (TorchSDE)", V1_AVAILABLE),
        ("V2", create_v2_model, "SDE Matching", V2_AVAILABLE)
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
            
            # Move model to training device
            model = model.to(TRAINING_DEVICE)
            
            print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"Model device: {TRAINING_DEVICE}")
            
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
            
            batch_data = train_data[batch_indices].to(TRAINING_DEVICE)
            
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
    
    # Move training data to device
    train_data = train_data.to(TRAINING_DEVICE)
    
    # Create data loader
    batch_size = 128  # Use consistent batch size of 128 for all models
    dataset = torchdata.TensorDataset(train_data, torch.zeros(train_data.shape[0], device=TRAINING_DEVICE))  # dummy labels
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
    import sys
    import os
    
    # Quick environment check for debugging server issues
    if not D1_AVAILABLE:
        print("⚠️ D1 model not available - checking environment:")
        print(f"   Working directory: {os.getcwd()}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        d1_file = os.path.join(script_dir, '..', 'models', 'implementations', 'd1_diffusion.py')
        print(f"   D1 file exists: {os.path.exists(os.path.abspath(d1_file))}")
        print(f"   Python path contains src: {any('src' in p for p in sys.path)}")
    
    parser = argparse.ArgumentParser(description="Train and save signature-based models")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--force", type=str, help="Force retrain specific model")
    parser.add_argument("--retrain-all", action="store_true", help="Force retrain all models (ignores existing checkpoints)")
    parser.add_argument("--dataset", type=str, help="Train on specific dataset (ou_process, heston, rbergomi, brownian, fbm_h03, fbm_h04, fbm_h06, fbm_h07)")
    parser.add_argument("--model", type=str, help="Train only specific model (A1, A2, A3, A4, B1, B2, B3, B4, B5, C1, C2, C3, C4, C5, C6, D1)")
    parser.add_argument("--list", action="store_true", help="List available trained models")
    parser.add_argument("--memory-opt", action="store_true", help="Enable memory optimization for B-type models (slower but uses less memory)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training (auto, cpu, cuda)")
    parser.add_argument("--test-mode", action="store_true", help="Use small datasets (1000 samples) for fast prototyping and testing")
    
    args = parser.parse_args()
    
    # Configure device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Training Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Set global device for training functions
    global TRAINING_DEVICE
    TRAINING_DEVICE = device
    
    # Configure test mode
    if args.test_mode:
        print(f"🧪 Test Mode Enabled:")
        print(f"   Using small datasets (1000 samples) for fast prototyping")
        print(f"   Reduced epochs and batch sizes for quick testing")
        
        # Override parameters for test mode
        global TEST_MODE_PARAMS
        TEST_MODE_PARAMS = {
            'num_samples': 1000,
            'n_points': 64,
            'batch_size': 32,
            'test_samples': 64
        }
    else:
        TEST_MODE_PARAMS = {
            'num_samples': 32768,
            'n_points': 64, 
            'batch_size': 128,
            'test_samples': 256
        }
    
    if args.list:
        # List models for all datasets
        datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
        for dataset_name in datasets:
            if os.path.exists(f'results/{dataset_name}'):
                print(f"\n{dataset_name.upper()} Dataset:")
                checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
                checkpoint_manager.print_available_models()
        return
    
    if args.model:
        # Train single specific model
        dataset = args.dataset or 'ou_process'
        retrain = args.retrain_all or (args.force == args.model)
        
        print(f"🎯 Single Model Training Mode")
        print(f"   Model: {args.model}")
        print(f"   Dataset: {dataset}")
        if retrain:
            print(f"   Mode: Force retrain")
        
        success = train_single_model(
            model_id=args.model,
            dataset_name=dataset,
            epochs=args.epochs,
            lr=args.lr,
            memory_optimized=args.memory_opt,
            retrain=retrain
        )
        
        if success:
            print(f"\n🎉 Single model training completed successfully!")
        else:
            print(f"\n❌ Single model training failed!")
        return
    elif args.force:
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


def train_single_model(model_id: str, dataset_name: str = 'ou_process', epochs: int = 100, 
                      lr: float = 0.001, memory_optimized: bool = False, retrain: bool = False):
    """Train a single specific model on a dataset."""
    # Setup logging for single model training
    log_dir = os.path.join(f'results/{dataset_name}', 'logs')
    logger = setup_logging(log_dir, model_id, dataset_name)
    
    logger.info(f"🎯 Training Single Model: {model_id}")
    logger.info(f"   Dataset: {dataset_name}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Learning rate: {lr}")
    if memory_optimized:
        logger.info("   🧠 Memory optimization enabled")
    if retrain:
        logger.info("   🔄 Force retrain mode")
    logger.info("=" * 50)
    
    # Get model configuration
    model_configs = {
        "A1": (create_a1_final_model, "CannedNet + T-Statistic", A1_AVAILABLE),
        "A2": (create_a2_model, "CannedNet + Signature Scoring", A2_AVAILABLE),
        "A3": (create_a3_model, "CannedNet + MMD", A3_AVAILABLE),
        "A4": (create_a4_model, "CannedNet + T-Statistic + Log Signatures", A4_AVAILABLE),
        "B1": (create_b1_model, "Neural SDE + Signature Scoring + PDE-Solved", B1_AVAILABLE),
        "B2": (create_b2_model, "Neural SDE + MMD + PDE-Solved", B2_AVAILABLE),
        "B3": (create_b3_model, "Neural SDE + T-Statistic", B3_AVAILABLE),
        "B4": (create_b4_model, "Neural SDE + MMD", B4_AVAILABLE),
        "B5": (create_b5_model, "Neural SDE + Signature Scoring", B5_AVAILABLE),
        "C1": (create_c1_model, "Hybrid Latent SDE + T-Statistic", C1_AVAILABLE),
        "C2": (create_c2_model, "Hybrid Latent SDE + Signature Scoring", C2_AVAILABLE),
        "C3": (create_c3_model, "Hybrid Latent SDE + Signature MMD", C3_AVAILABLE),
        "C4": (create_c4_model, "Hybrid SDE Matching + T-Statistic", C4_AVAILABLE),
        "C5": (create_c5_model, "Hybrid SDE Matching + Signature Scoring", C5_AVAILABLE),
        "C6": (create_c6_model, "Hybrid SDE Matching + Signature MMD", C6_AVAILABLE),
        "D1": (create_d1_model, "Time Series Diffusion Model", D1_AVAILABLE),
        "V1": (create_v1_model, "Latent SDE (TorchSDE)", V1_AVAILABLE),
        "V2": (create_v2_model, "SDE Matching", V2_AVAILABLE)
    }
    
    if model_id not in model_configs:
        logger.error(f"❌ Unknown model ID: {model_id}")
        logger.error(f"Available models: {list(model_configs.keys())}")
        return False
    
    create_fn, description, available = model_configs[model_id]
    
    if not available:
        logger.error(f"❌ Model {model_id} is not available (import failed)")
        logger.error(f"Debug info: D1_AVAILABLE = {D1_AVAILABLE}")
        if model_id == "D1":
            logger.info("Attempting to re-import D1 model for debugging...")
            try:
                from models.implementations.d1_diffusion import create_d1_model as debug_d1
                logger.info("✅ D1 re-import successful!")
                logger.info(f"Function location: {debug_d1.__module__}")
            except Exception as debug_e:
                logger.error(f"❌ D1 re-import failed: {debug_e}")
                logger.error("Exception details:", exc_info=True)
        return False
    
    logger.info(f"📋 Model: {model_id} ({description})")
    
    # Setup checkpoint manager
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    
    # Check if model already exists
    if not retrain and checkpoint_manager.model_exists(model_id):
        logger.info(f"⏭️ {model_id} already trained on {dataset_name}")
        logger.info(f"   Use --retrain-all or --force to retrain")
        return True
    
    if retrain and checkpoint_manager.model_exists(model_id):
        logger.info(f"🔄 {model_id} exists but retraining due to retrain flag")
    
    # Setup training data
    try:
        if dataset_name == 'ou_process':
            train_loader, example_batch, signals = setup_training_data(dataset_name=dataset_name)
        else:
            # Setup data for other datasets
            dataset_manager = MultiDatasetManager(use_persistence=True)
            dataset_samples = TEST_MODE_PARAMS['num_samples'] if TEST_MODE_PARAMS else 32768
            test_samples = TEST_MODE_PARAMS['test_samples'] if TEST_MODE_PARAMS else 256
            
            dataset_data = dataset_manager.get_dataset(dataset_name, num_samples=dataset_samples)
            train_data = torch.stack([dataset_data[i][0] for i in range(min(dataset_samples, len(dataset_data)))])
            signals = torch.stack([dataset_data[i][0] for i in range(min(test_samples, len(dataset_data)))])
            example_batch = train_data[:32]  # Use first 32 samples as example batch
            
            # Create data loader
            batch_size = TEST_MODE_PARAMS['batch_size'] if TEST_MODE_PARAMS else 128
            dataset_tensor = torch.utils.data.TensorDataset(train_data, torch.zeros(train_data.shape[0]))
            train_loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)
            
            logger.info(f"Training: {train_data.shape[0]} samples, batch size {batch_size}")
            logger.info(f"Test data: {signals.shape}")
    
    except Exception as e:
        logger.error(f"❌ Failed to setup training data: {e}")
        logger.error("Exception details:", exc_info=True)
        return False
    
    # Create and train model
    try:
        logger.info(f"\n🏗️ Creating {model_id} model...")
        torch.manual_seed(12345)  # Consistent seed
        model = create_fn(example_batch, signals)
        
        # Move model to training device
        model = model.to(TRAINING_DEVICE)
        
        logger.info(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"   Model device: {TRAINING_DEVICE}")
        
        # Setup trainer with logging
        trainer = LoggingModelTrainer(checkpoint_manager, logger)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        logger.info(f"\n🚀 Starting training...")
        
        # Choose training method
        if memory_optimized and model_id.startswith('B'):
            logger.info(f"🧠 Using memory-optimized training for {model_id}")
            train_data = signals if dataset_name == 'ou_process' else train_data
            success, best_loss, best_epoch = train_model_memory_optimized(
                model, model_id, checkpoint_manager, train_data, epochs
            )
            if success:
                logger.info(f"✅ {model_id} training completed!")
                logger.info(f"   Best loss: {best_loss:.6f} at epoch {best_epoch}")
            else:
                logger.error(f"❌ {model_id} training failed")
        else:
            # Standard training with checkpointing
            history = trainer.train_with_checkpointing(
                model=model,
                model_id=model_id,
                train_loader=train_loader,
                optimizer=optimizer,
                num_epochs=epochs,
                save_every=25,
                patience=10
            )
            
            logger.info(f"✅ {model_id} training completed!")
            logger.info(f"   Best loss: {history['best_loss']:.6f} at epoch {history['best_epoch']}")
            logger.info(f"   Final loss: {history['final_loss']:.6f}")
            logger.info(f"   Total time: {history['total_time']:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed for {model_id}: {e}")
        logger.error("Exception details:", exc_info=True)
        return False


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

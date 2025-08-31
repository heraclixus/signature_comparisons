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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path  
from scipy import stats
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from dataset.multi_dataset import MultiDatasetManager
from utils.model_checkpoint import create_checkpoint_manager
from utils.model_trainer import ModelTrainer, LoggingModelTrainer, train_model_memory_optimized
from utils.training_visualization import create_final_training_summary
from utils.model_registry import check_model_availability, get_model_configs, print_model_availability_summary, get_models_to_train

# Check model availability
MODEL_AVAILABILITY, MODEL_CREATORS = check_model_availability()
MODEL_CONFIGS = get_model_configs()

# Print availability summary for debugging
print_model_availability_summary(MODEL_AVAILABILITY)

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


# ModelTrainer classes moved to utils.model_trainer


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
                          memory_optimized: bool = False, retrain_all: bool = False,
                          enable_trajectory_viz: bool = True, viz_every: int = 10):
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
    
    # Get models to train using registry
    models_to_train = get_models_to_train(MODEL_AVAILABILITY, MODEL_CREATORS, checkpoint_manager, retrain_all)
    
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
                    model, model_id, checkpoint_manager, train_data, num_epochs, TRAINING_DEVICE
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
                # Use reduced batch size for D2/D3 models to avoid sigkernel thread limit
                current_train_loader = train_loader
                if model_id in ['D2', 'D3']:
                    print(f"  🔧 Using reduced batch size for {model_id} to avoid sigkernel thread limit")
                    # Get original batch size and halve it
                    original_batch_size = train_loader.batch_size
                    reduced_batch_size = max(8, original_batch_size // 2)  # Minimum batch size of 8
                    print(f"     Batch size: {original_batch_size} → {reduced_batch_size}")
                    
                    # Create new DataLoader with reduced batch size
                    current_train_loader = torch.utils.data.DataLoader(
                        train_loader.dataset, 
                        batch_size=reduced_batch_size, 
                        shuffle=True, 
                        num_workers=0
                    )
                
                history = trainer.train_with_checkpointing(
                    model=model,
                    model_id=model_id,
                    train_loader=current_train_loader,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    save_every=25,
                    patience=10,
                    enable_trajectory_viz=enable_trajectory_viz,
                    viz_every=viz_every
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


def train_all_datasets(epochs: int = 100, lr: float = 0.001, memory_optimized: bool = False, retrain_all: bool = False,
                      enable_trajectory_viz: bool = True, viz_every: int = 10):
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
            train_available_models(epochs, lr, dataset_name='ou_process', memory_optimized=memory_optimized, retrain_all=retrain_all,
                                 enable_trajectory_viz=enable_trajectory_viz, viz_every=viz_every)
        else:
            # Train on new dataset
            train_available_models_on_dataset(dataset_name, dataset_data, epochs, lr, memory_optimized=memory_optimized, retrain_all=retrain_all,
                                            enable_trajectory_viz=enable_trajectory_viz, viz_every=viz_every)


def train_available_models_on_dataset(dataset_name: str, dataset_data, epochs: int = 100, lr: float = 0.001, 
                                     memory_optimized: bool = False, retrain_all: bool = False,
                                     enable_trajectory_viz: bool = True, viz_every: int = 10):
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
    
    # Get models to train using registry
    models_to_train = get_models_to_train(MODEL_AVAILABILITY, MODEL_CREATORS, checkpoint_manager, retrain_all)
    
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
                    model, model_id, checkpoint_manager, train_data, epochs, TRAINING_DEVICE
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


# Memory-optimized training moved to utils.model_trainer

def train_model_standard(model, model_id: str, checkpoint_manager, train_data: torch.Tensor, epochs: int, lr: float):
    """Standard training for regular models."""
    import torch.utils.data as torchdata
    
    # Move training data to device
    train_data = train_data.to(TRAINING_DEVICE)
    
    # Create data loader
    batch_size = 128  # Use consistent batch size of 128 for all models
    # Reduce batch size for D2/D3 models to avoid sigkernel thread limit
    if model_id in ['D2', 'D3']:
        original_batch_size = batch_size
        batch_size = max(8, batch_size // 2)  # Minimum batch size of 8
        print(f"  🔧 Reducing batch size for {model_id}: {original_batch_size} → {batch_size}")
    
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
    if not MODEL_AVAILABILITY.get('D1', False):
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
    parser.add_argument("--no-viz", action="store_true", help="Disable trajectory visualization during training")
    parser.add_argument("--viz-every", type=int, default=10, help="Create trajectory visualization every N epochs (default: 10)")
    
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
                train_available_models(args.epochs, args.lr, dataset_name='ou_process', memory_optimized=args.memory_opt, retrain_all=True,
                                      enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)
            else:
                dataset_manager = MultiDatasetManager(use_persistence=True)
                dataset_data = dataset_manager.get_dataset(args.dataset, num_samples=32768)
                train_available_models_on_dataset(args.dataset, dataset_data, args.epochs, args.lr, memory_optimized=args.memory_opt, retrain_all=True,
                                                  enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)
        else:
            # Retrain all models on all datasets
            train_all_datasets(args.epochs, args.lr, args.memory_opt, retrain_all=True,
                              enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)
    elif args.dataset:
        # Train on specific dataset
        if args.dataset == 'ou_process':
            train_available_models(args.epochs, args.lr, memory_optimized=args.memory_opt,
                                 enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)
        else:
            dataset_manager = MultiDatasetManager(use_persistence=True)
            dataset_data = dataset_manager.get_dataset(args.dataset, num_samples=32768)
            train_available_models_on_dataset(args.dataset, dataset_data, args.epochs, args.lr, memory_optimized=args.memory_opt,
                                            enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)
    else:
        # Train on all datasets (default behavior)
        train_all_datasets(args.epochs, args.lr, args.memory_opt,
                          enable_trajectory_viz=not args.no_viz, viz_every=args.viz_every)


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
    
    # Get model configuration from registry
    if model_id not in MODEL_AVAILABILITY:
        logger.error(f"❌ Unknown model ID: {model_id}")
        logger.error(f"Available models: {list(MODEL_AVAILABILITY.keys())}")
        return False
    
    if not MODEL_AVAILABILITY[model_id]:
        logger.error(f"❌ Model {model_id} is not available (import failed)")
        return False
    
    create_fn = MODEL_CREATORS[model_id]
    description = MODEL_CONFIGS[model_id]
    
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
        # Reduce batch size for D2/D3 models to avoid sigkernel thread limit
        if model_id in ['D2', 'D3']:
            logger.info(f"🔧 Reducing batch size for {model_id} model to avoid sigkernel thread limit")
            
        if dataset_name == 'ou_process':
            # For OU process, modify batch size in setup_training_data call
            if model_id in ['D2', 'D3']:
                # Get default batch size and halve it
                default_batch_size = TEST_MODE_PARAMS['batch_size'] if TEST_MODE_PARAMS else 128
                reduced_batch_size = max(8, default_batch_size // 2)  # Minimum batch size of 8
                logger.info(f"   Batch size: {default_batch_size} → {reduced_batch_size}")
                train_loader, example_batch, signals = setup_training_data(dataset_name=dataset_name, batch_size=reduced_batch_size)
            else:
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
            
            # Create data loader with reduced batch size for D2/D3
            batch_size = TEST_MODE_PARAMS['batch_size'] if TEST_MODE_PARAMS else 128
            if model_id in ['D2', 'D3']:
                original_batch_size = batch_size
                batch_size = max(8, batch_size // 2)  # Minimum batch size of 8
                logger.info(f"   Batch size: {original_batch_size} → {batch_size}")
            
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
                model, model_id, checkpoint_manager, train_data, epochs, TRAINING_DEVICE
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

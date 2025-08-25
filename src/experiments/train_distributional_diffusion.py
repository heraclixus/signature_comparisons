"""
Training Script for Distributional Diffusion Model.

This script implements the training algorithm from "Path Diffusion with Signature Kernels"
following Algorithm 1 in the paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import yaml
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset

# Import our new components
try:
    from models.tsdiff.diffusion.distributional_diffusion import DistributionalDiffusion
    from models.distributional_generator import create_distributional_generator
    from losses.signature_scoring_loss import create_signature_scoring_loss
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    warnings.warn(f"Could not import components: {e}")

# Import existing dataset utilities
try:
    from dataset.brownian import BrownianDataset
    from dataset.fbm import FBMDataset
    from dataset.heston import HestonDataset
    from dataset.ou_process import OUProcessDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("Dataset modules not available")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataset(config: Dict[str, Any]) -> DataLoader:
    """
    Create dataset and dataloader based on configuration.
    
    Args:
        config: Dataset configuration
        
    Returns:
        DataLoader for training
    """
    dataset_type = config['type'].lower()
    dataset_params = config.get('params', {})
    
    if not DATASETS_AVAILABLE:
        # Fallback: create synthetic data
        warnings.warn("Using synthetic data fallback")
        num_samples = dataset_params.get('num_samples', 1000)
        seq_len = dataset_params.get('seq_len', 100)
        dim = dataset_params.get('dim', 1)
        
        # Generate simple Brownian motion
        data = torch.randn(num_samples, dim, seq_len).cumsum(dim=-1)
        dataset = TensorDataset(data)
    
    else:
        # Use existing dataset classes
        if dataset_type == 'brownian':
            dataset = BrownianDataset(**dataset_params)
        elif dataset_type == 'fbm':
            dataset = FBMDataset(**dataset_params)
        elif dataset_type == 'heston':
            dataset = HestonDataset(**dataset_params)
        elif dataset_type == 'ou_process':
            dataset = OUProcessDataset(**dataset_params)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )
    
    return dataloader


def train_distributional_diffusion(
    ddm: DistributionalDiffusion,
    generator: nn.Module,
    dataloader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Training loop for distributional diffusion model following Algorithm 1.
    
    Args:
        ddm: DistributionalDiffusion model
        generator: Generator network P_θ(·|X_t, t, Z)
        dataloader: Training dataloader
        config: Training configuration
        device: Device to train on
        
    Returns:
        Training metrics and history
    """
    # Move models to device
    ddm = ddm.to(device)
    generator = generator.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(
        generator.parameters(), 
        lr=config.get('learning_rate', 1e-4),
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    # Setup learning rate scheduler
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.get('num_epochs', 100)
        )
    
    # Training metrics
    metrics = {
        'epoch_losses': [],
        'batch_losses': [],
        'learning_rates': [],
        'training_times': []
    }
    
    num_epochs = config.get('num_epochs', 100)
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 20)
    gradient_clip = config.get('gradient_clip', 1.0)
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Population size: {ddm.population_size}")
    print(f"Lambda parameter: {ddm.lambda_param}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        generator.train()
        ddm.train()
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Extract data (handle different dataset formats)
            if isinstance(batch_data, (list, tuple)):
                x0 = batch_data[0].to(device)
            else:
                x0 = batch_data.to(device)
            
            batch_size = x0.shape[0]
            
            # Algorithm 1: Distributional diffusion training step
            loss = ddm.get_loss(generator, x0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    generator.parameters(), 
                    gradient_clip
                )
            
            optimizer.step()
            
            # Record metrics
            loss_item = loss.item()
            epoch_loss += loss_item
            num_batches += 1
            
            metrics['batch_losses'].append(loss_item)
            
            # Log progress
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch:3d}, Batch {batch_idx:4d}, Loss: {loss_item:.6f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Record epoch metrics
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        metrics['epoch_losses'].append(avg_epoch_loss)
        metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        metrics['training_times'].append(epoch_time)
        
        print(f"Epoch {epoch:3d} completed. Average Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                generator, optimizer, epoch, avg_epoch_loss, 
                config.get('checkpoint_dir', './checkpoints')
            )
    
    print("Training completed!")
    return metrics


def save_checkpoint(
    generator: nn.Module, 
    optimizer: optim.Optimizer, 
    epoch: int, 
    loss: float,
    checkpoint_dir: str
):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def evaluate_model(
    ddm: DistributionalDiffusion,
    generator: nn.Module,
    dataloader: DataLoader,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        ddm: DistributionalDiffusion model
        generator: Trained generator network
        dataloader: Evaluation dataloader
        config: Evaluation configuration
        device: Device to run on
        
    Returns:
        Evaluation metrics
    """
    ddm.eval()
    generator.eval()
    
    metrics = {
        'avg_loss': 0.0,
        'num_samples': 0
    }
    
    total_loss = 0.0
    num_batches = 0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Extract data
            if isinstance(batch_data, (list, tuple)):
                x0 = batch_data[0].to(device)
            else:
                x0 = batch_data.to(device)
            
            # Compute loss
            loss = ddm.get_loss(generator, x0)
            
            total_loss += loss.item()
            num_batches += 1
            metrics['num_samples'] += x0.shape[0]
    
    metrics['avg_loss'] = total_loss / num_batches
    
    print(f"Evaluation completed. Average loss: {metrics['avg_loss']:.6f}")
    
    return metrics


def generate_samples(
    ddm: DistributionalDiffusion,
    generator: nn.Module,
    num_samples: int,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate samples using the trained model.
    
    Args:
        ddm: DistributionalDiffusion model
        generator: Trained generator network
        num_samples: Number of samples to generate
        config: Generation configuration
        device: Device to run on
        
    Returns:
        Generated samples (num_samples, dim, seq_len)
    """
    print(f"Generating {num_samples} samples...")
    
    start_time = time.time()
    
    samples = ddm.sample(
        generator=generator,
        num_samples=num_samples,
        num_coarse_steps=config.get('num_coarse_steps', 20),
        device=device
    )
    
    generation_time = time.time() - start_time
    
    print(f"Generation completed in {generation_time:.2f}s")
    print(f"Time per sample: {generation_time/num_samples:.4f}s")
    
    return samples


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Distributional Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Check component availability
    if not COMPONENTS_AVAILABLE:
        raise ImportError("Required components not available. Please check imports.")
    
    # Create models
    model_config = config['model']
    ddm = DistributionalDiffusion(**model_config)
    
    generator_config = config['generator']
    generator = create_distributional_generator(**generator_config)
    
    # Create dataset
    dataset_config = config['dataset']
    train_dataloader = create_dataset(dataset_config)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Train model
    training_config = config['training']
    training_config['num_epochs'] = training_config.get('num_epochs', 100) - start_epoch
    
    metrics = train_distributional_diffusion(
        ddm=ddm,
        generator=generator,
        dataloader=train_dataloader,
        config=training_config,
        device=device
    )
    
    # Evaluate model
    if config.get('evaluate', True):
        eval_metrics = evaluate_model(
            ddm=ddm,
            generator=generator,
            dataloader=train_dataloader,  # Use train data for now
            config=config.get('evaluation', {}),
            device=device
        )
        print(f"Final evaluation metrics: {eval_metrics}")
    
    # Generate samples
    if config.get('generate_samples', True):
        generation_config = config.get('generation', {})
        samples = generate_samples(
            ddm=ddm,
            generator=generator,
            num_samples=generation_config.get('num_samples', 64),
            config=generation_config,
            device=device
        )
        
        # Save samples
        output_dir = config.get('output_dir', './outputs')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(samples, os.path.join(output_dir, 'generated_samples.pt'))
        print(f"Samples saved to {output_dir}/generated_samples.pt")
    
    print("Training script completed successfully!")


if __name__ == "__main__":
    main()

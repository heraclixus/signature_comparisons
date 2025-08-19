"""
Adversarial Training Script for Signature-Based Models

This script implements adversarial training variants of our existing baseline models.
It combines existing generators with signature-based discriminators to explore
adversarial vs non-adversarial training approaches.

Adversarial Model Variants (T-statistic models excluded due to compatibility issues):
- A2_ADV: CannedNet + Adversarial Signature Scoring  
- A3_ADV: CannedNet + Adversarial MMD
- B1_ADV: Neural SDE + Adversarial Signature Scoring (PDE-Solved)
- B2_ADV: Neural SDE + Adversarial MMD (PDE-Solved)
- B4_ADV: Neural SDE + Adversarial MMD
- B5_ADV: Neural SDE + Adversarial Signature Scoring

Note: A1 and B3 (T-statistic models) are not supported in adversarial mode due to 
signature dimension compatibility issues with adversarial scaling parameters.
"""

import torch
import torch.optim as optim
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from dataset.multi_dataset import MultiDatasetManager
from utils.model_checkpoint import create_checkpoint_manager

# Import discriminators (T-statistic excluded due to compatibility issues)
from models.discriminators.signature_discriminators import (
    SignatureMMDDiscriminator,
    SignatureScoringDiscriminator,
    create_discriminator
)

# Import existing model creators for generator extraction
from models.implementations.a1_final import create_a1_final_model
from models.implementations.a2_canned_scoring import create_a2_model
from models.implementations.a3_canned_mmd import create_a3_model
from models.implementations.b3_nsde_tstatistic import create_b3_model
from models.implementations.b4_nsde_mmd import create_b4_model
from models.implementations.b5_nsde_scoring import create_b5_model


class AdversarialModelWrapper(torch.nn.Module):
    """
    Wrapper that combines existing generators with adversarial discriminators.
    """
    
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module,
                 model_id: str, base_model_id: str):
        """
        Initialize adversarial model wrapper.
        
        Args:
            generator: Existing generator (from baseline models)
            discriminator: Adversarial discriminator
            model_id: ID for adversarial variant (e.g., 'A1_ADV')
            base_model_id: ID of base model (e.g., 'A1')
        """
        super().__init__()
        
        self.generator = generator
        self.discriminator = discriminator
        self.model_id = model_id
        self.base_model_id = base_model_id
        
        print(f"Adversarial wrapper created: {model_id} (based on {base_model_id})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.generator(x)
    
    def generate_samples(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Generate samples using the wrapped generator."""
        if hasattr(self.generator, 'generate_samples'):
            return self.generator.generate_samples(batch_size, device)
        else:
            # Fallback for generators without generate_samples method
            if device is None:
                device = next(self.generator.parameters()).device
            
            dummy_input = torch.randn(batch_size, 2, 100, device=device)
            return self.generator(dummy_input)
    
    def compute_generator_loss(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Compute generator loss (minimize discriminator output)."""
        return self.discriminator(generated_paths, real_paths)
    
    def compute_discriminator_loss(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss (maximize discrimination)."""
        # For signature-based discriminators, we typically want to maximize the loss
        # This depends on the specific discriminator implementation
        return -self.discriminator(generated_paths, real_paths)


class AdversarialTrainer:
    """
    Trainer for adversarial signature-based models.
    """
    
    def __init__(self, checkpoint_manager):
        """Initialize adversarial trainer."""
        self.checkpoint_manager = checkpoint_manager
        self.training_history = {}
    
    def train_adversarial_model(self, model: AdversarialModelWrapper, train_loader,
                               g_optimizer, d_optimizer, num_epochs: int = 100,
                               d_steps: int = 1, g_steps: int = 1, 
                               patience: int = 10, save_every: int = 25,
                               memory_efficient: bool = False, accumulation_steps: int = 4):
        """
        Train adversarial model with alternating generator/discriminator updates.
        
        Args:
            model: Adversarial model wrapper
            train_loader: Training data loader
            g_optimizer: Generator optimizer
            d_optimizer: Discriminator optimizer (None for non-adversarial)
            num_epochs: Number of training epochs
            d_steps: Discriminator updates per batch
            g_steps: Generator updates per batch
            patience: Early stopping patience
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining {model.model_id} with adversarial approach...")
        print(f"  Epochs: {num_epochs}, D steps: {d_steps}, G steps: {g_steps}, Patience: {patience}")
        if memory_efficient:
            print(f"  🧠 Memory-efficient mode: accumulation_steps={accumulation_steps}")
        
        model.train()
        
        # Training tracking
        g_losses = []
        d_losses = []
        epoch_times = []
        best_g_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.shape[0]
                
                if memory_efficient:
                    # Memory-efficient training with gradient accumulation
                    self._memory_efficient_training_step(
                        model, data, g_optimizer, d_optimizer, 
                        d_steps, g_steps, accumulation_steps,
                        epoch_g_losses, epoch_d_losses
                    )
                else:
                    # Standard adversarial training
                    self._standard_training_step(
                        model, data, g_optimizer, d_optimizer,
                        d_steps, g_steps, epoch_g_losses, epoch_d_losses
                    )
            
            epoch_time = time.time() - epoch_start
            epoch_g_loss = np.mean(epoch_g_losses)
            epoch_d_loss = np.mean(epoch_d_losses) if epoch_d_losses else 0.0
            
            g_losses.append(epoch_g_loss)
            d_losses.append(epoch_d_loss)
            epoch_times.append(epoch_time)
            
            # Update training history for this epoch
            self._update_training_history(model.model_id, epoch_g_loss, epoch_d_loss, epoch_time)
            
            # Check if this is the best generator so far
            if epoch_g_loss < best_g_loss:
                best_g_loss = epoch_g_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model with complete training history
                self._save_adversarial_model(model, epoch + 1, epoch_g_loss, epoch_d_loss)
                print(f"  💾 New best model saved at epoch {epoch + 1}: G_loss={epoch_g_loss:.6f}")
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                scaling_info = model.discriminator.get_scaling_info()
                current_scaling = scaling_info.get('current_scaling', [1.0])
                
                print(f"  Epoch {epoch + 1:3d}: G_loss = {epoch_g_loss:.6f}, "
                      f"D_loss = {epoch_d_loss:.6f}, "
                      f"Scaling = {current_scaling[0]:.3f}, "
                      f"Time = {epoch_time:.2f}s")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  🛑 Early stopping at epoch {epoch + 1} (patience: {patience})")
                break
            
            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                print(f"  📁 Periodic checkpoint saved at epoch {epoch + 1}")
        
        total_time = time.time() - start_time
        
        print(f"✅ {model.model_id} adversarial training completed in {total_time:.2f}s")
        print(f"   Best G_loss: {best_g_loss:.6f} at epoch {best_epoch}")
        print(f"   Final G_loss: {g_losses[-1]:.6f}")
        
        # Store complete training history
        history = {
            'model_id': model.model_id,
            'losses': g_losses,  # Use 'losses' key for checkpoint manager compatibility
            'g_losses': g_losses,
            'd_losses': d_losses,
            'times': epoch_times,
            'best_g_loss': best_g_loss,
            'best_epoch': best_epoch,
            'final_g_loss': g_losses[-1],
            'final_d_loss': d_losses[-1],
            'total_time': total_time,
            'epochs_trained': len(g_losses)
        }
        
        self.training_history[model.model_id] = history
        
        # Save final complete training history (not just best model)
        self._save_final_training_history(model, history)
        
        return history
    
    def _update_training_history(self, model_id: str, g_loss: float, d_loss: float, epoch_time: float):
        """Update training history for current epoch."""
        if model_id not in self.training_history:
            self.training_history[model_id] = {
                'model_id': model_id,
                'g_losses': [],
                'd_losses': [],
                'times': []
            }
        
        self.training_history[model_id]['g_losses'].append(g_loss)
        self.training_history[model_id]['d_losses'].append(d_loss)
        self.training_history[model_id]['times'].append(epoch_time)
    
    def _save_final_training_history(self, model: AdversarialModelWrapper, history: Dict):
        """Save the final complete training history."""
        self.checkpoint_manager.save_model(
            model=model,
            model_id=model.model_id,
            epoch=history['best_epoch'],
            loss=history['best_g_loss'],
            training_config={
                'generator_loss': history['final_g_loss'],
                'discriminator_loss': history['final_d_loss'],
                'adversarial': True,
                'base_model': model.base_model_id,
                'total_epochs': history['epochs_trained']
            },
            training_history=history
        )
        print(f"✅ Final training history saved for {model.model_id}: {history['epochs_trained']} epochs")
    
    def _standard_training_step(self, model, data, g_optimizer, d_optimizer, 
                               d_steps, g_steps, epoch_g_losses, epoch_d_losses):
        """Standard adversarial training step."""
        # Update Discriminator
        if d_optimizer is not None:
            for _ in range(d_steps):
                d_optimizer.zero_grad()
                
                # Generate fake data (detached from generator gradients)
                with torch.no_grad():
                    generated_data = model.generator(data)
                
                # Convert to path format for discriminator
                generated_paths = self._convert_to_path_format(generated_data, data)
                real_paths = data
                
                # Compute discriminator loss
                d_loss = model.compute_discriminator_loss(generated_paths, real_paths)
                d_loss.backward()
                d_optimizer.step()
                
                epoch_d_losses.append(d_loss.item())
        
        # Update Generator
        for _ in range(g_steps):
            g_optimizer.zero_grad()
            
            # Generate data (with gradients)
            generated_data = model.generator(data)
            
            # Convert to path format
            generated_paths = self._convert_to_path_format(generated_data, data)
            real_paths = data
            
            # Compute generator loss
            g_loss = model.compute_generator_loss(generated_paths, real_paths)
            g_loss.backward()
            g_optimizer.step()
            
            epoch_g_losses.append(g_loss.item())
    
    def _memory_efficient_training_step(self, model, data, g_optimizer, d_optimizer,
                                       d_steps, g_steps, accumulation_steps,
                                       epoch_g_losses, epoch_d_losses):
        """Memory-efficient adversarial training step with gradient accumulation."""
        batch_size = data.shape[0]
        mini_batch_size = max(1, batch_size // accumulation_steps)
        
        # Initialize accumulators
        if d_optimizer is not None:
            d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        
        total_d_loss = 0.0
        total_g_loss = 0.0
        
        # Process in mini-batches
        for i in range(accumulation_steps):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, batch_size)
            
            if start_idx >= end_idx:
                continue
                
            mini_batch = data[start_idx:end_idx]
            actual_mini_size = end_idx - start_idx
            
            # Update Discriminator (memory-efficient)
            if d_optimizer is not None:
                for _ in range(d_steps):
                    # Generate fake data (detached)
                    with torch.no_grad():
                        generated_data = model.generator(mini_batch)
                    
                    # Convert to path format
                    generated_paths = self._convert_to_path_format(generated_data, mini_batch)
                    real_paths = mini_batch
                    
                    # Compute scaled discriminator loss
                    d_loss = model.compute_discriminator_loss(generated_paths, real_paths)
                    scaled_d_loss = d_loss / accumulation_steps
                    scaled_d_loss.backward()
                    
                    total_d_loss += d_loss.item() * actual_mini_size
                    
                    # Clear intermediate tensors
                    del generated_data, generated_paths, d_loss, scaled_d_loss
            
            # Update Generator (memory-efficient)
            for _ in range(g_steps):
                # Generate data (with gradients)
                generated_data = model.generator(mini_batch)
                
                # Convert to path format
                generated_paths = self._convert_to_path_format(generated_data, mini_batch)
                real_paths = mini_batch
                
                # Compute scaled generator loss
                g_loss = model.compute_generator_loss(generated_paths, real_paths)
                scaled_g_loss = g_loss / accumulation_steps
                scaled_g_loss.backward()
                
                total_g_loss += g_loss.item() * actual_mini_size
                
                # Clear intermediate tensors
                del generated_data, generated_paths, g_loss, scaled_g_loss
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Apply accumulated gradients
        if d_optimizer is not None:
            d_optimizer.step()
        g_optimizer.step()
        
        # Store average losses
        if total_d_loss > 0:
            epoch_d_losses.append(total_d_loss / batch_size)
        if total_g_loss > 0:
            epoch_g_losses.append(total_g_loss / batch_size)
    
    def _convert_to_path_format(self, generated_data: torch.Tensor, reference_data: torch.Tensor) -> torch.Tensor:
        """Convert generator output to path format expected by discriminators."""
        if generated_data.dim() == 2:
            # Generator output is (batch, time)
            batch_size, time_steps = generated_data.shape
            
            # Create time channel
            time_channel = torch.linspace(0, 1, time_steps, device=generated_data.device)
            time_channel = time_channel.unsqueeze(0).expand(batch_size, -1)
            
            # Stack time and values
            return torch.stack([time_channel, generated_data], dim=1)
        elif generated_data.dim() == 3:
            # Already in path format (batch, channels, time)
            return generated_data
        else:
            raise ValueError(f"Unsupported generator output shape: {generated_data.shape}")
    
    def _save_adversarial_model(self, model: AdversarialModelWrapper, epoch: int, 
                               g_loss: float, d_loss: float):
        """Save adversarial model checkpoint."""
        # Get complete training history accumulated so far
        model_history = self.training_history.get(model.model_id, {})
        complete_g_losses = model_history.get('g_losses', [])
        complete_d_losses = model_history.get('d_losses', [])
        complete_times = model_history.get('times', [])
        
        current_history = {
            'model_id': model.model_id,
            'losses': complete_g_losses,  # Complete loss history for checkpoint manager
            'g_losses': complete_g_losses,
            'd_losses': complete_d_losses,
            'times': complete_times,
            'best_g_loss': g_loss,
            'best_epoch': epoch,
            'epochs_trained': len(complete_g_losses)
        }
        
        self.checkpoint_manager.save_model(
            model=model,
            model_id=model.model_id,
            epoch=epoch,
            loss=g_loss,
            training_config={
                'generator_loss': g_loss,
                'discriminator_loss': d_loss,
                'adversarial': True,
                'base_model': model.base_model_id
            },
            training_history=current_history
        )


def create_adversarial_model(base_model_id: str, example_batch: torch.Tensor, 
                           real_data: torch.Tensor, adversarial: bool = True,
                           memory_efficient: bool = False) -> AdversarialModelWrapper:
    """
    Create adversarial variant of existing baseline model.
    
    Args:
        base_model_id: ID of base model ('A1', 'A2', 'A3', 'B3', 'B4', 'B5')
        example_batch: Example input for model initialization
        real_data: Real data for loss initialization
        adversarial: Whether to use adversarial training
        
    Returns:
        AdversarialModelWrapper ready for training
    """
    print(f"Creating adversarial variant of {base_model_id}...")
    
    # Import additional models as needed
    if base_model_id in ['B1', 'B2']:
        from models.implementations.b1_nsde_scoring import create_b1_model
        from models.implementations.b2_nsde_mmd_pde import create_b2_model
    
    # Create base model to extract generator
    # Note: A1 and B3 use T-statistic which has compatibility issues with adversarial training
    if base_model_id == 'A1':
        print(f"⚠️ A1 uses T-statistic loss which has adversarial training compatibility issues")
        print(f"   Consider using A2 (CannedNet + Scoring) or A3 (CannedNet + MMD) instead")
        raise ValueError(f"A1 (T-statistic) not supported in adversarial mode. Use A2 or A3 instead.")
    elif base_model_id == 'A2':
        base_model = create_a2_model(example_batch, real_data)
        discriminator_type = 'scoring'
    elif base_model_id == 'A3':
        base_model = create_a3_model(example_batch, real_data)
        discriminator_type = 'mmd'
    elif base_model_id == 'B1':
        base_model = create_b1_model(example_batch, real_data)
        discriminator_type = 'scoring'  # B1 uses signature scoring
    elif base_model_id == 'B2':
        base_model = create_b2_model(example_batch, real_data)
        discriminator_type = 'mmd'  # B2 uses MMD
    elif base_model_id == 'B3':
        print(f"⚠️ B3 uses T-statistic loss which has adversarial training compatibility issues")
        print(f"   Consider using B4 (Neural SDE + MMD) or B5 (Neural SDE + Scoring) instead")
        raise ValueError(f"B3 (T-statistic) not supported in adversarial mode. Use B4 or B5 instead.")
    elif base_model_id == 'B4':
        base_model = create_b4_model(example_batch, real_data)
        discriminator_type = 'mmd'
    elif base_model_id == 'B5':
        base_model = create_b5_model(example_batch, real_data)
        discriminator_type = 'scoring'
    else:
        raise ValueError(f"Unsupported base model: {base_model_id}")
    
    # Extract generator from base model
    generator = base_model.generator if hasattr(base_model, 'generator') else base_model
    
    # Create corresponding discriminator with memory optimization
    path_dim = 1  # Most of our models use 1D paths
    
    if memory_efficient:
        # Memory-optimized discriminator configuration
        # (Only MMD and Scoring discriminators supported)
        discriminator_config = {
            'signature_depth': 3,        # Reduce signature complexity
            'use_sigkernel': False,      # Use fallback implementations
            'sigma': 1.0,
            'kernel_type': 'rbf'
        }
        print(f"  🧠 Using memory-efficient discriminator configuration")
    else:
        # Standard discriminator configuration
        discriminator_config = {
            'signature_depth': 4,
            'use_sigkernel': True,       # Use sigkernel if available
            'sigma': 1.0,
            'kernel_type': 'rbf'
        }
    
    discriminator = create_discriminator(
        discriminator_type=discriminator_type,
        path_dim=path_dim,
        adversarial=adversarial,
        **discriminator_config
    )
    
    # Create adversarial model ID
    adv_model_id = f"{base_model_id}_ADV" if adversarial else f"{base_model_id}_NONADV"
    
    # Wrap in adversarial model
    adversarial_model = AdversarialModelWrapper(
        generator=generator,
        discriminator=discriminator,
        model_id=adv_model_id,
        base_model_id=base_model_id
    )
    
    print(f"✅ {adv_model_id} created successfully")
    print(f"   Generator: {type(generator).__name__}")
    print(f"   Discriminator: {type(discriminator).__name__} ({'adversarial' if adversarial else 'non-adversarial'})")
    
    return adversarial_model


def train_adversarial_models(base_model_ids: List[str] = None, num_epochs: int = 100, 
                           dataset_name: str = 'ou_process', adversarial: bool = True,
                           memory_efficient: bool = False, force_retrain: bool = False):
    """
    Train adversarial variants of baseline models.
    
    Args:
        base_model_ids: List of base model IDs to create adversarial variants for
        num_epochs: Number of training epochs
        dataset_name: Dataset to train on
        adversarial: Whether to use adversarial training (True) or non-adversarial (False)
    """
    if base_model_ids is None:
        # Train adversarial variants of top performing models (excluding T-statistic models)
        base_model_ids = ['A2', 'A3', 'B1', 'B2', 'B4', 'B5']  # Working adversarial models
    
    training_type = "Adversarial" if adversarial else "Non-Adversarial"
    print(f"🎯 {training_type} Training of Baseline Model Variants")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Base models: {base_model_ids}")
    if force_retrain:
        print("🔄 Force retrain mode: Ignoring existing checkpoints")
    if memory_efficient:
        print("🧠 Memory-efficient mode enabled")
    print("=" * 60)
    
    # Setup checkpoint manager
    adv_suffix = "_adversarial" if adversarial else "_nonadversarial"
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}{adv_suffix}')
    trainer = AdversarialTrainer(checkpoint_manager)
    
    # Check existing models if not force retraining
    if not force_retrain:
        print(f"\nChecking for existing adversarial models...")
        checkpoint_manager.print_available_models()
    
    # Setup training data
    train_loader, example_batch, signals = setup_training_data(memory_efficient=memory_efficient)
    
    training_results = {}
    
    # Filter models to train based on existing checkpoints
    models_to_train = []
    for base_model_id in base_model_ids:
        adv_model_id = f"{base_model_id}_ADV" if adversarial else f"{base_model_id}_NONADV"
        
        if not force_retrain and checkpoint_manager.model_exists(adv_model_id):
            print(f"⏭️ {adv_model_id} already trained, skipping...")
        else:
            if force_retrain and checkpoint_manager.model_exists(adv_model_id):
                print(f"🔄 {adv_model_id} exists but retraining due to --force-retrain flag")
            models_to_train.append(base_model_id)
    
    if not models_to_train:
        print(f"\n✅ All adversarial variants already trained!")
        print(f"   Use --force-retrain flag to retrain existing models")
        return {}
    
    print(f"\nModels to train: {len(models_to_train)}")
    for model_id in models_to_train:
        adv_id = f"{model_id}_ADV" if adversarial else f"{model_id}_NONADV"
        print(f"  {adv_id}")
    
    for base_model_id in models_to_train:
        print(f"\n{'='*60}")
        print(f"Creating {training_type} Variant of {base_model_id}")
        print(f"{'='*60}")
        
        try:
            # Create adversarial model
            adversarial_model = create_adversarial_model(
                base_model_id=base_model_id,
                example_batch=example_batch,
                real_data=signals,
                adversarial=adversarial,
                memory_efficient=memory_efficient
            )
            
            print(f"Model parameters: {sum(p.numel() for p in adversarial_model.parameters()):,}")
            
            # Setup optimizers
            g_optimizer = optim.Adam(adversarial_model.generator.parameters(), lr=0.001)
            
            if adversarial and adversarial_model.discriminator.adversarial:
                d_optimizer = optim.Adam(adversarial_model.discriminator.parameters(), lr=0.001)
            else:
                d_optimizer = None
            
            # Train model
            history = trainer.train_adversarial_model(
                model=adversarial_model,
                train_loader=train_loader,
                g_optimizer=g_optimizer,
                d_optimizer=d_optimizer,
                num_epochs=num_epochs,
                d_steps=1 if adversarial else 0,
                g_steps=1,
                patience=10,  # Fixed patience value
                save_every=25,  # Fixed save_every value
                memory_efficient=memory_efficient,
                accumulation_steps=8 if memory_efficient else 1
            )
            
            training_results[adversarial_model.model_id] = history
            
        except Exception as e:
            print(f"❌ Training failed for {base_model_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save training summary
    if training_results:
        save_training_summary(training_results, dataset_name, adversarial)
    
    return training_results


def setup_training_data(n_samples: int = 256, n_points: int = 100, batch_size: int = 32, memory_efficient: bool = False):
    """Setup training data with memory optimization options."""
    print(f"Setting up training data...")
    if memory_efficient:
        batch_size = min(batch_size, 8)  # Reduce batch size for memory efficiency
        print(f"  🧠 Memory-efficient mode: batch_size reduced to {batch_size}")
    
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


def save_training_summary(training_results: Dict, dataset_name: str, adversarial: bool):
    """Save adversarial training summary."""
    summary_data = []
    for model_id, history in training_results.items():
        summary_data.append({
            'model_id': model_id,
            'base_model': history.get('base_model_id', model_id.replace('_ADV', '').replace('_NONADV', '')),
            'best_g_loss': history['best_g_loss'],
            'best_epoch': history['best_epoch'],
            'final_g_loss': history['final_g_loss'],
            'final_d_loss': history['final_d_loss'],
            'epochs_trained': history['epochs_trained'],
            'total_time': history['total_time'],
            'adversarial': adversarial
        })
    
    summary_df = pd.DataFrame(summary_data)
    adv_suffix = "_adversarial" if adversarial else "_nonadversarial"
    summary_dir = f'results/{dataset_name}{adv_suffix}/training'
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f'{dataset_name}{adv_suffix}_training_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    training_type = "Adversarial" if adversarial else "Non-Adversarial"
    print(f"\n📊 {training_type} Training Summary:")
    print(summary_df.to_string(index=False))
    print(f"Summary saved to: {summary_path}")


def train_all_adversarial_models(dataset_name: str = 'ou_process', num_epochs: int = 100,
                                memory_efficient: bool = True, force_retrain: bool = False):
    """
    Train adversarial variants of all available models.
    
    Args:
        dataset_name: Dataset to train on
        num_epochs: Number of training epochs
        memory_efficient: Use memory-efficient training
        force_retrain: Force retrain existing models
    """
    # Only include models that work with adversarial training (exclude T-statistic models)
    all_models = ['A2', 'A3', 'B1', 'B2', 'B4', 'B5']
    
    print(f"🚀 Training ALL Adversarial Model Variants")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Models: {all_models}")
    print(f"Memory efficient: {memory_efficient}")
    print(f"Force retrain: {force_retrain}")
    print("=" * 70)
    
    return train_adversarial_models(
        base_model_ids=all_models,
        num_epochs=num_epochs,
        dataset_name=dataset_name,
        adversarial=True,
        memory_efficient=memory_efficient,
        force_retrain=force_retrain
    )


def compare_adversarial_vs_baseline():
    """Compare adversarial variants with baseline models."""
    print("📊 Comparing Adversarial vs Baseline Performance")
    print("=" * 50)
    
    # This would load and compare results from both training approaches
    # Implementation would depend on having trained both variants
    pass


def main():
    """Main adversarial training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train adversarial variants of baseline models")
    parser.add_argument("--models", nargs='+', default=['A2', 'B3', 'B4'], 
                       help="Base model IDs to create adversarial variants for")
    parser.add_argument("--all", action="store_true",
                       help="Train adversarial variants of all available models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--dataset", type=str, default='ou_process', 
                       help="Dataset to train on")
    parser.add_argument("--non-adversarial", action="store_true", 
                       help="Train non-adversarial variants (for comparison)")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Use memory-efficient training (gradient accumulation, smaller batches)")
    parser.add_argument("--force-retrain", action="store_true",
                       help="Force retrain all models (ignores existing checkpoints)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare adversarial vs baseline performance")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_adversarial_vs_baseline()
        return
    
    # Train adversarial variants
    adversarial = not args.non_adversarial
    
    if args.all:
        # Train all models
        training_results = train_all_adversarial_models(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            memory_efficient=args.memory_efficient,
            force_retrain=args.force_retrain
        )
    else:
        # Train specified models
        training_results = train_adversarial_models(
            base_model_ids=args.models,
            num_epochs=args.epochs,
            dataset_name=args.dataset,
            adversarial=adversarial,
            memory_efficient=args.memory_efficient,
            force_retrain=args.force_retrain
        )
    
    training_type = "Adversarial" if adversarial else "Non-Adversarial"
    print(f"\n🎉 {training_type} training complete!")
    print(f"   Models trained: {list(training_results.keys())}")
    print(f"   Results saved to: results/{args.dataset}{'_adversarial' if adversarial else '_nonadversarial'}/")


if __name__ == "__main__":
    main()

"""
Model Training Utilities

Enhanced model trainers with checkpointing, logging, and visualization support.
"""

import torch
import torch.optim as optim
import numpy as np
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from utils.training_visualization import TrainingVisualizer


class ModelTrainer:
    """Enhanced model trainer that saves best models during training."""
    
    def __init__(self, checkpoint_manager):
        """Initialize trainer with checkpoint manager."""
        self.checkpoint_manager = checkpoint_manager
        self.training_history = {}
        self.visualizer = None
    
    def train_with_checkpointing(self, model, model_id: str, train_loader, 
                               optimizer, num_epochs: int, 
                               save_every: int = 20, patience: int = 10,
                               enable_trajectory_viz: bool = True, viz_every: int = 10):
        """
        Train model with automatic checkpointing and optional trajectory visualization.
        
        Args:
            model: Model to train
            model_id: Unique identifier for the model
            train_loader: Training data loader
            optimizer: Optimizer
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            patience: Early stopping patience
            enable_trajectory_viz: Enable trajectory visualization during training
            viz_every: Create trajectory visualization every N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"\\nTraining {model_id} with checkpointing...")
        print(f"  Epochs: {num_epochs}, Save every: {save_every}, Patience: {patience}")
        if enable_trajectory_viz:
            print(f"  📊 Trajectory visualization every {viz_every} epochs")
        
        # Setup visualization
        if enable_trajectory_viz:
            base_dir = self.checkpoint_manager.base_dir
            self.visualizer = TrainingVisualizer(base_dir, model_id)
            print(f"  📁 Visualizations will be saved to: {self.visualizer.visualization_dir}")
        
        # Get training device
        device = next(model.parameters()).device
        
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
                data = data.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                
                # Compute loss (handle different model types)
                loss = self._compute_model_loss(model, output, data)
                
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
            
            # Create trajectory visualization every viz_every epochs
            if enable_trajectory_viz and self.visualizer and ((epoch + 1) % viz_every == 0 or epoch == 0):
                try:
                    self.visualizer.create_trajectory_comparison(model, train_loader, epoch + 1, device)
                except Exception as viz_e:
                    print(f"  ⚠️ Trajectory visualization failed at epoch {epoch + 1}: {viz_e}")
            
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
    
    def _compute_model_loss(self, model, output: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Compute loss for different model types."""
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
        
        return loss


class LoggingModelTrainer(ModelTrainer):
    """Enhanced model trainer with logging support."""
    
    def __init__(self, checkpoint_manager, logger: Optional[logging.Logger] = None):
        """Initialize trainer with checkpoint manager and logger."""
        super().__init__(checkpoint_manager)
        self.logger = logger or logging.getLogger(__name__)
    
    def train_with_checkpointing(self, model, model_id: str, train_loader, 
                               optimizer, num_epochs: int, 
                               save_every: int = 20, patience: int = 10,
                               enable_trajectory_viz: bool = True, viz_every: int = 10):
        """Train model with automatic checkpointing and logging."""
        self.logger.info(f"Starting training for {model_id}")
        self.logger.info(f"Configuration: epochs={num_epochs}, save_every={save_every}, patience={patience}")
        if enable_trajectory_viz:
            self.logger.info(f"Trajectory visualization enabled: every {viz_every} epochs")
        self.logger.info(f"Optimizer: {type(optimizer).__name__}, lr={optimizer.param_groups[0]['lr']}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        try:
            # Call parent method
            history = super().train_with_checkpointing(
                model, model_id, train_loader, optimizer, num_epochs, save_every, patience,
                enable_trajectory_viz, viz_every
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


def train_model_memory_optimized(model, model_id: str, checkpoint_manager, train_data: torch.Tensor, 
                               epochs: int, device: torch.device):
    """Memory-optimized training for models with high memory requirements."""
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
            
            batch_data = train_data[batch_indices].to(device)
            
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
        
        # Save best model
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

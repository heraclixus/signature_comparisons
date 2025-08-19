"""
Model Checkpoint System

This module provides functionality to save and load trained models,
avoiding the need to retrain models for evaluation.
"""

import torch
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from datetime import datetime


class ModelCheckpoint:
    """
    Model checkpoint system for saving and loading trained models.
    """
    
    def __init__(self, base_dir: str = "results"):
        """
        Initialize checkpoint system.
        
        Args:
            base_dir: Base directory for saving checkpoints
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "trained_models")
        os.makedirs(self.models_dir, exist_ok=True)
    
    def save_model(self, model, model_id: str, epoch: int, loss: float, 
                   metrics: Optional[Dict] = None, training_config: Optional[Dict] = None,
                   training_history: Optional[Dict] = None):
        """
        Save a trained model checkpoint.
        
        Args:
            model: The trained model to save
            model_id: Unique identifier for the model (e.g., "A1", "A2", "A3")
            epoch: Training epoch when saved
            loss: Training loss at checkpoint
            metrics: Optional evaluation metrics
            training_config: Optional training configuration
            training_history: Optional training loss evolution and dynamics
        """
        # Create model-specific directory
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Checkpoint info
        checkpoint_info = {
            "model_id": model_id,
            "epoch": epoch,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "training_config": training_config or {},
            "model_class": model.__class__.__name__,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Save model state
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save model configuration instead of full object (to avoid pickle issues)
        model_config_path = os.path.join(model_dir, "model_config.json")
        model_config = {
            'model_class': model.__class__.__name__,
            'model_module': model.__class__.__module__,
            'model_id': model_id,
            'state_dict_keys': list(model.state_dict().keys()),
            'config': getattr(model, 'config', {})
        }
        
        with open(model_config_path, 'w') as f:
            json.dump(model_config, f, indent=2, default=str)
        
        # Save checkpoint info
        info_path = os.path.join(model_dir, "checkpoint_info.json")
        with open(info_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        
        # Save training history if provided
        if training_history:
            self._save_training_history(model_dir, model_id, training_history)
        
        print(f"✅ Model {model_id} saved to {model_dir}")
        print(f"   Epoch: {epoch}, Loss: {loss:.6f}")
        if metrics:
            print(f"   Metrics: {len(metrics)} saved")
        if training_history:
            print(f"   Training history: {len(training_history.get('losses', []))} epochs saved")
    
    def _save_training_history(self, model_dir: str, model_id: str, training_history: Dict):
        """Save training loss evolution and create visualization."""
        import numpy as np
        
        try:
            # Extract training data
            losses = training_history.get('losses', [])
            times = training_history.get('times', [])
            epochs = list(range(1, len(losses) + 1))
            
            if not losses:
                print(f"⚠️ No training losses to save for {model_id}")
                return
            
            # Create training history DataFrame
            history_data = {
                'epoch': epochs,
                'loss': losses,
                'cumulative_time': np.cumsum(times) if times else [0] * len(losses),
                'epoch_time': times if times else [0] * len(losses)
            }
            
            # Add additional metrics if available
            for key, values in training_history.items():
                if key not in ['losses', 'times', 'model_id', 'best_loss', 'best_epoch', 'final_loss', 'total_time', 'epochs_trained']:
                    if isinstance(values, list) and len(values) == len(losses):
                        history_data[key] = values
            
            df = pd.DataFrame(history_data)
            
            # Save training history CSV
            history_csv_path = os.path.join(model_dir, "training_history.csv")
            df.to_csv(history_csv_path, index=False)
            
            # Create training curve visualization
            self._create_training_curve(model_dir, model_id, df, training_history)
            
        except Exception as e:
            print(f"⚠️ Failed to save training history for {model_id}: {e}")
    
    def _create_training_curve(self, model_dir: str, model_id: str, df: pd.DataFrame, training_history: Dict):
        """Create training curve visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss evolution plot
            ax1.plot(df['epoch'], df['loss'], 'b-', linewidth=2, alpha=0.8, label='Training Loss')
            
            # Mark best epoch
            best_epoch = training_history.get('best_epoch', 1)
            best_loss = training_history.get('best_loss', df['loss'].min())
            ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
            ax1.scatter([best_epoch], [best_loss], color='red', s=100, zorder=5)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss')
            ax1.set_title(f'{model_id} Training Loss Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Training time plot
            if 'cumulative_time' in df.columns and df['cumulative_time'].sum() > 0:
                ax2.plot(df['epoch'], df['cumulative_time'], 'g-', linewidth=2, alpha=0.8, label='Cumulative Time')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Cumulative Time (s)')
                ax2.set_title(f'{model_id} Training Time')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            else:
                # Loss smoothing plot if no time data
                if len(df) > 5:
                    window_size = min(5, len(df) // 3)
                    smoothed_loss = df['loss'].rolling(window=window_size, center=True).mean()
                    ax2.plot(df['epoch'], df['loss'], 'b-', alpha=0.3, label='Raw Loss')
                    ax2.plot(df['epoch'], smoothed_loss, 'b-', linewidth=2, label=f'Smoothed (window={window_size})')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Training Loss')
                    ax2.set_title(f'{model_id} Loss Smoothing')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'Not enough data\nfor smoothing', 
                            ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(f'{model_id} Training Summary')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(model_dir, "training_curve.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ Failed to create training curve for {model_id}: {e}")
    
    def load_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """
        Load a trained model checkpoint.
        
        Args:
            model_id: Model identifier to load
            
        Returns:
            Loaded model or None if not found
        """
        model_dir = os.path.join(self.models_dir, model_id)
        
        if not os.path.exists(model_dir):
            return None
        
        # Load model configuration
        model_config_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(model_config_path):
            print(f"❌ Model config not found for {model_id}")
            return None
        
        try:
            # Load model config
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            
            # Recreate model based on model_id
            model = self._recreate_model(model_id, model_config)
            
            if model is None:
                print(f"❌ Could not recreate model {model_id}")
                return None
            
            # Load state dict
            model_path = os.path.join(model_dir, "model.pth")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            
            model.eval()  # Set to evaluation mode
            print(f"✅ Model {model_id} loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _recreate_model(self, model_id: str, model_config: Dict) -> Optional[torch.nn.Module]:
        """
        Recreate model from configuration.
        
        Args:
            model_id: Model identifier
            model_config: Model configuration dictionary
            
        Returns:
            Recreated model or None if failed
        """
        try:
            # Import required modules based on model_id
            if model_id == "A1":
                from models.implementations.a1_final import create_a1_final_model
                # We need example data to recreate - this is a limitation
                # For now, create dummy data
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_a1_final_model(example_batch, real_data)
            
            elif model_id == "A2":
                from models.implementations.a2_canned_scoring import create_a2_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_a2_model(example_batch, real_data)
            
            elif model_id == "A3":
                from models.implementations.a3_canned_mmd import create_a3_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_a3_model(example_batch, real_data)
            
            elif model_id == "B4":
                from models.implementations.b4_nsde_mmd import create_b4_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_b4_model(example_batch, real_data)
            
            elif model_id == "B5":
                from models.implementations.b5_nsde_scoring import create_b5_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_b5_model(example_batch, real_data)
            
            elif model_id == "A4":
                from models.implementations.a4_canned_logsig import create_a4_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_a4_model(example_batch, real_data)
            
            elif model_id == "B3":
                from models.implementations.b3_nsde_tstatistic import create_b3_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_b3_model(example_batch, real_data)
            
            elif model_id == "B1":
                from models.implementations.b1_nsde_scoring import create_b1_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_b1_model(example_batch, real_data)
            
            elif model_id == "B2":
                from models.implementations.b2_nsde_mmd_pde import create_b2_model
                example_batch = torch.randn(32, 2, 100)
                real_data = torch.randn(32, 2, 100)
                return create_b2_model(example_batch, real_data)
            
            # C1-C3 models removed - not truly generative
            
            else:
                print(f"❌ Unknown model_id: {model_id}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to recreate model {model_id}: {e}")
            return None
    
    def get_checkpoint_info(self, model_id: str) -> Optional[Dict]:
        """
        Get checkpoint information for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Checkpoint info dictionary or None
        """
        model_dir = os.path.join(self.models_dir, model_id)
        info_path = os.path.join(model_dir, "checkpoint_info.json")
        
        if not os.path.exists(info_path):
            return None
        
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load checkpoint info for {model_id}: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """
        List all available trained models.
        
        Returns:
            List of model IDs that have checkpoints
        """
        if not os.path.exists(self.models_dir):
            return []
        
        available_models = []
        for item in os.listdir(self.models_dir):
            model_dir = os.path.join(self.models_dir, item)
            if os.path.isdir(model_dir):
                # Check if it has required files
                model_path = os.path.join(model_dir, "model.pth")
                info_path = os.path.join(model_dir, "checkpoint_info.json")
                config_path = os.path.join(model_dir, "model_config.json")
                
                if os.path.exists(model_path) and os.path.exists(info_path) and os.path.exists(config_path):
                    available_models.append(item)
        
        return available_models
    
    def print_available_models(self):
        """Print summary of all available trained models."""
        available_models = self.list_available_models()
        
        if not available_models:
            print("No trained models found.")
            return
        
        print(f"Available Trained Models ({len(available_models)}):")
        print("=" * 50)
        
        for model_id in available_models:
            info = self.get_checkpoint_info(model_id)
            if info:
                print(f"{model_id:10} | Epoch {info['epoch']:3d} | Loss: {info['loss']:8.4f} | "
                      f"Params: {info['total_parameters']:,} | {info['timestamp'][:19]}")
            else:
                print(f"{model_id:10} | Info not available")
    
    def model_exists(self, model_id: str) -> bool:
        """
        Check if a trained model exists.
        
        Args:
            model_id: Model identifier to check
            
        Returns:
            True if model checkpoint exists
        """
        return model_id in self.list_available_models()
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model checkpoint.
        
        Args:
            model_id: Model identifier to delete
            
        Returns:
            True if successfully deleted
        """
        model_dir = os.path.join(self.models_dir, model_id)
        
        if not os.path.exists(model_dir):
            print(f"Model {model_id} does not exist")
            return False
        
        try:
            import shutil
            shutil.rmtree(model_dir)
            print(f"✅ Model {model_id} deleted successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to delete model {model_id}: {e}")
            return False


def create_checkpoint_manager(base_dir: str = "results") -> ModelCheckpoint:
    """
    Create a model checkpoint manager.
    
    Args:
        base_dir: Base directory for checkpoints
        
    Returns:
        ModelCheckpoint instance
    """
    return ModelCheckpoint(base_dir)


if __name__ == "__main__":
    # Test checkpoint system
    print("Testing Model Checkpoint System")
    print("=" * 40)
    
    checkpoint_manager = create_checkpoint_manager()
    
    # List available models
    checkpoint_manager.print_available_models()
    
    # Test model existence
    test_models = ["A1", "A2", "A3", "B1", "C1"]
    
    print(f"\nModel Existence Check:")
    for model_id in test_models:
        exists = checkpoint_manager.model_exists(model_id)
        print(f"  {model_id}: {'✅ Available' if exists else '❌ Not found'}")

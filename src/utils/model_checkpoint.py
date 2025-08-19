"""
Model Checkpoint System

This module provides functionality to save and load trained models,
avoiding the need to retrain models for evaluation.
"""

import torch
import os
import json
import pickle
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
                   metrics: Optional[Dict] = None, training_config: Optional[Dict] = None):
        """
        Save a trained model checkpoint.
        
        Args:
            model: The trained model to save
            model_id: Unique identifier for the model (e.g., "A1", "A2", "A3")
            epoch: Training epoch when saved
            loss: Training loss at checkpoint
            metrics: Optional evaluation metrics
            training_config: Optional training configuration
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
        
        print(f"✅ Model {model_id} saved to {model_dir}")
        print(f"   Epoch: {epoch}, Loss: {loss:.6f}")
        if metrics:
            print(f"   Metrics: {len(metrics)} saved")
    
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

"""
Base Model Template for Signature-based Time Series Generation

This provides the abstract base class that all signature-based models should inherit from,
implementing a consistent interface across different generator/loss/signature combinations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum
import json
import os


class GeneratorType(Enum):
    """Available generator types (truly generative only)."""
    NEURAL_SDE = "neural_sde"
    CANNED_NET = "canned_net" 
    SIMPLE_CANNED_NET = "simple_canned_net"
    # RNN, LSTM, GRU, TRANSFORMER removed - not truly generative for stochastic processes


class LossType(Enum):
    """Available loss types."""
    T_STATISTIC = "t_statistic"
    SIGNATURE_SCORING = "signature_scoring"
    SIGNATURE_MMD = "signature_mmd"
    DIRECT_SIGNATURE = "direct_signature"


class SignatureMethod(Enum):
    """Available signature computation methods."""
    TRUNCATED = "truncated"
    PDE_SOLVED = "pde_solved"
    LOG_SIGNATURES = "log_signatures"
    KERNEL_METHODS = "kernel_methods"


@dataclass
class ModelConfig:
    """
    Unified configuration for signature-based models.
    
    This provides a consistent configuration interface across all model types.
    """
    
    # Required fields (no defaults)
    model_id: str
    name: str
    generator_type: GeneratorType
    loss_type: LossType
    signature_method: SignatureMethod
    
    # Optional fields (with defaults)
    description: str = ""
    priority: str = "medium"  # "high", "medium", "low"
    status: str = "proposed"  # "implemented", "running", "completed", "failed"
    
    # Component-specific configurations
    generator_config: Dict[str, Any] = field(default_factory=dict)
    loss_config: Dict[str, Any] = field(default_factory=dict)
    signature_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation configuration
    eval_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default configurations if not provided."""
        self._set_default_configs()
    
    def _set_default_configs(self):
        """Set default configurations based on component types."""
        # Generator defaults
        if not self.generator_config:
            self.generator_config = self._get_default_generator_config()
        
        # Loss defaults
        if not self.loss_config:
            self.loss_config = self._get_default_loss_config()
        
        # Signature defaults
        if not self.signature_config:
            self.signature_config = self._get_default_signature_config()
        
        # Training defaults
        if not self.training_config:
            self.training_config = self._get_default_training_config()
        
        # Data defaults
        if not self.data_config:
            self.data_config = self._get_default_data_config()
        
        # Eval defaults
        if not self.eval_config:
            self.eval_config = self._get_default_eval_config()
    
    def _get_default_generator_config(self) -> Dict[str, Any]:
        """Get default generator configuration."""
        defaults = {
            GeneratorType.NEURAL_SDE: {
                'data_size': 2,
                'hidden_size': 64,
                'mlp_size': 128,
                'num_layers': 3,
                'activation': 'LipSwish',
                'noise_type': 'diagonal',
                'sde_type': 'stratonovich'
            },
            GeneratorType.CANNED_NET: {
                'augment_configs': [
                    {'layer_sizes': (8, 8, 2), 'kernel_size': 1, 'include_original': True, 'include_time': False}
                ],
                'window_config': {'window_size': 2, 'stride': 0, 'dilation': 1},
                'signature_config': {'depth': 3},
                'final_augment_config': {'layer_sizes': (1,), 'kernel_size': 1, 'include_original': False, 'include_time': False}
            }
            # RNN, LSTM, GRU, TRANSFORMER configs removed - not truly generative
        }
        return defaults.get(self.generator_type, {})
    
    def _get_default_loss_config(self) -> Dict[str, Any]:
        """Get default loss configuration."""
        defaults = {
            LossType.T_STATISTIC: {
                'sig_depth': 4,
                'normalise_sigs': True
            },
            LossType.SIGNATURE_SCORING: {
                'kernel_type': 'rbf',
                'sigma': 1.0,
                'adversarial': False,
                'max_batch': 128,
                'path_dim': 2
            },
            LossType.SIGNATURE_MMD: {
                'kernel_type': 'rbf',
                'sigma': 1.0,
                'adversarial': False,
                'max_batch': 128,
                'path_dim': 2
            }
        }
        return defaults.get(self.loss_type, {})
    
    def _get_default_signature_config(self) -> Dict[str, Any]:
        """Get default signature configuration."""
        defaults = {
            SignatureMethod.TRUNCATED: {
                'depth': 4
            },
            SignatureMethod.PDE_SOLVED: {
                'dyadic_order': 4,
                'kernel_type': 'rbf',
                'sigma': 1.0
            },
            SignatureMethod.KERNEL_METHODS: {
                'dyadic_order': 4,
                'kernel_type': 'rbf',
                'sigma': 1.0
            }
        }
        return defaults.get(self.signature_method, {})
    
    def _get_default_training_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            'batch_size': 128,
            'max_epochs': 100,
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'log_interval': 10,
            'eval_interval': 10,
            'save_interval': 25,
            'early_stopping': True,
            'patience': 20
        }
    
    def _get_default_data_config(self) -> Dict[str, Any]:
        """Get default data configuration."""
        return {
            'dataset_type': 'ornstein_uhlenbeck',
            'n_points': 100,
            'train_samples': 1024,
            'val_samples': 256,
            'test_samples': 256,
            'num_workers': 0,
            'data_params': {
                'theta': 1.0,
                'mu': 0.0,
                'sigma': 0.3
            }
        }
    
    def _get_default_eval_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration."""
        return {
            'metrics': ['mse', 'signature_distance', 'ks_test', 'moment_matching'],
            'save_plots': True,
            'save_models': True,
            'plot_samples': 50,
            'eval_samples': 1000
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'generator_type': self.generator_type.value,
            'loss_type': self.loss_type.value,
            'signature_method': self.signature_method.value,
            'generator_config': self.generator_config,
            'loss_config': self.loss_config,
            'signature_config': self.signature_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            'eval_config': self.eval_config,
            'priority': self.priority,
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(
            model_id=config_dict['model_id'],
            name=config_dict['name'],
            description=config_dict.get('description', ''),
            generator_type=GeneratorType(config_dict['generator_type']),
            loss_type=LossType(config_dict['loss_type']),
            signature_method=SignatureMethod(config_dict['signature_method']),
            generator_config=config_dict.get('generator_config', {}),
            loss_config=config_dict.get('loss_config', {}),
            signature_config=config_dict.get('signature_config', {}),
            training_config=config_dict.get('training_config', {}),
            data_config=config_dict.get('data_config', {}),
            eval_config=config_dict.get('eval_config', {}),
            priority=config_dict.get('priority', 'medium'),
            status=config_dict.get('status', 'proposed')
        )
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BaseSignatureModel(nn.Module, ABC):
    """
    Abstract base class for all signature-based models.
    
    This provides a consistent interface that all model implementations should follow,
    regardless of the specific combination of generator, loss, and signature method.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base signature model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.device = config.training_config.get('device', 'cpu')
        
        # Training state
        self.training_step = 0
        self.epoch = 0
        self.is_trained = False
        
        # Initialize components (to be implemented by subclasses)
        self.generator = None
        self.loss_function = None
        self.signature_transform = None
        
        # Build model
        self._build_model()
        
        # Move to device
        self.to(self.device)
    
    @abstractmethod
    def _build_model(self):
        """
        Build the model components (generator, loss, signature transform).
        
        This method must be implemented by each subclass to create the specific
        combination of components for that model.
        """
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            *args, **kwargs: Model-specific inputs
            
        Returns:
            Model output
        """
        pass
    
    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the model's loss function.
        
        Args:
            *args, **kwargs: Loss-specific inputs
            
        Returns:
            Loss value
        """
        pass
    
    def generate_samples(self, batch_size: int, 
                        device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            batch_size: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples
        """
        if device is None:
            device = self.device
        
        with torch.no_grad():
            # This is a default implementation - subclasses can override
            noise = torch.randn(batch_size, 100, device=device)
            return self.forward(noise)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_id': self.config.model_id,
            'name': self.config.name,
            'description': self.config.description,
            'generator_type': self.config.generator_type.value,
            'loss_type': self.config.loss_type.value,
            'signature_method': self.config.signature_method.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'training_step': self.training_step,
            'epoch': self.epoch,
            'priority': self.config.priority,
            'status': self.config.status
        }
    
    def save_model(self, filepath: str, include_config: bool = True):
        """
        Save model state and configuration.
        
        Args:
            filepath: Path to save the model
            include_config: Whether to include configuration in save
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info(),
            'training_step': self.training_step,
            'epoch': self.epoch,
            'is_trained': self.is_trained
        }
        
        if include_config:
            save_dict['config'] = self.config.to_dict()
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str):
        """
        Load model state from file.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.is_trained = checkpoint.get('is_trained', False)
    
    def summary(self) -> str:
        """Get a string summary of the model."""
        info = self.get_model_info()
        
        summary = f"""
{info['name']} ({info['model_id']})
{'=' * (len(info['name']) + len(info['model_id']) + 3)}

Description: {info['description']}

Architecture:
  Generator: {info['generator_type']}
  Loss: {info['loss_type']}
  Signature Method: {info['signature_method']}

Parameters: {info['trainable_parameters']:,} trainable / {info['total_parameters']:,} total

Training Status: {'Trained' if info['is_trained'] else 'Not trained'}
  Epoch: {info['epoch']}
  Training Step: {info['training_step']}

Device: {info['device']}
Priority: {info['priority']}
Status: {info['status']}
"""
        return summary.strip()
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model_id='{self.config.model_id}', name='{self.config.name}')"


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.
    
    This provides a consistent training interface across all model types.
    """
    
    def __init__(self, model: BaseSignatureModel):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
        """
        self.model = model
        self.config = model.config
        self.device = model.device
        
        # Training components (to be initialized by subclasses)
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        pass
    
    @abstractmethod
    def train(self, save_dir: Optional[str] = None) -> Dict[str, List]:
        """Run full training loop."""
        pass

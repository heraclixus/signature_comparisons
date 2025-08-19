"""
Experiment configuration for systematic benchmarking.

This module defines the configuration structure for running experiments
across the complete design space of signature-based methods.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List
from enum import Enum

# Import generator types from the new generators module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.generators import GeneratorType as GenType
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False
    # Fallback enum definition
    from enum import Enum
    class GenType(Enum):
        NEURAL_SDE = "neural_sde"
        CANNED_NET = "canned_net"
        RNN = "rnn"
        LSTM = "lstm"
        TRANSFORMER = "transformer"


class GeneratorType(Enum):
    """Available generator architectures."""
    NEURAL_SDE = "neural_sde"
    CANNED_NET = "canned_net"
    RNN = "rnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    HYBRID = "hybrid"


class LossType(Enum):
    """Available loss functions."""
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
class ExperimentConfig:
    """
    Configuration for a single experiment in the design space.
    
    This corresponds to one cell in the experimental matrix from README.md.
    """
    
    # Experiment identification
    experiment_id: str  # e.g., "A1", "B3", "C2"
    name: str          # Human-readable name
    
    # Core method configuration
    generator_type: GeneratorType
    loss_type: LossType
    signature_method: SignatureMethod
    
    # Generator-specific parameters
    generator_config: Dict[str, Any]
    
    # Loss-specific parameters
    loss_config: Dict[str, Any]
    
    # Signature-specific parameters
    signature_config: Dict[str, Any]
    
    # Training configuration
    training_config: Dict[str, Any]
    
    # Data configuration
    data_config: Dict[str, Any]
    
    # Evaluation configuration
    eval_config: Dict[str, Any]
    
    # Priority and status
    priority: str = "medium"  # "high", "medium", "low"
    status: str = "proposed"  # "implemented", "running", "completed", "failed"
    
    # Metadata
    description: Optional[str] = None
    expected_runtime_hours: Optional[float] = None
    computational_requirements: Optional[Dict[str, Any]] = None


def create_experiment_config(experiment_id: str, 
                           generator_type: Union[str, GeneratorType],
                           loss_type: Union[str, LossType],
                           signature_method: Union[str, SignatureMethod],
                           **kwargs) -> ExperimentConfig:
    """
    Factory function to create experiment configurations with sensible defaults.
    
    Args:
        experiment_id: Unique identifier (e.g., "A1", "B3")
        generator_type: Type of generator to use
        loss_type: Type of loss function to use
        signature_method: Signature computation method
        **kwargs: Additional configuration parameters
        
    Returns:
        ExperimentConfig instance with defaults filled in
    """
    
    # Convert string enums to enum types
    if isinstance(generator_type, str):
        generator_type = GeneratorType(generator_type)
    if isinstance(loss_type, str):
        loss_type = LossType(loss_type)
    if isinstance(signature_method, str):
        signature_method = SignatureMethod(signature_method)
    
    # Default configurations based on method types
    default_configs = _get_default_configs(generator_type, loss_type, signature_method)
    
    # Merge with provided kwargs
    for key, default_value in default_configs.items():
        if key in kwargs:
            if isinstance(default_value, dict) and isinstance(kwargs[key], dict):
                default_value.update(kwargs[key])
            else:
                default_configs[key] = kwargs[key]
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        name=kwargs.get('name', f"{generator_type.value}+{loss_type.value}+{signature_method.value}"),
        generator_type=generator_type,
        loss_type=loss_type,
        signature_method=signature_method,
        **default_configs,
        **{k: v for k, v in kwargs.items() if k not in default_configs}
    )


def _get_default_configs(generator_type: GeneratorType, 
                        loss_type: LossType,
                        signature_method: SignatureMethod) -> Dict[str, Any]:
    """Get default configurations based on method combination."""
    
    # Generator defaults
    generator_defaults = {
        GeneratorType.NEURAL_SDE: {
            'hidden_size': 64,
            'mlp_size': 128,
            'num_layers': 3,
            'sde_type': 'stratonovich',
            'noise_type': 'diagonal'
        },
        GeneratorType.CANNED_NET: {
            'augment_sizes': [(8, 8, 2), (1,)],
            'window_size': 2,
            'include_original': True,
            'include_time': False
        },
        GeneratorType.RNN: {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        },
        GeneratorType.LSTM: {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1
        },
        GeneratorType.TRANSFORMER: {
            'embed_dim': 64,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
    }
    
    # Loss defaults
    loss_defaults = {
        LossType.T_STATISTIC: {
            'sig_depth': 4,
            'normalise_sigs': True
        },
        LossType.SIGNATURE_SCORING: {
            'max_batch': 128,
            'adversarial': False
        },
        LossType.SIGNATURE_MMD: {
            'max_batch': 128,
            'adversarial': False
        }
    }
    
    # Signature method defaults
    signature_defaults = {
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
    
    # Training defaults
    training_defaults = {
        'batch_size': 128,
        'max_epochs': 100,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'device': 'cuda'
    }
    
    # Data defaults
    data_defaults = {
        'dataset_type': 'ornstein_uhlenbeck',
        'n_points': 100,
        'train_samples': 1000,
        'val_samples': 200,
        'test_samples': 200
    }
    
    # Evaluation defaults
    eval_defaults = {
        'metrics': ['mse', 'ks_test', 'moment_matching'],
        'save_plots': True,
        'save_models': True
    }
    
    return {
        'generator_config': generator_defaults.get(generator_type, {}),
        'loss_config': loss_defaults.get(loss_type, {}),
        'signature_config': signature_defaults.get(signature_method, {}),
        'training_config': training_defaults,
        'data_config': data_defaults,
        'eval_config': eval_defaults
    }


# Predefined experiment configurations from README.md
IMPLEMENTED_EXPERIMENTS = {
    "A1": create_experiment_config(
        "A1", GeneratorType.CANNED_NET, LossType.T_STATISTIC, SignatureMethod.TRUNCATED,
        status="implemented",
        priority="high",
        description="Original deep_signature_transform implementation"
    ),
    
    "B1": create_experiment_config(
        "B1", GeneratorType.NEURAL_SDE, LossType.SIGNATURE_SCORING, SignatureMethod.PDE_SOLVED,
        status="implemented", 
        priority="high",
        description="Non-adversarial sigker_nsdes implementation"
    ),
    
    "B2": create_experiment_config(
        "B2", GeneratorType.NEURAL_SDE, LossType.SIGNATURE_MMD, SignatureMethod.PDE_SOLVED,
        status="implemented",
        priority="high", 
        description="Non-adversarial MMD sigker_nsdes implementation"
    )
}

PROPOSED_EXPERIMENTS = {
    "A2": create_experiment_config(
        "A2", GeneratorType.CANNED_NET, LossType.SIGNATURE_SCORING, SignatureMethod.TRUNCATED,
        priority="high",
        description="Test scoring rule with signature-aware architecture"
    ),
    
    "A3": create_experiment_config(
        "A3", GeneratorType.CANNED_NET, LossType.SIGNATURE_MMD, SignatureMethod.TRUNCATED,
        priority="high",
        description="Compare MMD vs T-statistic with same architecture"
    ),
    
    "B3": create_experiment_config(
        "B3", GeneratorType.NEURAL_SDE, LossType.T_STATISTIC, SignatureMethod.PDE_SOLVED,
        priority="high",
        description="Physical dynamics + robust loss"
    ),
    
    "B4": create_experiment_config(
        "B4", GeneratorType.NEURAL_SDE, LossType.T_STATISTIC, SignatureMethod.TRUNCATED,
        priority="high",
        description="Physical dynamics + efficient signatures"
    ),
    
    "C1": create_experiment_config(
        "C1", GeneratorType.RNN, LossType.T_STATISTIC, SignatureMethod.TRUNCATED,
        priority="high",
        description="Baseline comparison with standard architectures"
    ),
    
    "C2": create_experiment_config(
        "C2", GeneratorType.RNN, LossType.SIGNATURE_SCORING, SignatureMethod.TRUNCATED,
        priority="high",
        description="Test scoring rule with standard RNN"
    ),
    
    "C3": create_experiment_config(
        "C3", GeneratorType.RNN, LossType.SIGNATURE_MMD, SignatureMethod.TRUNCATED,
        priority="high",
        description="Standard architecture with signature-based MMD"
    )
}

ALL_EXPERIMENTS = {**IMPLEMENTED_EXPERIMENTS, **PROPOSED_EXPERIMENTS}

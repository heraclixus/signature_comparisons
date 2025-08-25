"""
Model Factory for Signature-based Models

This provides factory methods for creating models with different combinations
of generators, losses, and signature methods, using the registry system.
"""

import torch
import sys
import os
from typing import Dict, Any, Optional, List, Type, Union
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod
from .model_registry import get_model_registry, register_model

# Import D2 model
try:
    from .d2_distributional_diffusion import D2DistributionalDiffusion, create_d2_config
    D2_AVAILABLE = True
except ImportError:
    D2_AVAILABLE = False

# Import GenericSignatureModel (defined later in this file)
# This will be available after the class definition

# Import component modules
try:
    from generators import create_generator
    from losses import TStatisticLoss, SignatureScoringLoss, SignatureMMDLoss
    from signatures import TruncatedSignature, get_signature_kernel
except ImportError:
    warnings.warn("Could not import all component modules. Some functionality may be limited.")


class ModelFactory:
    """
    Factory for creating signature-based models.
    
    This provides methods to create models either by registered ID
    or by specifying component combinations.
    """
    
    def __init__(self):
        """Initialize model factory."""
        self.registry = get_model_registry()
    
    def create_model(self, model_id: str, 
                    config_overrides: Optional[Dict[str, Any]] = None) -> BaseSignatureModel:
        """
        Create a model by registered ID.
        
        Args:
            model_id: Registered model identifier
            config_overrides: Optional configuration overrides
            
        Returns:
            Model instance
        """
        return self.registry.create_model(model_id, config_overrides)
    
    def create_model_from_config(self, config: ModelConfig) -> BaseSignatureModel:
        """
        Create a model from a configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
        """
        # Try to find registered model with matching configuration
        for model_id in self.registry.list_models():
            registered_config = self.registry.get_model_config(model_id)
            if self._configs_match(config, registered_config):
                return self.registry.create_model(model_id, config.to_dict())
        
        # If no registered model matches, create a generic model
        return self._create_generic_model(config)
    
    def create_model_from_components(self, 
                                   generator_type: Union[str, GeneratorType],
                                   loss_type: Union[str, LossType],
                                   signature_method: Union[str, SignatureMethod],
                                   model_id: Optional[str] = None,
                                   **kwargs) -> BaseSignatureModel:
        """
        Create a model by specifying component types.
        
        Args:
            generator_type: Type of generator
            loss_type: Type of loss function
            signature_method: Type of signature method
            model_id: Optional model identifier
            **kwargs: Additional configuration parameters
            
        Returns:
            Model instance
        """
        # Convert string types to enums
        if isinstance(generator_type, str):
            generator_type = GeneratorType(generator_type)
        if isinstance(loss_type, str):
            loss_type = LossType(loss_type)
        if isinstance(signature_method, str):
            signature_method = SignatureMethod(signature_method)
        
        # Generate model ID if not provided
        if model_id is None:
            model_id = f"{generator_type.value}_{loss_type.value}_{signature_method.value}"
        
        # Create configuration
        config = ModelConfig(
            model_id=model_id,
            name=f"{generator_type.value.title()} + {loss_type.value.title()} + {signature_method.value.title()}",
            description=f"Model combining {generator_type.value}, {loss_type.value}, and {signature_method.value}",
            generator_type=generator_type,
            loss_type=loss_type,
            signature_method=signature_method,
            **kwargs
        )
        
        return self.create_model_from_config(config)
    
    def _configs_match(self, config1: ModelConfig, config2: ModelConfig) -> bool:
        """Check if two configurations match in core components."""
        return (config1.generator_type == config2.generator_type and
                config1.loss_type == config2.loss_type and
                config1.signature_method == config2.signature_method)
    
    def _create_generic_model(self, config: ModelConfig) -> BaseSignatureModel:
        """Create a generic model from configuration."""
        return GenericSignatureModel(config)
    
    def list_available_models(self) -> List[str]:
        """List all available model IDs."""
        return self.registry.list_models()
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a registered model."""
        config = self.registry.get_model_config(model_id)
        metadata = self.registry.get_model_metadata(model_id)
        
        return {
            'model_id': model_id,
            'name': config.name,
            'description': config.description,
            'generator_type': config.generator_type.value,
            'loss_type': config.loss_type.value,
            'signature_method': config.signature_method.value,
            'priority': config.priority,
            'status': config.status,
            'metadata': metadata
        }
    
    def print_available_models(self):
        """Print information about all available models."""
        self.registry.print_registry_summary()


class GenericSignatureModel(BaseSignatureModel):
    """
    Generic signature-based model that can be configured with any
    combination of generator, loss, and signature method.
    
    This is used when no specific model implementation is registered
    for a given configuration.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize generic model."""
        super().__init__(config)
    
    def _build_model(self):
        """Build model components from configuration."""
        # Create generator
        self.generator = self._create_generator()
        
        # Create signature transform
        self.signature_transform = self._create_signature_transform()
        
        # Create loss function
        self.loss_function = self._create_loss_function()
    
    def _create_generator(self):
        """Create generator from configuration."""
        try:
            return create_generator(
                self.config.generator_type,
                self.config.generator_config
            )
        except Exception as e:
            warnings.warn(f"Could not create {self.config.generator_type.value} generator: {e}")
            return self._create_fallback_generator()
    
    def _create_fallback_generator(self):
        """Create a simple fallback generator."""
        return torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 99)
        )
    
    def _create_signature_transform(self):
        """Create signature transform from configuration."""
        try:
            if self.config.signature_method == SignatureMethod.TRUNCATED:
                return TruncatedSignature(depth=self.config.signature_config.get('depth', 4))
            elif self.config.signature_method == SignatureMethod.PDE_SOLVED:
                return get_signature_kernel(**self.config.signature_config)
            else:
                warnings.warn(f"Signature method {self.config.signature_method.value} not implemented")
                return torch.nn.Identity()
        except Exception as e:
            warnings.warn(f"Could not create signature transform: {e}")
            return torch.nn.Identity()
    
    def _create_loss_function(self):
        """Create loss function from configuration."""
        try:
            if self.config.loss_type == LossType.T_STATISTIC:
                return TStatisticLoss(
                    signature_transform=self.signature_transform,
                    **self.config.loss_config
                )
            elif self.config.loss_type == LossType.SIGNATURE_SCORING:
                return SignatureScoringLoss(
                    signature_kernel=None,  # Would need actual kernel
                    **self.config.loss_config
                )
            elif self.config.loss_type == LossType.SIGNATURE_MMD:
                return SignatureMMDLoss(
                    signature_kernel=None,  # Would need actual kernel
                    **self.config.loss_config
                )
            else:
                warnings.warn(f"Loss type {self.config.loss_type.value} not implemented")
                return torch.nn.MSELoss()
        except Exception as e:
            warnings.warn(f"Could not create loss function: {e}")
            return torch.nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.generator(x)
    
    def compute_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between generated and target data."""
        if hasattr(self.loss_function, '__call__'):
            return self.loss_function(generated, target)
        else:
            return torch.nn.functional.mse_loss(generated, target)


# Global factory instance
_global_factory = ModelFactory()


def create_model(model_id_or_config: Union[str, ModelConfig], 
                config_overrides: Optional[Dict[str, Any]] = None) -> BaseSignatureModel:
    """
    Create a model from ID or configuration.
    
    Args:
        model_id_or_config: Model ID string or ModelConfig object
        config_overrides: Optional configuration overrides
        
    Returns:
        Model instance
    """
    if isinstance(model_id_or_config, str):
        return _global_factory.create_model(model_id_or_config, config_overrides)
    elif isinstance(model_id_or_config, ModelConfig):
        return _global_factory.create_model_from_config(model_id_or_config)
    else:
        raise TypeError("model_id_or_config must be str or ModelConfig")


def create_model_from_components(generator_type: Union[str, GeneratorType],
                               loss_type: Union[str, LossType], 
                               signature_method: Union[str, SignatureMethod],
                               **kwargs) -> BaseSignatureModel:
    """
    Create a model by specifying component types.
    
    Args:
        generator_type: Type of generator
        loss_type: Type of loss function
        signature_method: Type of signature method
        **kwargs: Additional configuration parameters
        
    Returns:
        Model instance
    """
    return _global_factory.create_model_from_components(
        generator_type, loss_type, signature_method, **kwargs
    )


def get_available_models() -> List[str]:
    """Get list of available model IDs."""
    return _global_factory.list_available_models()


def get_model_info(model_id: str) -> Dict[str, Any]:
    """Get information about a model."""
    return _global_factory.get_model_info(model_id)


def print_available_models():
    """Print information about all available models."""
    _global_factory.print_available_models()


# Predefined model configurations from README.md
def register_predefined_models():
    """Register the predefined model configurations from README.md."""
    
    # A1: CannedNet + T-Statistic + Truncated (implemented)
    a1_config = ModelConfig(
        model_id="A1",
        name="CannedNet + T-Statistic + Truncated",
        description="Original deep_signature_transform implementation",
        generator_type=GeneratorType.CANNED_NET,
        loss_type=LossType.T_STATISTIC,
        signature_method=SignatureMethod.TRUNCATED,
        status="implemented",
        priority="high"
    )
    
    # A2: CannedNet + Signature Scoring + Truncated (high priority)
    a2_config = ModelConfig(
        model_id="A2", 
        name="CannedNet + Signature Scoring + Truncated",
        description="Test scoring rule with signature-aware architecture",
        generator_type=GeneratorType.CANNED_NET,
        loss_type=LossType.SIGNATURE_SCORING,
        signature_method=SignatureMethod.TRUNCATED,
        priority="high"
    )
    
    # B1: Neural SDE + Signature Scoring + PDE-Solved (implemented)
    b1_config = ModelConfig(
        model_id="B1",
        name="Neural SDE + Signature Scoring + PDE-Solved", 
        description="Non-adversarial sigker_nsdes implementation",
        generator_type=GeneratorType.NEURAL_SDE,
        loss_type=LossType.SIGNATURE_SCORING,
        signature_method=SignatureMethod.PDE_SOLVED,
        status="implemented",
        priority="high"
    )
    
    # B3: Neural SDE + T-Statistic + PDE-Solved (high priority)
    b3_config = ModelConfig(
        model_id="B3",
        name="Neural SDE + T-Statistic + PDE-Solved",
        description="Physical dynamics + robust loss",
        generator_type=GeneratorType.NEURAL_SDE,
        loss_type=LossType.T_STATISTIC,
        signature_method=SignatureMethod.PDE_SOLVED,
        priority="high"
    )
    
    # D2: Distributional Diffusion + Signature Kernel Scoring (new implementation)
    d2_config = None
    if D2_AVAILABLE:
        d2_config = create_d2_config()
    
    # Store configs for later registration (after GenericSignatureModel is defined)
    configs = [a1_config, a2_config, b1_config, b3_config]
    if d2_config is not None:
        configs.append(d2_config)
    
    return configs


class GenericSignatureModel(BaseSignatureModel):
    """
    Generic signature-based model that can be configured with any
    combination of generator, loss, and signature method.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize generic model."""
        super().__init__(config)
    
    def _build_model(self):
        """Build model components from configuration."""
        # Create generator
        self.generator = self._create_generator()
        
        # Create signature transform
        self.signature_transform = self._create_signature_transform()
        
        # Create loss function
        self.loss_function = self._create_loss_function()
    
    def _create_generator(self):
        """Create generator from configuration."""
        try:
            return create_generator(
                self.config.generator_type,
                self.config.generator_config
            )
        except Exception as e:
            warnings.warn(f"Could not create {self.config.generator_type.value} generator: {e}")
            return self._create_fallback_generator()
    
    def _create_fallback_generator(self):
        """Create a simple fallback generator."""
        return torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 99)
        )
    
    def _create_signature_transform(self):
        """Create signature transform from configuration."""
        try:
            if self.config.signature_method == SignatureMethod.TRUNCATED:
                return TruncatedSignature(depth=self.config.signature_config.get('depth', 4))
            else:
                warnings.warn(f"Signature method {self.config.signature_method.value} not implemented")
                return torch.nn.Identity()
        except Exception as e:
            warnings.warn(f"Could not create signature transform: {e}")
            return torch.nn.Identity()
    
    def _create_loss_function(self):
        """Create loss function from configuration."""
        try:
            if self.config.loss_type == LossType.T_STATISTIC:
                return TStatisticLoss(
                    signature_transform=self.signature_transform,
                    **self.config.loss_config
                )
            else:
                warnings.warn(f"Loss type {self.config.loss_type.value} not implemented")
                return torch.nn.MSELoss()
        except Exception as e:
            warnings.warn(f"Could not create loss function: {e}")
            return torch.nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.generator(x)
    
    def compute_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss between generated and target data."""
        if hasattr(self.loss_function, '__call__'):
            return self.loss_function(generated, target)
        else:
            return torch.nn.functional.mse_loss(generated, target)


def _register_predefined_models():
    """Register the predefined model configurations."""
    configs = register_predefined_models()
    
    for config in configs:
        try:
            # Use D2 specific model class for D2, generic for others
            if config.model_id == "D2" and D2_AVAILABLE:
                register_model(config.model_id, D2DistributionalDiffusion, config)
            else:
                register_model(config.model_id, GenericSignatureModel, config)
        except ValueError:
            # Already registered
            pass


# Register predefined models on import
_register_predefined_models()

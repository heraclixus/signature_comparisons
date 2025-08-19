"""
Model Registry for Signature-based Models

This provides a registry system for registering and discovering different
model implementations, enabling dynamic model creation and management.
"""

from typing import Dict, Type, List, Optional, Any
from .base_model import BaseSignatureModel, ModelConfig, GeneratorType, LossType, SignatureMethod


class ModelRegistry:
    """
    Registry for signature-based model implementations.
    
    This allows model classes to register themselves and provides
    discovery and creation functionality.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._models: Dict[str, Type[BaseSignatureModel]] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, model_id: str, model_class: Type[BaseSignatureModel], 
                config: ModelConfig, metadata: Optional[Dict[str, Any]] = None):
        """
        Register a model implementation.
        
        Args:
            model_id: Unique identifier for the model
            model_class: Model class that inherits from BaseSignatureModel
            config: Default configuration for the model
            metadata: Optional metadata about the model
        """
        if model_id in self._models:
            raise ValueError(f"Model '{model_id}' is already registered")
        
        if not issubclass(model_class, BaseSignatureModel):
            raise TypeError(f"Model class must inherit from BaseSignatureModel")
        
        self._models[model_id] = model_class
        self._configs[model_id] = config
        self._metadata[model_id] = metadata or {}
        
        print(f"Registered model: {model_id} ({config.name})")
    
    def unregister(self, model_id: str):
        """
        Unregister a model.
        
        Args:
            model_id: Model identifier to unregister
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' is not registered")
        
        del self._models[model_id]
        del self._configs[model_id]
        del self._metadata[model_id]
    
    def get_model_class(self, model_id: str) -> Type[BaseSignatureModel]:
        """
        Get model class by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model class
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' is not registered")
        
        return self._models[model_id]
    
    def get_model_config(self, model_id: str) -> ModelConfig:
        """
        Get default model configuration by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model configuration
        """
        if model_id not in self._configs:
            raise ValueError(f"Model '{model_id}' is not registered")
        
        return self._configs[model_id]
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata
        """
        return self._metadata.get(model_id, {})
    
    def list_models(self) -> List[str]:
        """
        List all registered model IDs.
        
        Returns:
            List of model identifiers
        """
        return list(self._models.keys())
    
    def list_models_by_type(self, generator_type: Optional[GeneratorType] = None,
                           loss_type: Optional[LossType] = None,
                           signature_method: Optional[SignatureMethod] = None) -> List[str]:
        """
        List models filtered by component types.
        
        Args:
            generator_type: Filter by generator type
            loss_type: Filter by loss type
            signature_method: Filter by signature method
            
        Returns:
            List of matching model identifiers
        """
        matching_models = []
        
        for model_id, config in self._configs.items():
            if generator_type and config.generator_type != generator_type:
                continue
            if loss_type and config.loss_type != loss_type:
                continue
            if signature_method and config.signature_method != signature_method:
                continue
            
            matching_models.append(model_id)
        
        return matching_models
    
    def get_models_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered models.
        
        Returns:
            List of model information dictionaries
        """
        models_info = []
        
        for model_id in self._models:
            config = self._configs[model_id]
            metadata = self._metadata[model_id]
            
            info = {
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
            
            models_info.append(info)
        
        return models_info
    
    def create_model(self, model_id: str, 
                    config_overrides: Optional[Dict[str, Any]] = None) -> BaseSignatureModel:
        """
        Create a model instance.
        
        Args:
            model_id: Model identifier
            config_overrides: Optional configuration overrides
            
        Returns:
            Model instance
        """
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' is not registered")
        
        # Get base configuration
        config = self._configs[model_id]
        
        # Apply overrides if provided
        if config_overrides:
            config_dict = config.to_dict()
            config_dict.update(config_overrides)
            config = ModelConfig.from_dict(config_dict)
        
        # Create model instance
        model_class = self._models[model_id]
        return model_class(config)
    
    def print_registry_summary(self):
        """Print a summary of all registered models."""
        print("Model Registry Summary")
        print("=" * 50)
        
        if not self._models:
            print("No models registered.")
            return
        
        models_info = self.get_models_info()
        
        # Group by status
        by_status = {}
        for info in models_info:
            status = info['status']
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(info)
        
        for status in ['implemented', 'running', 'completed', 'proposed', 'failed']:
            if status in by_status:
                print(f"\n{status.upper()} ({len(by_status[status])} models):")
                for info in by_status[status]:
                    print(f"  {info['model_id']:15} | {info['name']:30} | "
                          f"{info['generator_type']:12} + {info['loss_type']:15} + {info['signature_method']:12}")
        
        # Summary statistics
        print(f"\nTotal Models: {len(models_info)}")
        
        # By generator type
        gen_counts = {}
        for info in models_info:
            gen_type = info['generator_type']
            gen_counts[gen_type] = gen_counts.get(gen_type, 0) + 1
        
        print(f"By Generator: {dict(gen_counts)}")
        
        # By loss type
        loss_counts = {}
        for info in models_info:
            loss_type = info['loss_type']
            loss_counts[loss_type] = loss_counts.get(loss_type, 0) + 1
        
        print(f"By Loss: {dict(loss_counts)}")


# Global registry instance
_global_registry = ModelRegistry()


def register_model(model_id: str, model_class: Type[BaseSignatureModel], 
                  config: ModelConfig, metadata: Optional[Dict[str, Any]] = None):
    """
    Register a model in the global registry.
    
    Args:
        model_id: Unique identifier for the model
        model_class: Model class that inherits from BaseSignatureModel
        config: Default configuration for the model
        metadata: Optional metadata about the model
    """
    _global_registry.register(model_id, model_class, config, metadata)


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry.
    
    Returns:
        Global ModelRegistry instance
    """
    return _global_registry


def list_registered_models() -> List[str]:
    """
    List all registered model IDs.
    
    Returns:
        List of model identifiers
    """
    return _global_registry.list_models()


def create_registered_model(model_id: str, 
                           config_overrides: Optional[Dict[str, Any]] = None) -> BaseSignatureModel:
    """
    Create a model instance from the global registry.
    
    Args:
        model_id: Model identifier
        config_overrides: Optional configuration overrides
        
    Returns:
        Model instance
    """
    return _global_registry.create_model(model_id, config_overrides)


# Decorator for easy model registration
def signature_model(model_id: str, config: Optional[ModelConfig] = None, 
                   metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator to register a model class.
    
    Args:
        model_id: Unique identifier for the model
        config: Model configuration (will use class default if None)
        metadata: Optional metadata
        
    Returns:
        Decorator function
    """
    def decorator(model_class: Type[BaseSignatureModel]):
        # Use class method to get config if not provided
        if config is None:
            if hasattr(model_class, 'get_default_config'):
                model_config = model_class.get_default_config()
            else:
                raise ValueError(f"Model class {model_class.__name__} must provide config or implement get_default_config()")
        else:
            model_config = config
        
        register_model(model_id, model_class, model_config, metadata)
        return model_class
    
    return decorator

"""
Generator models for signature-based time series generation.

This module contains all generator architectures extracted from the existing
deep_signature_transform and sigker_nsdes codebases, organized for easy reuse.
"""

from .canned_net import CannedNet, CannedResNet, create_canned_net_generator, create_simple_canned_net
from .neural_sde import NeuralSDEGenerator, GeneratorFunc, create_nsde_generator
from .factory import (
    GeneratorType, 
    RNNGenerator, 
    TransformerGenerator,
    create_generator, 
    get_default_config
)

__all__ = [
    # CannedNet generators
    'CannedNet',
    'CannedResNet', 
    'create_canned_net_generator',
    'create_simple_canned_net',
    
    # Neural SDE generators
    'NeuralSDEGenerator',
    'GeneratorFunc',
    'create_nsde_generator',
    
    # Baseline generators
    'RNNGenerator',
    'TransformerGenerator',
    
    # Factory functions
    'GeneratorType',
    'create_generator',
    'get_default_config'
]

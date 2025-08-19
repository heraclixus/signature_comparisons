"""
CannedNet Generator Implementation (Extracted from deep_signature_transform)

This implements the signature-aware CannedNet architecture originally from
deep_signature_transform/candle/modules.py and dataset/generative_model.py
"""

import torch
import torch.nn as nn
import functools as ft
from typing import Tuple, Union, Any, Optional

# Import the required components from the original codebase
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from deep_signature_transform.candle.modules import Lambda, NoInputSpec
    from deep_signature_transform.candle.utils import batch_flatten
    from deep_signature_transform.candle.recurrent import Window
    from deep_signature_transform.siglayer.modules import Augment
    from deep_signature_transform.siglayer.backend import Signature
except ImportError:
    # Fallback imports if the structure is different
    try:
        from models.deep_signature_transform.candle.modules import Lambda, NoInputSpec
        from models.deep_signature_transform.candle.utils import batch_flatten
        from models.deep_signature_transform.candle.recurrent import Window
        from models.deep_signature_transform.siglayer.modules import Augment
        from models.deep_signature_transform.siglayer.backend import Signature
    except ImportError:
        raise ImportError("Could not import required deep_signature_transform components")


class CannedNet(nn.Module):
    """
    CannedNet Generator - Signature-aware neural network architecture.
    
    Provides a simple extensible way to specify a neural network without having to define a class.
    A bit like Sequential, but more general:
    - A Module may be specified by something as simple as an integer (width of Linear layer)
    - Input size of Linear layer does not need computing in advance
    - Framework can be extended to create more complicated nets (e.g., ResNets)
    
    Originally implemented in deep_signature_transform/candle/modules.py
    """

    def __init__(self, hidden_blocks: Tuple, debug: bool = False, **kwargs):
        """
        Initialize CannedNet generator.

        Args:
            hidden_blocks: Tuple of elements specifying the layers of the network. Each element may be:
                - An integer, specifying the size of a Linear layer
                - A Module
                - A callable (will be wrapped in Lambda)
                - Other types as interpreted by _interpret_element
            debug: Whether to print tensor sizes as they go through the network
        """
        super(CannedNet, self).__init__(**kwargs)

        self.hidden_blocks = hidden_blocks
        self.debug = debug

        self.layers = nn.ModuleList()
        self._build_network()

    def _build_network(self):
        """Build the network from hidden_blocks specification."""
        current_shapes = None
        
        for elem in self.hidden_blocks:
            layer = self._interpret_element_wrapper(elem)
            self.layers.append(layer)

    def _interpret_element(self, elem):
        """
        Interpret an element of the hidden_blocks tuple into a PyTorch module.
        
        This method can be overridden in subclasses to support additional element types.
        
        Args:
            elem: Element of the hidden_blocks tuple
            
        Returns:
            PyTorch module or None if element couldn't be interpreted
        """
        if isinstance(elem, int):
            if elem < 1:
                raise ValueError(f'Integers specifying layer sizes must be >= 1. Given {elem}.')
            layer = NoInputSpec(nn.Linear, elem)
            return layer
        elif isinstance(elem, nn.Module):
            return elem
        elif callable(elem):
            return Lambda(elem)
        
        return None

    def _interpret_element_wrapper(self, elem):
        """Wrap _interpret_element to check if element was understood."""
        out = self._interpret_element(elem)
        if out is None:
            raise ValueError(f'Element {elem} of type {type(elem)} in hidden_blocks was not understood.')
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        if self.debug:
            print(f'Input: {x.shape}')
        
        for layer in self.layers:
            x = layer(x)
            if self.debug:
                print(f'{type(layer).__name__}: {x.shape}')
        
        return x


class CannedResNet(CannedNet):
    """
    CannedResNet - CannedNet with ResNet-style skip connections.
    
    Extends CannedNet to understand tuples as elements of hidden_blocks.
    Tuple elements are interpreted recursively as another CannedResNet,
    with skip connections added across the layers specified in the tuple.
    """

    def _interpret_element(self, elem):
        """Interpret element, with additional support for tuples (skip connections)."""
        if isinstance(elem, tuple):
            # Create a sub-network with skip connection
            sub_net = CannedResNet(elem, debug=self.debug)
            
            # Wrap with skip connection
            def skip_connection(x):
                return x + sub_net(x)
            
            return Lambda(skip_connection)
        else:
            # Fall back to parent class interpretation
            return super()._interpret_element(elem)


def create_canned_net_generator(augment_configs: list = None, 
                               window_config: dict = None,
                               signature_config: dict = None,
                               final_augment_config: dict = None) -> CannedNet:
    """
    Create the standard CannedNet generator from deep_signature_transform.
    
    This recreates the generator from dataset/generative_model.py:
    CannedNet((
        Augment((8, 8, 2), 1, include_original=True, include_time=False),
        Window(2, 0, 1, transformation=Signature(3)),
        Augment((1,), 1, include_original=False, include_time=False),
        batch_flatten
    ))
    
    Args:
        augment_configs: Configuration for augmentation layers
        window_config: Configuration for windowing layer
        signature_config: Configuration for signature computation
        final_augment_config: Configuration for final augmentation
        
    Returns:
        CannedNet generator instance
    """
    # Default configurations matching the original implementation
    if augment_configs is None:
        augment_configs = [
            {'layer_sizes': (8, 8, 2), 'kernel_size': 1, 'include_original': True, 'include_time': False}
        ]
    
    if window_config is None:
        window_config = {'window_size': 2, 'stride': 0, 'dilation': 1}
    
    if signature_config is None:
        signature_config = {'depth': 3}
    
    if final_augment_config is None:
        final_augment_config = {'layer_sizes': (1,), 'kernel_size': 1, 'include_original': False, 'include_time': False}
    
    # Build the network components
    blocks = []
    
    # First augmentation layer
    for aug_config in augment_configs:
        augment_layer = Augment(**aug_config)
        blocks.append(augment_layer)
    
    # Window layer with signature transformation
    signature_transform = Signature(**signature_config)
    window_layer = Window(transformation=signature_transform, **window_config)
    blocks.append(window_layer)
    
    # Final augmentation layer
    final_augment = Augment(**final_augment_config)
    blocks.append(final_augment)
    
    # Flatten layer
    blocks.append(batch_flatten)
    
    return CannedNet(tuple(blocks))


def create_simple_canned_net(layer_sizes: Tuple[int, ...] = (64, 64, 32),
                            activation: str = 'ReLU',
                            final_activation: Optional[str] = None) -> CannedNet:
    """
    Create a simple feedforward CannedNet generator.
    
    Args:
        layer_sizes: Sizes of hidden layers
        activation: Activation function name
        final_activation: Final activation function (optional)
        
    Returns:
        Simple CannedNet generator
    """
    blocks = []
    
    # Add hidden layers with activations
    for size in layer_sizes:
        blocks.append(size)  # Linear layer
        if activation:
            activation_fn = getattr(nn, activation)()
            blocks.append(activation_fn)
    
    # Add final activation if specified
    if final_activation:
        final_activation_fn = getattr(nn, final_activation)()
        blocks.append(final_activation_fn)
    
    return CannedNet(tuple(blocks))

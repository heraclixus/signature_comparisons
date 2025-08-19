"""
Generator Factory for creating different types of generators.

This provides a unified interface for creating generators of different types
based on configuration parameters.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional
from enum import Enum

from .canned_net import CannedNet, create_canned_net_generator, create_simple_canned_net
from .neural_sde import NeuralSDEGenerator, create_nsde_generator


class GeneratorType(Enum):
    """Available generator types."""
    NEURAL_SDE = "neural_sde"
    CANNED_NET = "canned_net"
    SIMPLE_CANNED_NET = "simple_canned_net"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"


class RNNGenerator(nn.Module):
    """Simple RNN-based generator for baseline comparisons."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 1, rnn_type: str = "RNN", dropout: float = 0.1,
                 sequence_length: int = 100):
        """
        Initialize RNN generator.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            output_size: Output feature size
            rnn_type: Type of RNN ("RNN", "LSTM", "GRU")
            dropout: Dropout rate
            sequence_length: Length of generated sequences
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_size = output_size
        
        # RNN layer
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              dropout=dropout, batch_first=True)
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.rnn_type = rnn_type.upper()
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate sequences from noise.
        
        Args:
            noise: Input noise, shape (batch_size, sequence_length, input_size)
            
        Returns:
            Generated sequences, shape (batch_size, sequence_length, output_size)
        """
        batch_size = noise.size(0)
        
        # Initialize hidden state
        if self.rnn_type == "LSTM":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=noise.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=noise.device)
            hidden = (h0, c0)
        else:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=noise.device)
        
        # Forward pass through RNN
        rnn_output, _ = self.rnn(noise, hidden)
        
        # Apply output layer
        output = self.output_layer(rnn_output)
        
        return output
    
    def generate(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate sequences from random noise.
        
        Args:
            batch_size: Number of sequences to generate
            device: Device to generate on
            
        Returns:
            Generated sequences
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Generate random noise
        noise = torch.randn(batch_size, self.sequence_length, 1, device=device)
        
        return self.forward(noise)


class TransformerGenerator(nn.Module):
    """Transformer-based generator for sequence generation."""
    
    def __init__(self, vocab_size: int = 1, embed_dim: int = 64, num_heads: int = 8,
                 num_layers: int = 4, sequence_length: int = 100, dropout: float = 0.1):
        """
        Initialize Transformer generator.
        
        Args:
            vocab_size: Vocabulary size (for discrete tokens) or feature size (for continuous)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            sequence_length: Length of generated sequences
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # Input projection
        self.input_projection = nn.Linear(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(sequence_length, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate sequences from input.
        
        Args:
            x: Input tensor, shape (batch_size, sequence_length, vocab_size)
            
        Returns:
            Generated sequences, shape (batch_size, sequence_length, vocab_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Output projection
        output = self.output_projection(x)
        
        return output
    
    def generate(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate sequences from random noise.
        
        Args:
            batch_size: Number of sequences to generate
            device: Device to generate on
            
        Returns:
            Generated sequences
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Generate random input
        x = torch.randn(batch_size, self.sequence_length, self.vocab_size, device=device)
        
        return self.forward(x)


def create_generator(generator_type: Union[str, GeneratorType], 
                    config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create generators of different types.
    
    Args:
        generator_type: Type of generator to create
        config: Configuration dictionary
        
    Returns:
        Generator instance
    """
    if isinstance(generator_type, str):
        generator_type = GeneratorType(generator_type)
    
    if generator_type == GeneratorType.NEURAL_SDE:
        return create_nsde_generator(**config)
    
    elif generator_type == GeneratorType.CANNED_NET:
        return create_canned_net_generator(**config)
    
    elif generator_type == GeneratorType.SIMPLE_CANNED_NET:
        return create_simple_canned_net(**config)
    
    elif generator_type == GeneratorType.RNN:
        return RNNGenerator(rnn_type="RNN", **config)
    
    elif generator_type == GeneratorType.LSTM:
        return RNNGenerator(rnn_type="LSTM", **config)
    
    elif generator_type == GeneratorType.GRU:
        return RNNGenerator(rnn_type="GRU", **config)
    
    elif generator_type == GeneratorType.TRANSFORMER:
        return TransformerGenerator(**config)
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


def get_default_config(generator_type: Union[str, GeneratorType]) -> Dict[str, Any]:
    """
    Get default configuration for a generator type.
    
    Args:
        generator_type: Type of generator
        
    Returns:
        Default configuration dictionary
    """
    if isinstance(generator_type, str):
        generator_type = GeneratorType(generator_type)
    
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
        },
        
        GeneratorType.SIMPLE_CANNED_NET: {
            'layer_sizes': (64, 64, 32),
            'activation': 'ReLU',
            'final_activation': None
        },
        
        GeneratorType.RNN: {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.1,
            'sequence_length': 100
        },
        
        GeneratorType.LSTM: {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.1,
            'sequence_length': 100
        },
        
        GeneratorType.GRU: {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'dropout': 0.1,
            'sequence_length': 100
        },
        
        GeneratorType.TRANSFORMER: {
            'vocab_size': 1,
            'embed_dim': 64,
            'num_heads': 8,
            'num_layers': 4,
            'sequence_length': 100,
            'dropout': 0.1
        }
    }
    
    return defaults.get(generator_type, {})

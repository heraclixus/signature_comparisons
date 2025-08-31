"""
Transformer-Based Distributional Generator for D2 Model

This module implements a transformer-based generator for the D2 distributional
diffusion model, replacing the simple MLP with attention-based sequence modeling.

Key improvements:
- Self-attention across time steps
- Positional encoding for temporal awareness  
- Better long-range dependency modeling
- Enhanced sequence coherence
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input.
    
    Adds sinusoidal positional information to help the transformer
    understand the temporal ordering of time series data.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (seq_len, batch, d_model)
            
        Returns:
            Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps.
    
    Creates rich time representations that help the transformer
    understand the diffusion process state.
    """
    
    def __init__(self, dim: int):
        """
        Initialize time embedding.
        
        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal time embedding.
        
        Args:
            t: Time values (batch,)
            
        Returns:
            Time embeddings (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class TransformerDistributionalGenerator(nn.Module):
    """
    Transformer-based generator for distributional diffusion.
    
    This generator uses self-attention to model temporal dependencies
    in time series, providing better sequence coherence and long-range
    dependency modeling compared to the MLP-based generator.
    
    Architecture:
    1. Input projection: (data + time) per timestep
    2. Noise projection: noise per timestep  
    3. Transformer encoder: Multi-head self-attention
    4. Output projection: Generate clean path
    """
    
    def __init__(
        self,
        data_size: int = 2,
        seq_len: int = 64,
        hidden_size: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_embed_dim: int = 32,
        feedforward_dim: Optional[int] = None
    ):
        """
        Initialize transformer distributional generator.
        
        Args:
            data_size: Dimension of each time point (typically 2)
            seq_len: Number of time points in sequence
            hidden_size: Hidden dimension for transformer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            time_embed_dim: Dimension of time embedding
            feedforward_dim: Feedforward dimension (default: 4 * hidden_size)
        """
        super().__init__()
        
        self.data_size = data_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.time_embed_dim = time_embed_dim
        
        if feedforward_dim is None:
            feedforward_dim = hidden_size * 4
        
        # Time embedding for diffusion timestep
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        
        # Input projections
        # Project (data + time_per_step) to hidden dimension
        self.input_projection = nn.Linear(data_size + time_embed_dim // seq_len, hidden_size)
        
        # Project noise to hidden dimension
        self.noise_projection = nn.Linear(data_size, hidden_size)
        
        # Positional encoding for sequence position
        self.positional_encoding = PositionalEncoding(hidden_size * 2, max_len=seq_len)
        
        # Transformer encoder (with PyTorch version compatibility)
        encoder_kwargs = {
            'd_model': hidden_size * 2,  # Combined input + noise projections
            'nhead': num_heads,
            'dim_feedforward': feedforward_dim,
            'dropout': dropout,
            'activation': 'relu',
            'batch_first': True
        }
        
        # Add norm_first only if supported (PyTorch >= 1.9.0)
        try:
            import torch
            # Parse version more carefully to handle versions like "1.8.1+cu111"
            version_str = torch.__version__.split('+')[0]  # Remove build info
            version_parts = version_str.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            
            if (major > 1) or (major == 1 and minor >= 9):
                # Test if norm_first is actually supported by trying to create a layer
                try:
                    test_layer = torch.nn.TransformerEncoderLayer(
                        d_model=32, nhead=4, norm_first=True, batch_first=True
                    )
                    encoder_kwargs['norm_first'] = True  # Pre-norm for better training
                    print(f"      ✅ PyTorch {torch.__version__} - using pre-norm (norm_first=True)")
                except TypeError:
                    print(f"      ℹ️ PyTorch {torch.__version__} - norm_first not supported, using post-norm")
            else:
                print(f"      ℹ️ PyTorch {torch.__version__} - using post-norm (norm_first not available)")
        except Exception as e:
            # Fallback: don't use norm_first
            print(f"      ⚠️ Version detection failed ({e}) - using post-norm as fallback")
        
        encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, data_size)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        print(f"🤖 TransformerDistributionalGenerator initialized:")
        print(f"   Data size: {data_size}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Transformer layers: {num_layers}")
        print(f"   Attention heads: {num_heads}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer generator.
        
        Args:
            x_t: Noisy paths (batch_size, data_size, seq_len)
            t: Diffusion timesteps (batch_size,)
            z: Noise (batch_size, data_size, seq_len)
            
        Returns:
            Generated clean paths (batch_size, data_size, seq_len)
        """
        batch_size, data_size, seq_len = x_t.shape
        
        # Create time embedding
        t_emb = self.time_embedding(t)  # (batch_size, time_embed_dim)
        
        # Reshape inputs to sequence format: (batch, seq_len, data_size)
        x_seq = x_t.transpose(1, 2)  # (batch_size, seq_len, data_size)
        z_seq = z.transpose(1, 2)    # (batch_size, seq_len, data_size)
        
        # Add time information to each timestep
        # Distribute time embedding across sequence
        t_per_step = t_emb.unsqueeze(1).expand(-1, seq_len, -1) / seq_len  # (batch, seq_len, time_embed_dim)
        t_per_step = t_per_step[:, :, :self.time_embed_dim // seq_len]  # Truncate to reasonable size
        
        # Combine input with time
        x_with_time = torch.cat([x_seq, t_per_step], dim=-1)  # (batch, seq_len, data_size + time_dim)
        
        # Project inputs
        x_proj = self.input_projection(x_with_time)  # (batch, seq_len, hidden_size)
        z_proj = self.noise_projection(z_seq)        # (batch, seq_len, hidden_size)
        
        # Combine input and noise projections
        combined = torch.cat([x_proj, z_proj], dim=-1)  # (batch, seq_len, hidden_size * 2)
        
        # Add positional encoding and layer norm
        combined = self.layer_norm(combined)
        
        # Transformer expects (seq_len, batch, d_model) for positional encoding
        combined_transposed = combined.transpose(0, 1)  # (seq_len, batch, hidden_size * 2)
        combined_with_pos = self.positional_encoding(combined_transposed)
        combined_final = combined_with_pos.transpose(0, 1)  # (batch, seq_len, hidden_size * 2)
        
        # Apply transformer
        transformer_output = self.transformer(combined_final)  # (batch, seq_len, hidden_size * 2)
        
        # Project to output
        output_seq = self.output_projection(transformer_output)  # (batch, seq_len, data_size)
        
        # Return in original format: (batch, data_size, seq_len)
        return output_seq.transpose(1, 2)
    
    def get_attention_weights(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for analysis (useful for debugging/visualization).
        
        Returns:
            Attention weights from the last transformer layer
        """
        # This would require modifying the transformer to return attention weights
        # For now, return None (can be implemented if needed for analysis)
        return None


def create_transformer_distributional_generator(
    data_size: int = 2,
    seq_len: int = 64,
    hidden_size: int = 64,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> TransformerDistributionalGenerator:
    """
    Factory function for creating transformer distributional generator.
    
    Args:
        data_size: Data dimension
        seq_len: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        **kwargs: Additional parameters
        
    Returns:
        TransformerDistributionalGenerator instance
    """
    return TransformerDistributionalGenerator(
        data_size=data_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        **kwargs
    )


def test_transformer_generator():
    """Test the transformer generator implementation."""
    print("🧪 Testing TransformerDistributionalGenerator")
    print("=" * 60)
    
    # Create generator
    generator = TransformerDistributionalGenerator(
        data_size=2,
        seq_len=64,
        hidden_size=64,
        num_layers=4,
        num_heads=8
    )
    
    # Test forward pass
    batch_size = 8
    x_t = torch.randn(batch_size, 2, 64)  # Noisy paths
    t = torch.rand(batch_size)            # Diffusion timesteps
    z = torch.randn(batch_size, 2, 64)    # Noise
    
    # Forward pass
    output = generator(x_t, t, z)
    
    print(f"✅ Forward pass successful:")
    print(f"   Input shape: {x_t.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    # Test gradient flow
    loss = output.mean()
    loss.backward()
    
    print(f"✅ Gradient flow successful")
    print(f"🎉 TransformerDistributionalGenerator is ready!")


if __name__ == "__main__":
    test_transformer_generator()

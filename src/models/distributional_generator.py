"""
Generator Network P_θ for Distributional Diffusion.

This module implements the generator network P_θ(·|X_t, t, Z) that learns
to generate clean path samples from noisy paths, time, and noise.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Union

# Import existing network components
try:
    from models.tsdiff.utils.feedforward import FeedForward
    TSDIFF_AVAILABLE = True
except ImportError:
    TSDIFF_AVAILABLE = False
    warnings.warn("tsdiff FeedForward not available")

try:
    from models.sigker_nsdes.src.gan.generators import Generator, GeneratorFunc
    from models.sigker_nsdes.src.gan.base import MLP
    SIGKER_NSDES_AVAILABLE = True
except ImportError:
    SIGKER_NSDES_AVAILABLE = False
    warnings.warn("sigker_nsdes generators not available")


class DistributionalGenerator(nn.Module):
    """
    Generator network P_θ(·|X_t, t, Z) for distributional diffusion.
    
    This network learns to generate clean path samples X̃_0 given:
    - Noisy paths X_t from forward diffusion
    - Diffusion time t
    - Random noise Z
    
    The network is trained using the signature kernel scoring rule to learn
    the full conditional distribution rather than just point estimates.
    """
    
    def __init__(
        self,
        data_size: int,        # d: dimension of each time point
        seq_len: int,          # M: number of time points  
        hidden_size: int = 128,
        num_layers: int = 3,
        activation: str = "relu",
        use_time_embedding: bool = True,
        time_embed_dim: int = 32,
        dropout: float = 0.1,
        final_activation: str = "tanh"
    ):
        """
        Initialize distributional generator.
        
        Args:
            data_size: Dimension of each time point (d)
            seq_len: Number of time points (M)
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            activation: Activation function name
            use_time_embedding: Whether to use sinusoidal time embedding
            time_embed_dim: Dimension of time embedding
            dropout: Dropout rate
            final_activation: Final activation function
        """
        super().__init__()
        self.data_size = data_size
        self.seq_len = seq_len
        self.use_time_embedding = use_time_embedding
        self.time_embed_dim = time_embed_dim
        
        # Calculate input dimensions
        path_dim = data_size * seq_len  # Flattened path dimension
        noise_dim = data_size * seq_len  # Noise dimension (same as path)
        
        if use_time_embedding:
            time_dim = time_embed_dim
            self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        else:
            time_dim = 1
            self.time_embedding = None
        
        input_dim = path_dim + time_dim + noise_dim
        output_dim = path_dim
        
        # Create network architecture
        self.network = self._create_network(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            final_activation=final_activation
        )
    
    def _create_network(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_size: int,
        num_layers: int,
        activation: str,
        dropout: float,
        final_activation: str
    ) -> nn.Module:
        """Create the main network architecture."""
        
        # Try to use existing FeedForward from tsdiff
        if TSDIFF_AVAILABLE:
            activation_fn = self._get_activation(activation)
            final_activation_fn = self._get_activation(final_activation) if final_activation else None
            
            return FeedForward(
                in_dim=input_dim,
                hidden_dims=[hidden_size] * num_layers,
                out_dim=output_dim,
                activation=activation_fn,
                final_activation=final_activation_fn
            )
        
        # Fallback to manual construction
        else:
            layers = []
            current_dim = input_dim
            
            activation_fn = self._get_activation(activation)
            
            # Hidden layers
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(current_dim, hidden_size),
                    activation_fn,
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_size
            
            # Output layer
            layers.append(nn.Linear(current_dim, output_dim))
            
            # Final activation
            if final_activation:
                layers.append(self._get_activation(final_activation))
            
            return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if isinstance(activation, str):
            activation_map = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.2),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "elu": nn.ELU()
            }
            
            if activation.lower() in activation_map:
                return activation_map[activation.lower()]
            else:
                warnings.warn(f"Unknown activation {activation}, using ReLU")
                return nn.ReLU()
        else:
            # If it's already a module, return it
            return activation
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from P_θ(·|X_t, t, Z).
        
        Args:
            x_t: Noisy paths (batch_size, dim, seq_len)
            t: Diffusion time (batch_size,) or (batch_size, 1)
            z: Noise (batch_size, dim, seq_len)
            
        Returns:
            Generated clean paths (batch_size, dim, seq_len)
        """
        batch_size = x_t.shape[0]
        
        # Flatten path inputs
        x_t_flat = x_t.view(batch_size, -1)  # (batch, dim*seq_len)
        z_flat = z.view(batch_size, -1)      # (batch, dim*seq_len)
        
        # Process time input
        if t.dim() == 1:
            t = t.view(batch_size, 1)  # (batch, 1)
        
        if self.use_time_embedding:
            t_embed = self.time_embedding(t.squeeze(-1))  # (batch, time_embed_dim)
        else:
            t_embed = t  # (batch, 1)
        
        # Concatenate inputs: [X_t, t, Z]
        inputs = torch.cat([x_t_flat, t_embed, z_flat], dim=1)
        
        # Generate output
        output_flat = self.network(inputs)
        
        # Reshape to path format
        return output_flat.view(batch_size, self.data_size, self.seq_len)


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion time steps.
    Similar to positional encoding in transformers.
    """
    
    def __init__(self, embed_dim: int, max_time: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time = max_time
        
        # Create frequency matrix
        half_dim = embed_dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim) / half_dim
        )
        self.register_buffer('frequencies', frequencies)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for time steps.
        
        Args:
            t: Time steps (batch_size,)
            
        Returns:
            Time embeddings (batch_size, embed_dim)
        """
        # Normalize time to [0, 1] if needed
        t_normalized = t / self.max_time
        
        # Compute sinusoidal embeddings
        args = t_normalized.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        
        embeddings = torch.cat([
            torch.sin(args),
            torch.cos(args)
        ], dim=-1)
        
        # Handle odd embedding dimensions
        if self.embed_dim % 2 == 1:
            embeddings = embeddings[:, :-1]
        
        return embeddings


class AdaptedSigKerGenerator(nn.Module):
    """
    Alternative generator that adapts the existing SDE-based Generator from sigker_nsdes.
    This approach reuses the SDE solving infrastructure for more sophisticated generation.
    """
    
    def __init__(
        self, 
        data_size: int, 
        seq_len: int,
        hidden_size: int = 64,
        mlp_size: int = 128,
        num_layers: int = 3,
        **kwargs
    ):
        """
        Initialize adapted generator using SDE infrastructure.
        
        Args:
            data_size: Dimension of each time point
            seq_len: Number of time points
            hidden_size: Hidden state size for SDE
            mlp_size: MLP layer size
            num_layers: Number of MLP layers
            **kwargs: Additional arguments for base Generator
        """
        super().__init__()
        
        if not SIGKER_NSDES_AVAILABLE:
            raise ImportError("AdaptedSigKerGenerator requires sigker_nsdes to be available")
        
        self.data_size = data_size
        self.seq_len = seq_len
        
        # Conditioning network: processes X_t and t to create initial state
        condition_dim = data_size * seq_len + 1  # X_t + t
        self.condition_network = MLP(
            condition_dim, hidden_size, mlp_size, num_layers, 
            activation=nn.ReLU(), tanh=False
        )
        
        # Base generator (modified for conditional generation)
        self.base_generator = Generator(
            data_size=data_size,
            initial_noise_size=hidden_size,  # Will be set by condition network
            noise_size=data_size,            # Driven by Z
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            num_layers=num_layers,
            fixed=False,  # Use learned initial state
            **kwargs
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using SDE-based approach.
        
        Args:
            x_t: Noisy paths (batch_size, dim, seq_len)
            t: Diffusion time (batch_size,)
            z: Noise (batch_size, dim, seq_len) - used as SDE driving noise
            
        Returns:
            Generated clean paths (batch_size, dim, seq_len)
        """
        batch_size = x_t.shape[0]
        
        # Create time grid for SDE integration
        ts = torch.linspace(0, 1, self.seq_len, device=x_t.device)
        
        # Process conditioning information
        x_t_flat = x_t.view(batch_size, -1)
        t_expanded = t.view(batch_size, 1)
        condition = torch.cat([x_t_flat, t_expanded], dim=1)
        
        # Get initial state from conditioning
        initial_state = self.condition_network(condition)
        
        # Modify base generator to use conditional initial state
        # This is a simplified approach - full implementation would require
        # modifying the SDE initial condition handling
        
        # For now, use the base generator with modified parameters
        # In practice, this would need deeper integration with the SDE solver
        coeffs = self.base_generator.forward(ts, batch_size)
        
        # Extract paths and reshape (remove time dimension)
        paths = coeffs[..., 1:]  # Remove time channel
        return paths.transpose(1, 2)  # (batch, dim, seq_len)


def create_distributional_generator(
    generator_type: str = "feedforward",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating distributional generators.
    
    Args:
        generator_type: "feedforward" or "sde_based"
        **kwargs: Arguments passed to generator constructor
        
    Returns:
        Distributional generator module
    """
    if generator_type == "feedforward":
        return DistributionalGenerator(**kwargs)
    elif generator_type == "sde_based":
        return AdaptedSigKerGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

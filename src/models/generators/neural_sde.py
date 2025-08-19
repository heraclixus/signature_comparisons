"""
Neural SDE Generator Implementation (Extracted from sigker_nsdes)

This implements Neural Stochastic Differential Equation generators originally from
sigker_nsdes/src/gan/generators.py
"""

import torch
import torch.nn as nn
from typing import Optional, Union

# Import required components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import torchsde
    import torchcde
    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    import warnings
    warnings.warn("torchsde/torchcde not available. Neural SDE functionality will be limited.")

try:
    from sigker_nsdes.src.gan.base import MLP
except ImportError:
    try:
        from models.sigker_nsdes.src.gan.base import MLP
    except ImportError:
        # Fallback MLP implementation
        class MLP(nn.Module):
            def __init__(self, in_size, out_size, mlp_size, num_layers, activation="ReLU", tanh=False, tscale=1):
                super().__init__()
                
                if isinstance(activation, str):
                    activation_fn = getattr(nn, activation)()
                else:
                    activation_fn = activation
                
                layers = [nn.Linear(in_size, mlp_size), activation_fn]
                for _ in range(num_layers - 1):
                    layers.extend([nn.Linear(mlp_size, mlp_size), activation_fn])
                layers.append(nn.Linear(mlp_size, out_size))
                
                if tanh:
                    layers.append(nn.Tanh())
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)


class GeneratorFunc(nn.Module):
    """
    Neural SDE function class that supplies drift (f) and diffusion (g) methods.
    
    Defines the stochastic differential equation:
    dX_t = f(t, X_t) dt + g(t, X_t) dW_t
    
    Originally from sigker_nsdes/src/gan/generators.py
    """

    def __init__(self, noise_size: int, hidden_size: int, mlp_size: int, num_layers: int, 
                 activation: str, tanh: bool = True, sde_type: str = "stratonovich", 
                 noise_type: str = "diagonal", tscale: float = 1.0):
        """
        Initialize the SDE function.

        Args:
            noise_size: Size of noise dimension (set to 1 if noise_type is "diagonal")
            hidden_size: Size of hidden state
            mlp_size: Number of neurons in each MLP layer
            num_layers: Number of layers in each MLP
            activation: Activation function name
            tanh: Whether to add tanh regularization to outputs
            sde_type: Type of integration ("stratonovich" or "ito")
            noise_type: Type of noise ("general", "diagonal")
            tscale: Clamp parameter for final tanh activation layer
        """
        super().__init__()

        self.sde_type = sde_type
        self.noise_type = noise_type

        if noise_type == "diagonal":
            self._noise_size = 1
        else:
            self._noise_size = noise_size

        self._hidden_size = hidden_size

        # Drift and diffusion networks
        # Input: [time, hidden_state] -> output: [hidden_size] for drift, [hidden_size * noise_size] for diffusion
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, activation, tanh=tanh, tscale=tscale)
        self._diffusion = MLP(1 + hidden_size, hidden_size * self._noise_size, mlp_size, num_layers, activation,
                              tanh=tanh, tscale=tscale)

    def f_and_g(self, t: torch.Tensor, x: torch.Tensor):
        """
        Compute drift and diffusion simultaneously.
        
        Args:
            t: Time tensor, shape ()
            x: State tensor, shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (drift, diffusion)
        """
        # t has shape (), x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        drift = self._drift(tx)

        if self.noise_type == "diagonal":
            diffusion = self._diffusion(tx)
        else:
            diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

        return drift, diffusion

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute drift term f(t, x)."""
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute diffusion term g(t, x)."""
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        if self.noise_type == "diagonal":
            diffusion = self._diffusion(tx)
        else:
            diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)
        return diffusion


class NeuralSDEGenerator(nn.Module):
    """
    Neural SDE Generator for pathwise data generation.
    
    Generates time series by solving a neural stochastic differential equation:
    dX_t = f(t, X_t; θ) dt + g(t, X_t; θ) dW_t
    
    Originally from sigker_nsdes/src/gan/generators.py
    """

    def __init__(self, data_size: int, initial_noise_size: int, noise_size: int, 
                 hidden_size: int, mlp_size: int, num_layers: int, activation: str = "LipSwish",
                 tanh: bool = True, tscale: float = 1.0, fixed: bool = True, 
                 noise_type: str = "general", sde_type: str = "stratonovich",
                 integration_method: str = "reversible_heun", dt_scale: float = 1.0):
        """
        Initialize the Neural SDE Generator.

        Args:
            data_size: Number of spatial dimensions in output data
            initial_noise_size: Size of initial noise source (ignored if fixed=True)
            noise_size: Size of noise dimension for SDE
            hidden_size: Size of hidden state dimension
            mlp_size: Number of neurons in each MLP layer
            num_layers: Number of layers in each MLP
            activation: Activation function name
            tanh: Whether to apply tanh regularization
            tscale: Scale clamp for final tanh activation
            fixed: Whether to fix the initial point
            noise_type: Type of noise ("general", "diagonal")
            sde_type: Method of stochastic integration ("stratonovich", "ito")
            integration_method: Integration method ("reversible_heun", etc.)
            dt_scale: Shrink factor on time grid
        """
        super().__init__()
        
        if not TORCHSDE_AVAILABLE:
            raise ImportError("torchsde is required for Neural SDE Generator")
        
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self._dt_scale = dt_scale
        self._noise_type = noise_type
        self._fixed = fixed

        # Initial condition network (if not fixed)
        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, activation, tanh=False)
        
        # SDE function (drift and diffusion)
        self._func = GeneratorFunc(
            noise_size, hidden_size, mlp_size, num_layers, activation, tanh, sde_type, noise_type, tscale
        )
        
        # Readout layer (hidden state -> data space)
        self._readout = nn.Linear(hidden_size, data_size)

        # SDE integration configuration
        self.sdeint_args = {"method": integration_method}
        if integration_method == "reversible_heun":
            self.sdeint_args["adjoint_method"] = "adjoint_reversible_heun"
            self.sdeint_func = torchsde.sdeint_adjoint
        else:
            self.sdeint_func = torchsde.sdeint

    def forward(self, ts: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Generate paths by solving the Neural SDE.
        
        Args:
            ts: Time points to evaluate SDE at, shape (t_size,)
            batch_size: Number of paths to generate
            
        Returns:
            Generated paths with linear interpolation coefficients
        """
        if not TORCHSDE_AVAILABLE:
            raise RuntimeError("torchsde not available")
        
        # Initial condition
        if self._fixed:
            x0 = torch.full(size=(batch_size, self._hidden_size), fill_value=1., device=ts.device)
        else:
            init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
            x0 = self._initial(init_noise)

        # Set time step for integration
        self.sdeint_args["dt"] = torch.diff(ts)[0] * self._dt_scale
        
        # Solve the SDE
        xs = self.sdeint_func(self._func, x0, ts, **self.sdeint_args)

        # Transpose to get (batch, time, hidden) shape
        xs = xs.transpose(0, 1)
        
        # Map to data space
        ys = self._readout(xs)

        # Add time dimension and create interpolation coefficients
        ts_expanded = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        
        if TORCHSDE_AVAILABLE:
            return torchcde.linear_interpolation_coeffs(torch.cat([ts_expanded, ys], dim=2))
        else:
            # Fallback: just return the paths
            return torch.cat([ts_expanded, ys], dim=2)

    def generate_paths(self, ts: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Generate raw paths without interpolation coefficients.
        
        Args:
            ts: Time points, shape (t_size,)
            batch_size: Number of paths to generate
            
        Returns:
            Generated paths, shape (batch_size, t_size, data_size + 1)
        """
        with torch.no_grad():
            return self.forward(ts, batch_size)


def create_nsde_generator(data_size: int = 2, hidden_size: int = 64, mlp_size: int = 128,
                         num_layers: int = 3, **kwargs) -> NeuralSDEGenerator:
    """
    Factory function to create a Neural SDE generator with sensible defaults.
    
    Args:
        data_size: Output data dimension
        hidden_size: Hidden state dimension
        mlp_size: MLP layer size
        num_layers: Number of MLP layers
        **kwargs: Additional arguments for NeuralSDEGenerator
        
    Returns:
        NeuralSDEGenerator instance
    """
    default_kwargs = {
        'initial_noise_size': hidden_size,
        'noise_size': hidden_size,
        'activation': 'LipSwish',
        'tanh': True,
        'tscale': 1.0,
        'fixed': True,
        'noise_type': 'diagonal',
        'sde_type': 'stratonovich',
        'integration_method': 'reversible_heun',
        'dt_scale': 1.0
    }
    
    # Update defaults with provided kwargs
    default_kwargs.update(kwargs)
    
    return NeuralSDEGenerator(
        data_size=data_size,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        num_layers=num_layers,
        **default_kwargs
    )

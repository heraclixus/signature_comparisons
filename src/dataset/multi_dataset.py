"""
Multi-Dataset Support for Signature-Based Models

This module provides support for training and evaluating models on multiple datasets:
1. Ornstein-Uhlenbeck (OU) process (original)
2. Heston stochastic volatility model
3. Rough Bergomi (rBergomi) model
4. Standard Brownian Motion
5. Fractional Brownian Motion with various Hurst parameters (0.3, 0.4, 0.6, 0.7)

Each dataset tests different aspects of stochastic process modeling.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.generative_model import gen_data, get_signal  # Original OU process
from dataset.Heston import HestonModel
from dataset.rBergomi import rBergomi
from dataset.fractional_brownian import create_fbm_datasets, get_fbm_dataset


def generate_heston_data(num_samples: int = 1000, n_points: int = 100, 
                        S0: float = 1.0, V0: float = 0.04, 
                        mu: float = 0.05, kappa: float = 2.0, 
                        theta: float = 0.04, sigma: float = 0.3, 
                        rho: float = -0.7, T: float = 1.0) -> torch.utils.data.TensorDataset:
    """
    Generate Heston stochastic volatility model data.
    
    The Heston model:
    dS_t = μ S_t dt + √V_t S_t dW1_t
    dV_t = κ(θ - V_t)dt + σ√V_t (ρ dW1_t + √(1-ρ²) dW2_t)
    
    Args:
        num_samples: Number of sample paths to generate
        n_points: Number of time points per path
        S0: Initial stock price
        V0: Initial variance
        mu: Drift parameter
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of volatility
        rho: Correlation between price and volatility
        T: Time horizon
        
    Returns:
        TensorDataset with Heston paths
    """
    print(f"Generating {num_samples} Heston model paths...")
    
    # Time grid
    dt = T / (n_points - 1)
    timeline = np.linspace(0, T, n_points)
    
    paths = []
    
    for i in range(num_samples):
        # Initialize
        S = np.zeros(n_points)
        V = np.zeros(n_points)
        S[0] = S0
        V[0] = V0
        
        # Generate correlated Brownian motions
        dW1 = np.random.randn(n_points - 1) * np.sqrt(dt)
        dW2 = np.random.randn(n_points - 1) * np.sqrt(dt)
        
        # Euler-Maruyama scheme
        for j in range(n_points - 1):
            # Ensure variance stays positive
            V_curr = max(V[j], 1e-6)
            
            # Update stock price
            S[j + 1] = S[j] + mu * S[j] * dt + np.sqrt(V_curr) * S[j] * dW1[j]
            
            # Update variance with correlation
            dW_V = rho * dW1[j] + np.sqrt(1 - rho**2) * dW2[j]
            V[j + 1] = V[j] + kappa * (theta - V_curr) * dt + sigma * np.sqrt(V_curr) * dW_V
            
            # Ensure variance stays positive
            V[j + 1] = max(V[j + 1], 1e-6)
        
        # Create path in same format as OU process: [time, values]
        path = np.stack([timeline, S])
        paths.append(path)
    
    paths = np.array(paths)
    print(f"✅ Generated Heston data: {paths.shape}")
    
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


def generate_rbergomi_data(num_samples: int = 1000, n_points: int = 100,
                          T: float = 1.0, a: float = -0.4, rho: float = -0.7,
                          eta: float = 1.5, xi: float = 0.235**2) -> torch.utils.data.TensorDataset:
    """
    Generate simplified rough Bergomi-like data.
    
    This creates a simplified version that captures the rough volatility characteristics
    without the full complexity of the rBergomi implementation.
    
    Args:
        num_samples: Number of sample paths to generate
        n_points: Number of time points per path
        T: Time horizon
        a: Hurst-like parameter (roughness)
        rho: Correlation parameter
        eta: Volatility of volatility
        xi: Initial variance
        
    Returns:
        TensorDataset with rBergomi-like paths
    """
    print(f"Generating {num_samples} simplified rBergomi-like paths...")
    
    dt = T / (n_points - 1)
    timeline = np.linspace(0, T, n_points)
    
    paths = []
    
    for i in range(num_samples):
        # Generate correlated Brownian motions
        dW1 = np.random.randn(n_points - 1) * np.sqrt(dt)
        dW2 = np.random.randn(n_points - 1) * np.sqrt(dt)
        
        # Initialize
        S = np.zeros(n_points)
        V = np.zeros(n_points)
        S[0] = 1.0
        V[0] = xi
        
        # Simplified rough volatility dynamics
        for j in range(n_points - 1):
            # Rough volatility with mean reversion
            V_curr = max(V[j], 1e-6)
            
            # Add roughness through enhanced volatility of volatility
            roughness_factor = (j + 1) ** a  # Rough scaling
            vol_vol = eta * np.sqrt(V_curr) * abs(roughness_factor)
            
            # Update variance
            dW_V = rho * dW1[j] + np.sqrt(1 - rho**2) * dW2[j]
            V[j + 1] = V[j] + vol_vol * dW_V
            V[j + 1] = max(V[j + 1], 1e-6)  # Keep positive
            
            # Update price
            S[j + 1] = S[j] * (1 + np.sqrt(V_curr) * dW1[j])
        
        # Create path
        path = np.stack([timeline, S])
        paths.append(path)
    
    paths = np.array(paths)
    print(f"✅ Generated simplified rBergomi data: {paths.shape}")
    
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


def generate_brownian_motion_data(num_samples: int = 1000, n_points: int = 100,
                                 T: float = 1.0, mu: float = 0.0, 
                                 sigma: float = 1.0) -> torch.utils.data.TensorDataset:
    """
    Generate standard Brownian motion data for comparison.
    
    Args:
        num_samples: Number of sample paths
        n_points: Number of time points
        T: Time horizon
        mu: Drift parameter
        sigma: Volatility parameter
        
    Returns:
        TensorDataset with Brownian motion paths
    """
    print(f"Generating {num_samples} Brownian motion paths...")
    
    dt = T / (n_points - 1)
    timeline = np.linspace(0, T, n_points)
    
    paths = []
    
    for i in range(num_samples):
        # Generate increments
        dW = np.random.randn(n_points - 1) * np.sqrt(dt)
        
        # Integrate to get path
        values = np.zeros(n_points)
        for j in range(n_points - 1):
            values[j + 1] = values[j] + mu * dt + sigma * dW[j]
        
        # Create path
        path = np.stack([timeline, values])
        paths.append(path)
    
    paths = np.array(paths)
    print(f"✅ Generated Brownian motion data: {paths.shape}")
    
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


class MultiDatasetManager:
    """
    Manager for handling multiple datasets in the signature framework.
    """
    
    def __init__(self):
        """Initialize multi-dataset manager."""
        self.datasets = {
            'ou_process': {
                'name': 'Ornstein-Uhlenbeck Process',
                'generator': get_signal,
                'description': 'Mean-reverting process (original baseline)',
                'params': {'num_samples': 1000, 'n_points': 100}
            },
            'heston': {
                'name': 'Heston Stochastic Volatility',
                'generator': generate_heston_data,
                'description': 'Stochastic volatility model for financial applications',
                'params': {'num_samples': 1000, 'n_points': 100, 'mu': 0.05, 'kappa': 2.0, 
                          'theta': 0.04, 'sigma': 0.3, 'rho': -0.7}
            },
            'rbergomi': {
                'name': 'Rough Bergomi Model',
                'generator': generate_rbergomi_data,
                'description': 'Rough volatility model with fractional Brownian motion',
                'params': {'num_samples': 1000, 'n_points': 100, 'a': -0.4, 'rho': -0.7,
                          'eta': 1.5, 'xi': 0.235**2}
            },
            'brownian': {
                'name': 'Standard Brownian Motion',
                'generator': generate_brownian_motion_data,
                'description': 'Simple Brownian motion for baseline comparison',
                'params': {'num_samples': 1000, 'n_points': 100, 'mu': 0.0, 'sigma': 1.0}
            },
            'fbm_h03': {
                'name': 'Fractional Brownian Motion H=0.3',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.3, **kwargs),
                'description': 'Anti-persistent FBM (mean-reverting, H < 0.5)',
                'params': {'num_samples': 1000, 'n_points': 100}
            },
            'fbm_h04': {
                'name': 'Fractional Brownian Motion H=0.4',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.4, **kwargs),
                'description': 'Anti-persistent FBM (mean-reverting, H < 0.5)',
                'params': {'num_samples': 1000, 'n_points': 100}
            },
            'fbm_h06': {
                'name': 'Fractional Brownian Motion H=0.6',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.6, **kwargs),
                'description': 'Persistent FBM (trending, H > 0.5)',
                'params': {'num_samples': 1000, 'n_points': 100}
            },
            'fbm_h07': {
                'name': 'Fractional Brownian Motion H=0.7',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.7, **kwargs),
                'description': 'Persistent FBM (trending, H > 0.5)',
                'params': {'num_samples': 1000, 'n_points': 100}
            }
        }
    
    def _generate_fbm_data(self, hurst: float, num_samples: int = 1000, n_points: int = 100, **kwargs) -> torch.utils.data.TensorDataset:
        """
        Generate Fractional Brownian Motion dataset.
        
        Args:
            hurst: Hurst parameter
            num_samples: Number of sample paths
            n_points: Number of time points per path
            
        Returns:
            TensorDataset with FBM paths
        """
        # Generate FBM data using our implementation
        fbm_data = get_fbm_dataset(hurst=hurst, num_samples=num_samples, n_points=n_points)
        
        # Convert to TensorDataset format
        # fbm_data shape: (num_samples, 2, n_points)
        # Create dummy labels (zeros)
        labels = torch.zeros(num_samples, dtype=torch.long)
        
        return torchdata.TensorDataset(fbm_data, labels)
    
    def get_dataset(self, dataset_name: str, **kwargs) -> torch.utils.data.TensorDataset:
        """
        Get a specific dataset.
        
        Args:
            dataset_name: Name of dataset ('ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07')
            **kwargs: Override default parameters
            
        Returns:
            TensorDataset for the specified dataset
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        dataset_info = self.datasets[dataset_name]
        params = dataset_info['params'].copy()
        params.update(kwargs)
        
        print(f"Generating {dataset_info['name']} dataset...")
        print(f"  Description: {dataset_info['description']}")
        print(f"  Parameters: {params}")
        
        return dataset_info['generator'](**params)
    
    def list_datasets(self):
        """List available datasets."""
        print("Available Datasets:")
        print("=" * 50)
        
        for name, info in self.datasets.items():
            print(f"📊 {name.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Default params: {info['params']}")
            print()
    
    def get_all_datasets(self, num_samples: int = 256) -> Dict[str, torch.utils.data.TensorDataset]:
        """
        Get all datasets with consistent sample count.
        
        Args:
            num_samples: Number of samples for each dataset
            
        Returns:
            Dictionary of all datasets
        """
        all_datasets = {}
        
        for dataset_name in self.datasets.keys():
            try:
                dataset = self.get_dataset(dataset_name, num_samples=num_samples)
                all_datasets[dataset_name] = dataset
                print(f"✅ {dataset_name}: {len(dataset)} samples")
            except Exception as e:
                print(f"❌ Failed to generate {dataset_name}: {e}")
        
        return all_datasets


def test_multi_dataset():
    """Test multi-dataset functionality."""
    print("🧪 Testing Multi-Dataset Functionality")
    print("=" * 50)
    
    manager = MultiDatasetManager()
    manager.list_datasets()
    
    # Test each dataset
    for dataset_name in ['ou_process', 'heston', 'rbergomi', 'brownian']:
        try:
            dataset = manager.get_dataset(dataset_name, num_samples=32)
            
            # Test data format
            sample_data = torch.stack([dataset[i][0] for i in range(4)])
            print(f"✅ {dataset_name}: Generated {len(dataset)} samples, shape {sample_data.shape}")
            
            # Check data properties
            values = sample_data[:, 1, :].numpy()  # Extract value dimension
            print(f"   Value range: [{values.min():.3f}, {values.max():.3f}]")
            print(f"   Mean: {values.mean():.3f}, Std: {values.std():.3f}")
            
        except Exception as e:
            print(f"❌ {dataset_name} test failed: {e}")
        
        print()


if __name__ == "__main__":
    test_multi_dataset()

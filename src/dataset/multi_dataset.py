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

# Import dataset persistence utilities
try:
    from utils.dataset_persistence import create_dataset_persistence
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    import warnings
    warnings.warn("Dataset persistence not available. Datasets will be generated on-the-fly.")


def generate_heston_data(num_samples: int = 1000, n_points: int = 100, 
                        S0: float = 1.0, V0: float = 0.04, 
                        mu: float = 0.05, kappa: float = 2.0, 
                        theta: float = 0.04, sigma: float = 0.3, 
                        rho: float = -0.7, T: float = 1.0) -> torch.utils.data.TensorDataset:
    """
    Generate Heston stochastic volatility model data using the proper HestonModel class.
    
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
    print(f"Generating {num_samples} Heston model paths using proper implementation...")
    print(f"  Parameters: μ={mu}, κ={kappa}, θ={theta}, σ={sigma}, ρ={rho}")
    
    # Create HestonModel instance
    heston_model = HestonModel(mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho)
    
    # Time grid
    timeline = torch.linspace(0, T, n_points, dtype=torch.float32)
    dt = T / (n_points - 1)
    
    paths = []
    
    # Generate paths using Euler-Maruyama scheme with the HestonModel
    for i in range(num_samples):
        # Initial conditions [S, V]
        y = torch.tensor([[S0, V0]], dtype=torch.float32)
        path_S = [S0]
        path_V = [V0]
        
        for j in range(n_points - 1):
            t = timeline[j]
            
            # Get drift and diffusion from HestonModel
            drift = heston_model.f(t, y)  # Shape: (1, 2)
            diffusion = heston_model.g(t, y)  # Shape: (1, 2, 2)
            
            # Generate noise
            dW = torch.randn(1, 2) * np.sqrt(dt)  # Two-dimensional noise
            
            # Euler-Maruyama step
            dy = drift * dt + torch.sum(diffusion * dW.unsqueeze(1), dim=2)
            y = y + dy
            
            # Ensure variance stays positive
            y[0, 1] = torch.clamp(y[0, 1], min=1e-6)
            
            path_S.append(y[0, 0].item())
            path_V.append(y[0, 1].item())
        
        # Create path in expected format [time, price_values]
        path = np.stack([timeline.numpy(), np.array(path_S)])
        paths.append(path)
    
    paths = np.array(paths)
    print(f"✅ Generated proper Heston data: {paths.shape}")
    print(f"  Time horizon: [0, {T}], Points: {n_points}")
    print(f"  Price range: [{paths[:, 1, :].min():.3f}, {paths[:, 1, :].max():.3f}]")
    print(f"  Final variance range: [{np.array([path_V]).min():.6f}, {np.array([path_V]).max():.6f}]")
    
    return torchdata.TensorDataset(torch.tensor(paths, dtype=torch.float))


def generate_rbergomi_data(num_samples: int = 1000, n_points: int = 100,
                          T: float = 1.0, a: float = -0.4, rho: float = -0.7,
                          eta: float = 1.5, xi: float = 0.235**2) -> torch.utils.data.TensorDataset:
    """
    Generate proper rough Bergomi data using the rBergomi class implementation.
    
    This uses the full mathematical rBergomi model with Volterra processes
    and proper fractional Brownian motion characteristics.
    
    Args:
        num_samples: Number of sample paths to generate
        n_points: Number of time points per path
        T: Time horizon
        a: Roughness parameter (alpha)
        rho: Correlation parameter between price and volatility
        eta: Volatility of volatility
        xi: Initial variance
        
    Returns:
        TensorDataset with proper rBergomi paths
    """
    print(f"Generating {num_samples} rBergomi paths using full implementation...")
    print(f"  Parameters: a={a}, rho={rho}, eta={eta}, xi={xi:.6f}")
    
    # Create rBergomi model instance
    rbergomi_model = rBergomi(n=n_points, N=num_samples, T=T, a=a, rho=rho, eta=eta, xi=xi)
    
    # Generate the full rBergomi process
    print(f"  🔬 Generating Volterra process...")
    dW1 = rbergomi_model.dW1()  # Correlated 2D Brownian increments
    Y = rbergomi_model.Y(dW1)   # Volterra process
    
    print(f"  📈 Computing variance process...")
    V = rbergomi_model.V(Y, xi=xi, eta=eta)  # Variance process
    
    print(f"  💰 Computing price process...")
    dW2 = rbergomi_model.dW2()  # Orthogonal increments
    dB = rbergomi_model.dB(dW1, dW2, rho=rho)  # Correlated price Brownian
    S = rbergomi_model.S(V, dB, S0=1.0)  # Price process
    
    # Create paths in the expected format [time, values]
    paths = []
    timeline = rbergomi_model.t.flatten()  # Time grid
    
    for i in range(num_samples):
        path = np.stack([timeline, S[i, :]])
        paths.append(path)
    
    paths = np.array(paths)
    print(f"✅ Generated proper rBergomi data: {paths.shape}")
    print(f"  Time horizon: [0, {T}], Points: {len(timeline)}")
    print(f"  Price range: [{S.min():.3f}, {S.max():.3f}]")
    print(f"  Variance range: [{V.min():.6f}, {V.max():.6f}]")
    
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
    
    def __init__(self, use_persistence: bool = True):
        """
        Initialize multi-dataset manager.
        
        Args:
            use_persistence: If True, try to load datasets from disk before generating
        """
        self.use_persistence = use_persistence and PERSISTENCE_AVAILABLE
        
        # Initialize persistence manager if available
        if self.use_persistence:
            try:
                self.persistence = create_dataset_persistence()
                print(f"📁 Dataset persistence enabled: {self.persistence.data_root}")
            except Exception as e:
                print(f"⚠️ Failed to initialize dataset persistence: {e}")
                self.use_persistence = False
                self.persistence = None
        else:
            self.persistence = None
        
        self.datasets = {
            'ou_process': {
                'name': 'Ornstein-Uhlenbeck Process',
                'generator': get_signal,
                'description': 'Mean-reverting process (original baseline)',
                'params': {'num_samples': 32768, 'n_points': 64}  # 128 * 256 = 32768 paths, 64 time points
            },
            'heston': {
                'name': 'Heston Stochastic Volatility',
                'generator': generate_heston_data,
                'description': 'Stochastic volatility model for financial applications',
                'params': {'num_samples': 32768, 'n_points': 64, 'mu': 0.05, 'kappa': 2.0, 
                          'theta': 0.04, 'sigma': 0.3, 'rho': -0.7}
            },
            'rbergomi': {
                'name': 'Rough Bergomi Model',
                'generator': generate_rbergomi_data,
                'description': 'Rough volatility model with fractional Brownian motion',
                'params': {'num_samples': 32768, 'n_points': 64, 'a': -0.4, 'rho': -0.7,
                          'eta': 1.5, 'xi': 0.235**2}
            },
            'brownian': {
                'name': 'Standard Brownian Motion',
                'generator': generate_brownian_motion_data,
                'description': 'Simple Brownian motion for baseline comparison',
                'params': {'num_samples': 32768, 'n_points': 64, 'mu': 0.0, 'sigma': 1.0}
            },
            'fbm_h03': {
                'name': 'Fractional Brownian Motion H=0.3',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.3, **kwargs),
                'description': 'Anti-persistent FBM (mean-reverting, H < 0.5)',
                'params': {'num_samples': 32768, 'n_points': 64}
            },
            'fbm_h04': {
                'name': 'Fractional Brownian Motion H=0.4',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.4, **kwargs),
                'description': 'Anti-persistent FBM (mean-reverting, H < 0.5)',
                'params': {'num_samples': 32768, 'n_points': 64}
            },
            'fbm_h06': {
                'name': 'Fractional Brownian Motion H=0.6',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.6, **kwargs),
                'description': 'Persistent FBM (trending, H > 0.5)',
                'params': {'num_samples': 32768, 'n_points': 64}
            },
            'fbm_h07': {
                'name': 'Fractional Brownian Motion H=0.7',
                'generator': lambda **kwargs: self._generate_fbm_data(hurst=0.7, **kwargs),
                'description': 'Persistent FBM (trending, H > 0.5)',
                'params': {'num_samples': 32768, 'n_points': 64}
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
        Get a specific dataset, loading from disk if available.
        
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
        
        # Remove batch_size from params as it's not used by dataset generators
        generation_params = {k: v for k, v in params.items() if k != 'batch_size'}
        
        # Try to load from disk first if persistence is enabled
        if self.use_persistence and self.persistence is not None:
            print(f"🔍 Checking for saved {dataset_info['name']} dataset...")
            
            saved_data = self.persistence.load_dataset(dataset_name, params)
            if saved_data is not None:
                print(f"✅ Loaded {dataset_info['name']} from disk")
                print(f"  Shape: {saved_data.shape}")
                
                # Convert to TensorDataset format
                if dataset_name == 'ou_process':
                    # OU process needs dummy labels
                    dummy_labels = torch.zeros(saved_data.shape[0], saved_data.shape[2] - 1, dtype=torch.float)
                    return torchdata.TensorDataset(saved_data, dummy_labels)
                else:
                    # Other datasets need dummy labels too
                    dummy_labels = torch.zeros(saved_data.shape[0], dtype=torch.long)
                    return torchdata.TensorDataset(saved_data, dummy_labels)
        
        # Generate dataset if not found on disk
        print(f"🏭 Generating {dataset_info['name']} dataset...")
        print(f"  Description: {dataset_info['description']}")
        print(f"  Parameters: {generation_params}")
        
        dataset = dataset_info['generator'](**generation_params)
        
        # Save to disk if persistence is enabled
        if self.use_persistence and self.persistence is not None:
            try:
                # Extract data tensor from TensorDataset
                if hasattr(dataset, 'tensors'):
                    data_tensor = dataset.tensors[0]
                else:
                    data_tensor = torch.stack([dataset[i][0] for i in range(len(dataset))])
                
                # Save dataset
                metadata = {
                    'dataset_type': dataset_name,
                    'generation_method': 'MultiDatasetManager',
                    'description': dataset_info['description']
                }
                
                self.persistence.save_dataset(
                    dataset_name=dataset_name,
                    data=data_tensor,
                    params=params,
                    metadata=metadata
                )
                print(f"💾 Saved {dataset_info['name']} to disk for future use")
                
            except Exception as e:
                print(f"⚠️ Failed to save dataset to disk: {e}")
        
        return dataset
    
    def list_datasets(self):
        """List available datasets."""
        print("Available Datasets:")
        print("=" * 50)
        
        for name, info in self.datasets.items():
            print(f"📊 {name.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Default params: {info['params']}")
            
            # Check if saved version exists
            if self.use_persistence and self.persistence is not None:
                if self.persistence.dataset_exists(name, info['params']):
                    print(f"   💾 Saved version available")
                else:
                    print(f"   🏭 Will be generated on-demand")
            else:
                print(f"   🏭 Will be generated on-demand")
            print()
    
    def list_saved_datasets(self):
        """List all saved datasets."""
        if not self.use_persistence or self.persistence is None:
            print("❌ Dataset persistence not available")
            return
        
        print("💾 Saved Datasets:")
        print("=" * 50)
        
        saved_datasets = self.persistence.list_saved_datasets()
        if not saved_datasets:
            print("   No datasets currently saved.")
            return
        
        for dataset_key, metadata in saved_datasets.items():
            print(f"📊 {metadata['dataset_name'].upper()}")
            print(f"   Shape: {metadata['data_shape']}")
            print(f"   Size: {metadata['file_size_mb']:.2f} MB")
            print(f"   Created: {metadata['creation_timestamp']}")
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

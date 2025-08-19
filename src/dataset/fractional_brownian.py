"""
Fractional Brownian Motion Dataset Implementation

This module implements Fractional Brownian Motion (FBM) datasets using the stochastic package.
FBM is characterized by the Hurst parameter H:
- H = 0.5: Standard Brownian Motion
- H < 0.5: Anti-persistent (mean-reverting)
- H > 0.5: Persistent (trending)

Different Hurst parameters provide diverse stochastic behaviors for model testing.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import warnings

try:
    from stochastic.processes.continuous import FractionalBrownianMotion
    STOCHASTIC_AVAILABLE = True
except ImportError:
    STOCHASTIC_AVAILABLE = False
    warnings.warn("stochastic package not available. FBM datasets will use fallback implementation.")


class FractionalBrownianDataset:
    """
    Fractional Brownian Motion dataset generator using the stochastic package.
    
    Generates FBM paths with specified Hurst parameter for model evaluation.
    """
    
    def __init__(self, hurst: float = 0.5, t_max: float = 1.0, n_points: int = 100):
        """
        Initialize FBM dataset generator.
        
        Args:
            hurst: Hurst parameter (0 < H < 1)
                  H = 0.5: Standard Brownian Motion
                  H < 0.5: Anti-persistent (mean-reverting)
                  H > 0.5: Persistent (trending)
            t_max: Maximum time for the process
            n_points: Number of time points in each path
        """
        self.hurst = hurst
        self.t_max = t_max
        self.n_points = n_points
        
        # Validate Hurst parameter
        if not (0 < hurst < 1):
            raise ValueError(f"Hurst parameter must be in (0, 1), got {hurst}")
        
        if STOCHASTIC_AVAILABLE:
            # Create FBM process using stochastic package
            self.fbm_process = FractionalBrownianMotion(hurst=hurst, t=t_max)
            self.implementation = "stochastic_package"
            print(f"✅ FBM dataset created using stochastic package: H={hurst}")
        else:
            # Fallback implementation
            self.implementation = "fallback"
            warnings.warn(f"Using fallback FBM implementation for H={hurst}")
    
    def generate_sample(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a single FBM sample path.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            FBM path as tensor, shape (2, n_points) with [time, values]
        """
        if seed is not None:
            np.random.seed(seed)
        
        if STOCHASTIC_AVAILABLE:
            # Generate using stochastic package
            values = self.fbm_process.sample(self.n_points)
            times = self.fbm_process.times(self.n_points)
        else:
            # Fallback: approximate FBM using standard Brownian motion
            times = np.linspace(0, self.t_max, self.n_points)
            # Simple approximation - not true FBM but better than nothing
            values = np.cumsum(np.random.randn(self.n_points) * np.sqrt(self.t_max / self.n_points))
            warnings.warn("Using Brownian motion approximation for FBM")
        
        # Convert to tensor format
        path = torch.tensor(np.stack([times, values]), dtype=torch.float32)
        return path
    
    def generate_batch(self, batch_size: int, seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate a batch of FBM sample paths.
        
        Args:
            batch_size: Number of paths to generate
            seed: Random seed for reproducibility
            
        Returns:
            Batch of FBM paths, shape (batch_size, 2, n_points)
        """
        if seed is not None:
            np.random.seed(seed)
        
        paths = []
        for i in range(batch_size):
            # Use different seed for each path if seed is provided
            path_seed = seed + i if seed is not None else None
            path = self.generate_sample(path_seed)
            paths.append(path)
        
        return torch.stack(paths)
    
    def get_process_info(self) -> dict:
        """Get information about the FBM process."""
        return {
            'process_type': 'Fractional Brownian Motion',
            'hurst_parameter': self.hurst,
            'time_horizon': self.t_max,
            'time_points': self.n_points,
            'implementation': self.implementation,
            'properties': {
                'self_similar': True,
                'stationary_increments': True,
                'gaussian': True,
                'markov': self.hurst == 0.5,  # Only for standard BM
                'persistence': 'anti-persistent' if self.hurst < 0.5 else 'persistent' if self.hurst > 0.5 else 'neutral'
            }
        }


def create_fbm_datasets(hurst_values: List[float] = [0.3, 0.4, 0.6, 0.7], 
                       num_samples: int = 256, n_points: int = 100) -> dict:
    """
    Create multiple FBM datasets with different Hurst parameters.
    
    Args:
        hurst_values: List of Hurst parameters to generate
        num_samples: Number of samples per dataset
        n_points: Number of time points per path
        
    Returns:
        Dictionary with dataset name as key and tensor data as value
    """
    print(f"🌊 Creating Fractional Brownian Motion Datasets")
    print(f"   Hurst parameters: {hurst_values}")
    print(f"   Samples per dataset: {num_samples}")
    print(f"   Time points: {n_points}")
    
    datasets = {}
    
    for hurst in hurst_values:
        print(f"\n📊 Generating FBM dataset with H={hurst}...")
        
        # Create FBM dataset generator
        fbm_dataset = FractionalBrownianDataset(hurst=hurst, t_max=1.0, n_points=n_points)
        
        # Generate batch of samples
        batch_data = fbm_dataset.generate_batch(num_samples, seed=42)
        
        # Create dataset name
        dataset_name = f'fbm_h{str(hurst).replace(".", "")}'  # e.g., 'fbm_h03', 'fbm_h07'
        
        # Store dataset
        datasets[dataset_name] = batch_data
        
        # Print dataset info
        info = fbm_dataset.get_process_info()
        print(f"   ✅ {dataset_name}: {batch_data.shape}")
        print(f"      Properties: {info['properties']['persistence']}")
        print(f"      Implementation: {info['implementation']}")
    
    print(f"\n✅ Created {len(datasets)} FBM datasets")
    return datasets


def get_fbm_dataset(hurst: float, num_samples: int = 256, n_points: int = 100) -> torch.Tensor:
    """
    Get a single FBM dataset with specified Hurst parameter.
    
    Args:
        hurst: Hurst parameter
        num_samples: Number of samples
        n_points: Number of time points
        
    Returns:
        FBM dataset tensor, shape (num_samples, 2, n_points)
    """
    fbm_dataset = FractionalBrownianDataset(hurst=hurst, t_max=1.0, n_points=n_points)
    return fbm_dataset.generate_batch(num_samples, seed=42)


def analyze_fbm_properties(datasets: dict):
    """
    Analyze the statistical properties of FBM datasets.
    
    Args:
        datasets: Dictionary of FBM datasets
    """
    print(f"\n🔬 Analyzing FBM Dataset Properties")
    print("=" * 50)
    
    for dataset_name, data in datasets.items():
        # Extract Hurst parameter from name
        hurst_str = dataset_name.replace('fbm_h', '')
        if len(hurst_str) == 2:  # e.g., '03' -> 0.3, '07' -> 0.7
            hurst = float(f"0.{hurst_str}")
        else:  # Handle other formats
            hurst = float(hurst_str) / 10 if float(hurst_str) > 1 else float(hurst_str)
        
        print(f"\n📈 {dataset_name.upper()} (H={hurst}):")
        
        # Extract values (remove time dimension)
        values = data[:, 1, :].numpy()  # Shape: (samples, time_points)
        
        # Compute statistics
        final_values = values[:, -1]  # Final values
        increments = np.diff(values, axis=1)  # Increments
        
        print(f"   Final value stats: mean={np.mean(final_values):.3f}, std={np.std(final_values):.3f}")
        print(f"   Increment stats: mean={np.mean(increments):.6f}, std={np.std(increments):.3f}")
        print(f"   Path variance: {np.var(values):.3f}")
        
        # Theoretical properties
        theoretical_variance = hurst * 2  # Simplified
        persistence = "Anti-persistent" if hurst < 0.5 else "Persistent" if hurst > 0.5 else "Neutral"
        print(f"   Expected behavior: {persistence}")
        print(f"   Long-range dependence: {'Yes' if hurst != 0.5 else 'No'}")


def test_fbm_implementation():
    """Test the FBM implementation."""
    print("🧪 Testing Fractional Brownian Motion Implementation")
    print("=" * 60)
    
    if not STOCHASTIC_AVAILABLE:
        print("❌ stochastic package not available")
        return False
    
    try:
        # Test different Hurst parameters
        test_hurst_values = [0.3, 0.5, 0.7]
        
        for hurst in test_hurst_values:
            print(f"\n📊 Testing H={hurst}...")
            
            # Create FBM dataset
            fbm_dataset = FractionalBrownianDataset(hurst=hurst, n_points=50)
            
            # Generate sample
            sample = fbm_dataset.generate_sample(seed=123)
            print(f"   Sample shape: {sample.shape}")
            
            # Generate batch
            batch = fbm_dataset.generate_batch(5, seed=123)
            print(f"   Batch shape: {batch.shape}")
            
            # Check properties
            info = fbm_dataset.get_process_info()
            print(f"   Properties: {info['properties']['persistence']}")
        
        print(f"\n✅ FBM implementation test successful!")
        return True
        
    except Exception as e:
        print(f"❌ FBM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test FBM implementation
    test_success = test_fbm_implementation()
    
    if test_success:
        print(f"\n" + "="*60)
        print("🌊 Creating FBM Datasets for Model Testing")
        print("="*60)
        
        # Create FBM datasets
        fbm_datasets = create_fbm_datasets()
        
        # Analyze properties
        analyze_fbm_properties(fbm_datasets)
        
        print(f"\n🎉 FBM datasets ready for model training and evaluation!")
        print(f"   Use these datasets to test model performance on different stochastic behaviors")
        print(f"   H < 0.5: Tests anti-persistent (mean-reverting) behavior")
        print(f"   H > 0.5: Tests persistent (trending) behavior")

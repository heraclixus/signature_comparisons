"""
Moving MNIST Time Series Dataset

This module provides Moving MNIST dataset integration with the existing
time series infrastructure. It converts video sequences to time series
format compatible with signature analysis methods.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
from typing import Tuple, Dict, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.moving_mnist import MovingMNIST


def extract_center_of_mass_trajectory(sequence: np.ndarray) -> np.ndarray:
    """
    Extract center-of-mass trajectory from a Moving MNIST sequence.
    
    Args:
        sequence: Video sequence of shape (seq_len, height, width, channels)
        
    Returns:
        Trajectory of shape (seq_len, 2) for (x, y) coordinates
    """
    seq_len, height, width, channels = sequence.shape
    trajectory = np.zeros((seq_len, 2))
    
    for t in range(seq_len):
        frame = sequence[t, :, :, 0]  # Get grayscale frame
        
        if frame.sum() > 0:
            # Calculate center of mass
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            total_mass = frame.sum()
            center_y = (y_coords * frame).sum() / total_mass
            center_x = (x_coords * frame).sum() / total_mass
            trajectory[t] = [center_x, center_y]
        else:
            trajectory[t] = [0.0, 0.0]  # Fallback
    
    return trajectory


def extract_multiple_trajectories(sequence: np.ndarray, num_digits: int) -> np.ndarray:
    """
    Extract trajectories for multiple digits using connected component analysis.
    
    Args:
        sequence: Video sequence of shape (seq_len, height, width, channels)
        num_digits: Expected number of digits
        
    Returns:
        Trajectories of shape (seq_len, num_digits, 2)
    """
    seq_len, height, width, channels = sequence.shape
    trajectories = np.zeros((seq_len, num_digits, 2))
    
    try:
        from scipy import ndimage
        scipy_available = True
    except ImportError:
        scipy_available = False
        # Fallback to simple center of mass
        single_trajectory = extract_center_of_mass_trajectory(sequence)
        # Replicate for all digits (not ideal but works as fallback)
        for i in range(num_digits):
            trajectories[:, i] = single_trajectory
        return trajectories
    
    for t in range(seq_len):
        frame = sequence[t, :, :, 0]
        
        # Threshold the image to create binary mask
        binary = frame > 0.1
        
        if binary.sum() > 0:
            # Find connected components
            labeled, num_features = ndimage.label(binary)
            
            centers = []
            for i in range(1, min(num_features + 1, num_digits + 1)):
                mask = labeled == i
                if mask.sum() > 0:
                    y_coords, x_coords = np.mgrid[0:height, 0:width]
                    center_y = (y_coords * mask).sum() / mask.sum()
                    center_x = (x_coords * mask).sum() / mask.sum()
                    centers.append([center_x, center_y])
            
            # Fill trajectories (pad with zeros if fewer digits found)
            for i in range(num_digits):
                if i < len(centers):
                    trajectories[t, i] = centers[i]
                else:
                    trajectories[t, i] = [0.0, 0.0]
    
    return trajectories


def generate_moving_mnist_time_series(num_samples: int = 1000, seq_len: int = 20,
                                    num_digits: int = 2, image_size: int = 64,
                                    deterministic: bool = True, 
                                    data_root: str = "./data",
                                    extraction_method: str = "center_of_mass") -> torchdata.TensorDataset:
    """
    Generate Moving MNIST dataset in time series format.
    
    Args:
        num_samples: Number of video sequences to generate
        seq_len: Length of each video sequence
        num_digits: Number of digits per sequence
        image_size: Size of the canvas
        deterministic: Whether to use deterministic movement
        data_root: Root directory for MNIST data
        extraction_method: Method for extracting time series
                          - "center_of_mass": Single trajectory from center of mass
                          - "multi_trajectory": Multiple trajectories (one per digit)
                          - "flattened": Flattened pixel values
        
    Returns:
        TensorDataset with time series data
    """
    print(f"🎬 Generating Moving MNIST Time Series Dataset")
    print(f"   Samples: {num_samples}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Digits per sequence: {num_digits}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Extraction method: {extraction_method}")
    
    # Create MovingMNIST dataset
    mnist_dataset = MovingMNIST(
        train=True,
        data_root=data_root,
        seq_len=seq_len,
        num_digits=num_digits,
        image_size=image_size,
        deterministic=deterministic
    )
    
    # Generate sequences and convert to time series
    time_series_data = []
    
    for i in range(num_samples):
        if i % 500 == 0 and i > 0:
            print(f"   Processed {i}/{num_samples} sequences...")
        
        # Get video sequence: (seq_len, height, width, channels)
        sequence = mnist_dataset[i]
        
        # Extract time series based on method
        if extraction_method == "center_of_mass":
            # Single trajectory: (seq_len, 2)
            trajectory = extract_center_of_mass_trajectory(sequence)
            # Convert to (2, seq_len) format
            time_series = trajectory.T  # Shape: (2, seq_len)
            
        elif extraction_method == "multi_trajectory":
            # Multiple trajectories: (seq_len, num_digits, 2)
            trajectories = extract_multiple_trajectories(sequence, num_digits)
            # Flatten to (seq_len, num_digits * 2) then transpose
            flattened = trajectories.reshape(seq_len, -1)
            time_series = flattened.T  # Shape: (num_digits * 2, seq_len)
            
        elif extraction_method == "flattened":
            # Flatten frames: (seq_len, height * width)
            flattened = sequence.reshape(seq_len, -1)
            time_series = flattened.T  # Shape: (height * width, seq_len)
            
        else:
            raise ValueError(f"Unknown extraction method: {extraction_method}")
        
        time_series_data.append(torch.tensor(time_series, dtype=torch.float32))
    
    # Stack all time series
    data_tensor = torch.stack(time_series_data)  # Shape: (num_samples, channels, seq_len)
    
    print(f"✅ Generated Moving MNIST time series")
    print(f"   Final shape: {data_tensor.shape}")
    print(f"   Data type: {data_tensor.dtype}")
    print(f"   Value range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    
    return torchdata.TensorDataset(data_tensor)


def get_moving_mnist_dataset_info(extraction_method: str = "center_of_mass", 
                                num_digits: int = 2) -> Dict[str, Any]:
    """Get dataset information for Moving MNIST."""
    
    if extraction_method == "center_of_mass":
        channels = 2
        channel_meaning = "x_y_coordinates"
        description = f"Moving MNIST with {num_digits} bouncing digits, center-of-mass trajectory extraction"
        
    elif extraction_method == "multi_trajectory":
        channels = num_digits * 2
        channel_meaning = f"{num_digits}_digit_x_y_coordinates"
        description = f"Moving MNIST with {num_digits} bouncing digits, individual trajectory extraction"
        
    elif extraction_method == "flattened":
        channels = 64 * 64  # Assuming standard image size
        channel_meaning = "flattened_pixel_values"
        description = f"Moving MNIST with {num_digits} bouncing digits, flattened frame representation"
        
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")
    
    return {
        'name': f'Moving MNIST ({extraction_method})',
        'description': description,
        'channels': channels,
        'channel_meaning': channel_meaning,
        'data_type': 'video_sequence_converted_to_time_series',
        'physics': 'bouncing_ball_dynamics',
        'stochastic_properties': {
            'stationary': False,
            'mean_reverting': False,
            'bounded': True,  # Bounded by canvas size
            'periodic': False,
            'chaotic': True   # Due to bouncing dynamics
        }
    }


# Dataset configurations for integration with MultiDatasetManager
MOVING_MNIST_CONFIGS = {
    'moving_mnist_trajectory': {
        'name': 'Moving MNIST (Trajectory)',
        'description': 'Moving MNIST with center-of-mass trajectory extraction (2D time series)',
        'generator': lambda **kwargs: generate_moving_mnist_time_series(
            extraction_method="center_of_mass", **kwargs
        ),
        'default_params': {
            'num_samples': 1000,
            'seq_len': 20,
            'num_digits': 2,
            'image_size': 64,
            'deterministic': True
        }
    },
    'moving_mnist_single': {
        'name': 'Moving MNIST (Single Digit)',
        'description': 'Moving MNIST with single digit trajectory (2D time series)',
        'generator': lambda **kwargs: generate_moving_mnist_time_series(
            num_digits=1, extraction_method="center_of_mass", **kwargs
        ),
        'default_params': {
            'num_samples': 1000,
            'seq_len': 25,
            'num_digits': 1,
            'image_size': 64,
            'deterministic': True
        }
    },
    'moving_mnist_multi': {
        'name': 'Moving MNIST (Multi-Trajectory)',
        'description': 'Moving MNIST with multiple digit trajectories (4D time series for 2 digits)',
        'generator': lambda **kwargs: generate_moving_mnist_time_series(
            extraction_method="multi_trajectory", **kwargs
        ),
        'default_params': {
            'num_samples': 1000,
            'seq_len': 20,
            'num_digits': 2,
            'image_size': 64,
            'deterministic': True
        }
    }
}


def test_moving_mnist_time_series():
    """Test Moving MNIST time series generation."""
    print("🧪 Testing Moving MNIST Time Series Generation")
    print("=" * 60)
    
    # Test different extraction methods
    methods = ["center_of_mass", "multi_trajectory"]
    
    for method in methods:
        print(f"\n🔬 Testing {method} extraction...")
        
        try:
            dataset = generate_moving_mnist_time_series(
                num_samples=10,  # Small test
                seq_len=15,
                num_digits=2,
                extraction_method=method,
                data_root="../../data"
            )
            
            # Get sample
            sample = dataset[0][0]  # First sample, first element of tuple
            
            print(f"✅ {method} successful:")
            print(f"   Dataset size: {len(dataset)}")
            print(f"   Sample shape: {sample.shape}")
            print(f"   Data type: {sample.dtype}")
            print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
            
            # Get dataset info
            info = get_moving_mnist_dataset_info(method, num_digits=2)
            print(f"   Description: {info['description']}")
            print(f"   Channels: {info['channels']} ({info['channel_meaning']})")
            
        except Exception as e:
            print(f"❌ {method} failed: {e}")
    
    print(f"\n🎉 Moving MNIST time series testing completed!")


if __name__ == "__main__":
    test_moving_mnist_time_series()

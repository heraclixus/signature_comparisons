"""
Moving MNIST Video Dataset

This module provides Moving MNIST dataset integration that preserves
the raw video sequence format for encoder/decoder architectures.

Unlike the trajectory version, this keeps the full (seq_len, height, width, channels)
format suitable for video processing and latent space encoding.
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


class MovingMNISTVideoDataset(torchdata.Dataset):
    """
    Moving MNIST dataset that preserves raw video sequences.
    
    Returns video sequences in format: (seq_len, height, width, channels)
    Perfect for encoder/decoder architectures and latent space modeling.
    """
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 20,
                 num_digits: int = 2, image_size: int = 64,
                 deterministic: bool = True, data_root: str = "./data",
                 train: bool = True):
        """
        Initialize Moving MNIST video dataset.
        
        Args:
            num_samples: Number of video sequences to generate
            seq_len: Length of each video sequence
            num_digits: Number of digits per sequence
            image_size: Size of the canvas
            deterministic: Whether to use deterministic movement
            data_root: Root directory for MNIST data
            train: Whether to use train or test split
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.deterministic = deterministic
        self.data_root = data_root
        self.train = train
        
        # Create underlying MovingMNIST dataset
        self.mnist_dataset = MovingMNIST(
            train=train,
            data_root=data_root,
            seq_len=seq_len,
            num_digits=num_digits,
            image_size=image_size,
            deterministic=deterministic
        )
        
        print(f"🎬 MovingMNISTVideoDataset initialized:")
        print(f"   Samples: {num_samples}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Digits per sequence: {num_digits}")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Format: (seq_len, height, width, channels)")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video sequence.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (video_sequence, label)
            - video_sequence: Shape (seq_len, height, width, channels)
            - label: Dummy label (0) for compatibility
        """
        # Get raw video sequence from MovingMNIST
        sequence = self.mnist_dataset[idx]  # Shape: (seq_len, height, width, channels)
        
        # Convert to torch tensor
        video_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        # Create dummy label for compatibility with existing infrastructure
        label = torch.tensor(0, dtype=torch.long)
        
        return video_tensor, label
    
    def get_sequence_info(self) -> Dict[str, Any]:
        """Get information about the video sequences."""
        return {
            'sequence_shape': (self.seq_len, self.image_size, self.image_size, 1),
            'num_samples': self.num_samples,
            'num_digits': self.num_digits,
            'deterministic': self.deterministic,
            'data_type': 'video_sequence',
            'memory_per_sequence_kb': self.seq_len * self.image_size * self.image_size * 4 / 1024,  # float32
            'total_memory_mb': self.num_samples * self.seq_len * self.image_size * self.image_size * 4 / (1024*1024)
        }


def generate_moving_mnist_video_dataset(num_samples: int = 1000, seq_len: int = 20,
                                       num_digits: int = 2, image_size: int = 64,
                                       deterministic: bool = True, 
                                       data_root: str = "./data",
                                       train: bool = True) -> torchdata.TensorDataset:
    """
    Generate Moving MNIST dataset in raw video format.
    
    Args:
        num_samples: Number of video sequences to generate
        seq_len: Length of each video sequence
        num_digits: Number of digits per sequence
        image_size: Size of the canvas
        deterministic: Whether to use deterministic movement
        data_root: Root directory for MNIST data
        train: Whether to use train or test split
        
    Returns:
        TensorDataset with video sequences
    """
    print(f"🎬 Generating Moving MNIST Video Dataset")
    print(f"   Samples: {num_samples}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Digits per sequence: {num_digits}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Format: Raw video sequences (no trajectory extraction)")
    
    # Create MovingMNIST dataset
    mnist_dataset = MovingMNIST(
        train=train,
        data_root=data_root,
        seq_len=seq_len,
        num_digits=num_digits,
        image_size=image_size,
        deterministic=deterministic
    )
    
    # Generate sequences
    video_sequences = []
    
    for i in range(num_samples):
        if i % 500 == 0 and i > 0:
            print(f"   Generated {i}/{num_samples} sequences...")
        
        # Get video sequence: (seq_len, height, width, channels)
        sequence = mnist_dataset[i]
        video_sequences.append(torch.tensor(sequence, dtype=torch.float32))
    
    # Stack all video sequences
    data_tensor = torch.stack(video_sequences)  # Shape: (num_samples, seq_len, height, width, channels)
    
    print(f"✅ Generated Moving MNIST video dataset")
    print(f"   Final shape: {data_tensor.shape}")
    print(f"   Data type: {data_tensor.dtype}")
    print(f"   Value range: [{data_tensor.min():.3f}, {data_tensor.max():.3f}]")
    print(f"   Memory: {data_tensor.numel() * 4 / (1024*1024):.1f} MB")
    
    # Create dummy labels
    labels = torch.zeros(num_samples, dtype=torch.long)
    
    return torchdata.TensorDataset(data_tensor, labels)


def get_moving_mnist_video_info(num_digits: int = 2, seq_len: int = 20, 
                               image_size: int = 64) -> Dict[str, Any]:
    """Get dataset information for Moving MNIST video."""
    
    return {
        'name': f'Moving MNIST Video ({num_digits} digits)',
        'description': f'Moving MNIST with {num_digits} bouncing digits, raw video sequences',
        'sequence_shape': (seq_len, image_size, image_size, 1),
        'data_type': 'video_sequence',
        'physics': 'bouncing_ball_dynamics',
        'format': 'raw_video_frames',
        'suitable_for': ['encoder_decoder', 'video_prediction', 'latent_space_modeling'],
        'memory_per_sequence_mb': seq_len * image_size * image_size * 4 / (1024*1024),
        'stochastic_properties': {
            'stationary': False,
            'bounded': True,  # Bounded by canvas size
            'deterministic_physics': True,  # Bouncing is deterministic
            'high_dimensional': True,  # 4096 dimensions per frame
            'temporal_correlation': True  # Strong frame-to-frame correlation
        }
    }


# Dataset configurations for integration with MultiDatasetManager
MOVING_MNIST_VIDEO_CONFIGS = {
    'moving_mnist_video': {
        'name': 'Moving MNIST (Video)',
        'description': 'Moving MNIST raw video sequences for encoder/decoder architectures',
        'generator': lambda **kwargs: generate_moving_mnist_video_dataset(**kwargs),
        'default_params': {
            'num_samples': 1000,
            'seq_len': 20,
            'num_digits': 2,
            'image_size': 64,
            'deterministic': True
        }
    },
    'moving_mnist_video_single': {
        'name': 'Moving MNIST Video (Single Digit)',
        'description': 'Moving MNIST video with single digit for simpler encoder/decoder training',
        'generator': lambda **kwargs: generate_moving_mnist_video_dataset(num_digits=1, **kwargs),
        'default_params': {
            'num_samples': 1000,
            'seq_len': 25,
            'num_digits': 1,
            'image_size': 64,
            'deterministic': True
        }
    },
    'moving_mnist_video_long': {
        'name': 'Moving MNIST Video (Long Sequence)',
        'description': 'Moving MNIST video with longer sequences for temporal modeling',
        'generator': lambda **kwargs: generate_moving_mnist_video_dataset(seq_len=50, **kwargs),
        'default_params': {
            'num_samples': 500,  # Fewer samples due to longer sequences
            'seq_len': 50,
            'num_digits': 2,
            'image_size': 64,
            'deterministic': True
        }
    }
}


def test_moving_mnist_video():
    """Test Moving MNIST video dataset generation."""
    print("🧪 Testing Moving MNIST Video Dataset")
    print("=" * 60)
    
    # Test dataset creation
    dataset = generate_moving_mnist_video_dataset(
        num_samples=10,  # Small test
        seq_len=15,
        num_digits=2,
        data_root="../../data"
    )
    
    # Get sample
    sample, label = dataset[0]
    
    print(f"✅ Video dataset successful:")
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Sample shape: {sample.shape}")
    print(f"   Data type: {sample.dtype}")
    print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"   Label: {label}")
    
    # Get dataset info
    info = get_moving_mnist_video_info(num_digits=2, seq_len=15, image_size=64)
    print(f"   Description: {info['description']}")
    print(f"   Sequence shape: {info['sequence_shape']}")
    print(f"   Memory per sequence: {info['memory_per_sequence_mb']:.3f} MB")
    print(f"   Suitable for: {', '.join(info['suitable_for'])}")
    
    # Test minibatch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    minibatch = next(iter(dataloader))
    batch_data, batch_labels = minibatch
    
    print(f"\\n📦 Minibatch test:")
    print(f"   Minibatch shape: {batch_data.shape}")
    print(f"   Format: (batch_size, seq_len, height, width, channels)")
    print(f"   Memory: {batch_data.numel() * 4 / (1024*1024):.3f} MB")
    
    print(f"\\n🎉 Moving MNIST video dataset testing completed!")


if __name__ == "__main__":
    test_moving_mnist_video()

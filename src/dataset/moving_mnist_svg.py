"""
Moving MNIST Dataset for SVG (Stochastic Video Generation)

This module provides Moving MNIST dataset integration specifically for the SVG
baseline model, following the data format and conventions from the original
SVG implementation.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.moving_mnist import MovingMNIST


class MovingMNISTSVG(torchdata.Dataset):
    """
    Moving MNIST dataset adapted for SVG training.
    
    This wrapper around the original MovingMNIST ensures compatibility with
    the SVG training pipeline and data format expectations.
    """
    
    def __init__(self, train: bool = True, data_root: str = "./data",
                 seq_len: int = 20, image_size: int = 64,
                 deterministic: bool = False, num_digits: int = 2,
                 num_samples: Optional[int] = None):
        """
        Initialize Moving MNIST dataset for SVG.
        
        Args:
            train: Whether to use training or test split
            data_root: Root directory for MNIST data
            seq_len: Length of video sequences
            image_size: Size of the canvas (image_size x image_size)
            deterministic: Whether to use deterministic movement
            num_digits: Number of digits per sequence
            num_samples: Number of samples to generate (None = use full dataset)
        """
        self.train = train
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.deterministic = deterministic
        self.num_digits = num_digits
        
        # Create underlying MovingMNIST dataset
        self.mnist_dataset = MovingMNIST(
            train=train,
            data_root=data_root,
            seq_len=seq_len,
            image_size=image_size,
            deterministic=deterministic,
            num_digits=num_digits
        )
        
        # Set dataset size
        if num_samples is not None:
            self.num_samples = min(num_samples, len(self.mnist_dataset))
        else:
            self.num_samples = len(self.mnist_dataset)
        
        print(f"🎬 MovingMNISTSVG initialized:")
        print(f"   Split: {'train' if train else 'test'}")
        print(f"   Samples: {self.num_samples}")
        print(f"   Sequence length: {seq_len}")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Digits: {num_digits}")
        print(f"   Deterministic: {deterministic}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a video sequence in SVG-compatible format.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Video sequence tensor of shape (seq_len, height, width, channels)
        """
        # Get raw video sequence from MovingMNIST
        sequence = self.mnist_dataset[idx]  # Shape: (seq_len, height, width, channels)
        
        # Convert to torch tensor (SVG expects float32)
        video_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        return video_tensor


def create_svg_datasets(data_root: str = "./data", 
                       train_seq_len: int = 20,
                       test_seq_len: int = 30,
                       image_size: int = 64,
                       num_digits: int = 2,
                       deterministic: bool = False,
                       train_samples: Optional[int] = None,
                       test_samples: Optional[int] = None) -> Tuple[MovingMNISTSVG, MovingMNISTSVG]:
    """
    Create train and test datasets for SVG training.
    
    Args:
        data_root: Root directory for MNIST data
        train_seq_len: Sequence length for training (n_past + n_future)
        test_seq_len: Sequence length for testing/evaluation
        image_size: Canvas size
        num_digits: Number of digits per sequence
        deterministic: Whether to use deterministic movement
        train_samples: Number of training samples (None = use all)
        test_samples: Number of test samples (None = use all)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    print(f"🎬 Creating SVG Moving MNIST Datasets")
    print(f"   Data root: {data_root}")
    print(f"   Train seq length: {train_seq_len}")
    print(f"   Test seq length: {test_seq_len}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Digits: {num_digits}")
    print(f"   Deterministic: {deterministic}")
    
    train_dataset = MovingMNISTSVG(
        train=True,
        data_root=data_root,
        seq_len=train_seq_len,
        image_size=image_size,
        deterministic=deterministic,
        num_digits=num_digits,
        num_samples=train_samples
    )
    
    test_dataset = MovingMNISTSVG(
        train=False,
        data_root=data_root,
        seq_len=test_seq_len,
        image_size=image_size,
        deterministic=deterministic,
        num_digits=num_digits,
        num_samples=test_samples
    )
    
    return train_dataset, test_dataset


def test_svg_dataset_compatibility():
    """Test that the SVG dataset format works correctly."""
    print("🧪 Testing SVG Dataset Compatibility")
    print("=" * 60)
    
    # Create test dataset
    dataset = MovingMNISTSVG(
        train=True,
        data_root="../data",
        seq_len=15,
        image_size=64,
        num_digits=2,
        deterministic=False,
        num_samples=10
    )
    
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Test individual sample
    sample = dataset[0]
    print(f"✅ Sample shape: {sample.shape}")
    print(f"   Format: (seq_len, height, width, channels)")
    print(f"   Data type: {sample.dtype}")
    print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
    
    # Test with DataLoader (like SVG does)
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    batch = next(iter(dataloader))
    
    print(f"✅ DataLoader batch shape: {batch.shape}")
    print(f"   Format: (batch_size, seq_len, height, width, channels)")
    
    # Test SVG normalization
    print(f"🔄 Testing SVG normalize_data process...")
    
    # Simulate the normalize_data function from svg_utils
    sequence = batch.clone()
    print(f"   Original: {sequence.shape}")
    
    # Apply SVG transformations
    sequence.transpose_(0, 1)  # (seq_len, batch_size, height, width, channels)
    sequence.transpose_(3, 4).transpose_(2, 3)  # (seq_len, batch_size, channels, height, width)
    
    print(f"   After SVG normalize: {sequence.shape}")
    print(f"   Expected format: (seq_len, batch_size, channels, height, width)")
    
    # Convert to Variable list (like SVG does)
    from torch.autograd import Variable
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sequence_list = [Variable(x.to(device)) for x in sequence]
    
    print(f"✅ SVG sequence list created: {len(sequence_list)} frames")
    print(f"   Each frame shape: {sequence_list[0].shape}")
    print(f"   Expected: (batch_size, channels, height, width)")
    
    print(f"🎉 SVG dataset compatibility test passed!")
    
    return dataset


if __name__ == "__main__":
    test_svg_dataset_compatibility()

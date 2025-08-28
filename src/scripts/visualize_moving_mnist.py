#!/usr/bin/env python3
"""
Moving MNIST Dataset Visualization Script

This script loads and visualizes the Moving MNIST dataset, showing:
1. Individual video sequences as animations
2. Static frame grids for multiple sequences
3. Dataset statistics and properties
4. Export options for videos and images

The Moving MNIST dataset consists of sequences of bouncing MNIST digits
that move around a canvas with realistic physics (bouncing off walls).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from dataset.moving_mnist import MovingMNIST
    print("✅ Successfully imported MovingMNIST")
except ImportError as e:
    print(f"❌ Failed to import MovingMNIST: {e}")
    sys.exit(1)


class MovingMNISTVisualizer:
    """Visualizer for Moving MNIST dataset with various display options."""
    
    def __init__(self, data_root: str = './data', seq_len: int = 20, 
                 num_digits: int = 2, image_size: int = 64, 
                 deterministic: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            data_root: Root directory for MNIST data
            seq_len: Length of video sequences
            num_digits: Number of digits per sequence
            image_size: Size of the canvas (image_size x image_size)
            deterministic: Whether digit movement is deterministic
        """
        self.data_root = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.deterministic = deterministic
        
        print(f"🎬 Initializing Moving MNIST Visualizer")
        print(f"   📁 Data root: {data_root}")
        print(f"   📺 Sequence length: {seq_len}")
        print(f"   🔢 Number of digits: {num_digits}")
        print(f"   📏 Image size: {image_size}x{image_size}")
        print(f"   🎯 Deterministic: {deterministic}")
        
        # Create datasets
        try:
            self.train_dataset = MovingMNIST(
                train=True,
                data_root=data_root,
                seq_len=seq_len,
                num_digits=num_digits,
                image_size=image_size,
                deterministic=deterministic
            )
            print(f"   ✅ Train dataset: {len(self.train_dataset)} samples")
            
            self.test_dataset = MovingMNIST(
                train=False,
                data_root=data_root,
                seq_len=seq_len,
                num_digits=num_digits,
                image_size=image_size,
                deterministic=deterministic
            )
            print(f"   ✅ Test dataset: {len(self.test_dataset)} samples")
            
        except Exception as e:
            print(f"   ❌ Failed to create datasets: {e}")
            raise
    
    def get_sample(self, dataset_type: str = 'train', index: int = 0) -> np.ndarray:
        """
        Get a sample from the dataset.
        
        Args:
            dataset_type: 'train' or 'test'
            index: Sample index
            
        Returns:
            Video sequence of shape (seq_len, height, width, channels)
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        return dataset[index]
    
    def visualize_sequence(self, sequence: np.ndarray, title: str = "Moving MNIST Sequence",
                          save_path: Optional[str] = None, show_animation: bool = True) -> None:
        """
        Visualize a single video sequence as an animation.
        
        Args:
            sequence: Video sequence of shape (seq_len, height, width, channels)
            title: Title for the plot
            save_path: Optional path to save the animation
            show_animation: Whether to show the animation interactively
        """
        seq_len, height, width, channels = sequence.shape
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{title}\n{seq_len} frames, {height}x{width} pixels, {self.num_digits} digits")
        ax.axis('off')
        
        # Initialize image display
        if channels == 1:
            im = ax.imshow(sequence[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        else:
            im = ax.imshow(sequence[0])
        
        # Add frame counter
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            """Animation function."""
            if channels == 1:
                im.set_array(sequence[frame, :, :, 0])
            else:
                im.set_array(sequence[frame])
            frame_text.set_text(f'Frame {frame + 1}/{seq_len}')
            return [im, frame_text]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=seq_len, interval=200, blit=True, repeat=True
        )
        
        # Save if requested
        if save_path:
            print(f"💾 Saving animation to {save_path}")
            anim.save(save_path, writer='pillow', fps=5)
            print(f"   ✅ Animation saved")
        
        if show_animation:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()
        
        return anim
    
    def visualize_frame_grid(self, sequence: np.ndarray, title: str = "Moving MNIST Frames",
                           save_path: Optional[str] = None, max_frames: int = 16) -> None:
        """
        Visualize frames from a sequence in a grid layout.
        
        Args:
            sequence: Video sequence of shape (seq_len, height, width, channels)
            title: Title for the plot
            save_path: Optional path to save the image
            max_frames: Maximum number of frames to show
        """
        seq_len, height, width, channels = sequence.shape
        
        # Determine grid size
        n_frames = min(seq_len, max_frames)
        grid_size = int(np.ceil(np.sqrt(n_frames)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(f"{title}\nShowing {n_frames}/{seq_len} frames", fontsize=16)
        
        # Flatten axes for easier indexing
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot frames
        frame_indices = np.linspace(0, seq_len-1, n_frames, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            ax = axes[i]
            
            if channels == 1:
                ax.imshow(sequence[frame_idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(sequence[frame_idx])
            
            ax.set_title(f'Frame {frame_idx + 1}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_frames, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            print(f"💾 Saving frame grid to {save_path}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Frame grid saved")
        
        plt.show()
    
    def visualize_multiple_sequences(self, dataset_type: str = 'train', 
                                   num_sequences: int = 4,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize multiple sequences side by side.
        
        Args:
            dataset_type: 'train' or 'test'
            num_sequences: Number of sequences to show
            save_path: Optional path to save the image
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        
        # Get sequences
        sequences = []
        indices = np.random.choice(len(dataset), num_sequences, replace=False)
        
        for idx in indices:
            seq = dataset[idx]
            sequences.append(seq)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * num_sequences))
        gs = GridSpec(num_sequences, self.seq_len, figure=fig)
        
        fig.suptitle(f"Multiple Moving MNIST Sequences ({dataset_type.title()} Set)", 
                     fontsize=16, y=0.98)
        
        for seq_idx, sequence in enumerate(sequences):
            # Show every few frames to fit in the grid
            frame_step = max(1, self.seq_len // 8)  # Show ~8 frames per sequence
            frame_indices = range(0, self.seq_len, frame_step)
            
            for col_idx, frame_idx in enumerate(frame_indices):
                if col_idx >= self.seq_len:  # Safety check
                    break
                    
                ax = fig.add_subplot(gs[seq_idx, col_idx])
                
                if sequence.shape[-1] == 1:
                    ax.imshow(sequence[frame_idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(sequence[frame_idx])
                
                if seq_idx == 0:  # Only label top row
                    ax.set_title(f'Frame {frame_idx + 1}', fontsize=8)
                
                if col_idx == 0:  # Only label first column
                    ax.set_ylabel(f'Seq {indices[seq_idx]}', fontsize=10)
                
                ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            print(f"💾 Saving multiple sequences to {save_path}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Multiple sequences saved")
        
        plt.show()
    
    def analyze_dataset(self) -> None:
        """Analyze and print dataset statistics."""
        print("\n📊 DATASET ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"📈 Dataset sizes:")
        print(f"   Train: {len(self.train_dataset):,} samples")
        print(f"   Test:  {len(self.test_dataset):,} samples")
        print(f"   Total: {len(self.train_dataset) + len(self.test_dataset):,} samples")
        
        # Sequence properties
        sample = self.train_dataset[0]
        print(f"\n🎬 Sequence properties:")
        print(f"   Shape: {sample.shape} (seq_len, height, width, channels)")
        print(f"   Data type: {sample.dtype}")
        print(f"   Value range: [{sample.min():.3f}, {sample.max():.3f}]")
        print(f"   Memory per sequence: {sample.nbytes / 1024:.1f} KB")
        
        # Movement analysis
        print(f"\n🏃 Movement properties:")
        print(f"   Step length: {self.train_dataset.step_length}")
        print(f"   Digit size: {self.train_dataset.digit_size}x{self.train_dataset.digit_size}")
        print(f"   Canvas size: {self.image_size}x{self.image_size}")
        print(f"   Deterministic movement: {self.deterministic}")
        
        # Memory usage estimation
        total_memory_mb = (len(self.train_dataset) + len(self.test_dataset)) * sample.nbytes / (1024**2)
        print(f"\n💾 Memory usage:")
        print(f"   Total dataset size: {total_memory_mb:.1f} MB")
        print(f"   Estimated RAM for full load: {total_memory_mb:.1f} MB")
    
    def export_sample_data(self, output_dir: str = "./moving_mnist_samples", 
                          num_samples: int = 5) -> None:
        """
        Export sample sequences as images and animations.
        
        Args:
            output_dir: Directory to save samples
            num_samples: Number of samples to export
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📤 EXPORTING SAMPLES")
        print(f"Output directory: {output_path.absolute()}")
        
        for i in range(num_samples):
            print(f"Exporting sample {i+1}/{num_samples}...")
            
            # Get sequence
            sequence = self.train_dataset[i]
            
            # Save frame grid
            frame_grid_path = output_path / f"sample_{i+1}_frames.png"
            self.visualize_frame_grid(
                sequence, 
                title=f"Moving MNIST Sample {i+1}",
                save_path=str(frame_grid_path),
                max_frames=16
            )
            plt.close()
            
            # Save animation
            anim_path = output_path / f"sample_{i+1}_animation.gif"
            self.visualize_sequence(
                sequence,
                title=f"Moving MNIST Sample {i+1}",
                save_path=str(anim_path),
                show_animation=False
            )
            plt.close()
        
        print(f"✅ Exported {num_samples} samples to {output_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize Moving MNIST Dataset")
    parser.add_argument("--data_root", type=str, default="./data", 
                       help="Root directory for MNIST data")
    parser.add_argument("--seq_len", type=int, default=20, 
                       help="Length of video sequences")
    parser.add_argument("--num_digits", type=int, default=2, 
                       help="Number of digits per sequence")
    parser.add_argument("--image_size", type=int, default=64, 
                       help="Size of the canvas")
    parser.add_argument("--deterministic", action="store_true", 
                       help="Use deterministic digit movement")
    parser.add_argument("--sample_index", type=int, default=0, 
                       help="Index of sample to visualize")
    parser.add_argument("--export", action="store_true", 
                       help="Export sample data")
    parser.add_argument("--export_dir", type=str, default="./moving_mnist_samples",
                       help="Directory for exported samples")
    parser.add_argument("--num_export", type=int, default=5,
                       help="Number of samples to export")
    
    args = parser.parse_args()
    
    print("🎬 MOVING MNIST DATASET VISUALIZER")
    print("=" * 60)
    
    try:
        # Create visualizer
        visualizer = MovingMNISTVisualizer(
            data_root=args.data_root,
            seq_len=args.seq_len,
            num_digits=args.num_digits,
            image_size=args.image_size,
            deterministic=args.deterministic
        )
        
        # Analyze dataset
        visualizer.analyze_dataset()
        
        # Get a sample sequence
        print(f"\n🎯 Loading sample {args.sample_index}...")
        sequence = visualizer.get_sample('train', args.sample_index)
        print(f"✅ Sample loaded: shape {sequence.shape}")
        
        # Visualize single sequence animation
        print("\n🎬 Showing sequence animation...")
        visualizer.visualize_sequence(
            sequence, 
            title=f"Moving MNIST Sample {args.sample_index}"
        )
        
        # Visualize frame grid
        print("\n🖼️  Showing frame grid...")
        visualizer.visualize_frame_grid(
            sequence,
            title=f"Moving MNIST Sample {args.sample_index} - Frame Grid"
        )
        
        # Visualize multiple sequences
        print("\n📺 Showing multiple sequences...")
        visualizer.visualize_multiple_sequences('train', num_sequences=4)
        
        # Export if requested
        if args.export:
            print("\n📤 Exporting samples...")
            visualizer.export_sample_data(args.export_dir, args.num_export)
        
        print("\n✅ Visualization complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

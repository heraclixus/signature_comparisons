#!/usr/bin/env python3
"""
Moving MNIST + Signature Analysis Integration Demo

This script demonstrates how to use the Moving MNIST dataset
with the signature analysis methods available in this project.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.moving_mnist import MovingMNIST


def extract_digit_trajectories(sequence: np.ndarray, num_digits: int = 2) -> np.ndarray:
    """
    Extract center-of-mass trajectories for each digit in the sequence.
    
    Args:
        sequence: Video sequence of shape (seq_len, height, width, channels)
        num_digits: Expected number of digits
        
    Returns:
        Trajectories of shape (seq_len, num_digits, 2) for (x, y) coordinates
    """
    seq_len, height, width, channels = sequence.shape
    
    # Simple approach: use connected components to separate digits
    trajectories = []
    
    for t in range(seq_len):
        frame = sequence[t, :, :, 0]  # Get grayscale frame
        
        # Find center of mass (simple approach - could be improved with clustering)
        if frame.sum() > 0:
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            total_mass = frame.sum()
            center_y = (y_coords * frame).sum() / total_mass
            center_x = (x_coords * frame).sum() / total_mass
            
            # For simplicity, just track the overall center of mass
            # In practice, you'd want to separate individual digits
            trajectories.append([center_x, center_y])
        else:
            trajectories.append([0.0, 0.0])  # Fallback
    
    return np.array(trajectories)


def extract_multiple_trajectories(sequence: np.ndarray, num_digits: int = 2) -> np.ndarray:
    """
    Extract trajectories for multiple digits using a more sophisticated approach.
    
    This is a simplified version - in practice you'd use proper clustering
    or connected component analysis.
    """
    seq_len, height, width, channels = sequence.shape
    trajectories = np.zeros((seq_len, num_digits, 2))
    
    for t in range(seq_len):
        frame = sequence[t, :, :, 0]
        
        # Find peaks (digit centers) using a simple approach
        # Threshold the image
        binary = frame > 0.1
        
        if binary.sum() > 0:
            # Find connected components (simplified)
            from scipy import ndimage
            labeled, num_features = ndimage.label(binary)
            
            centers = []
            for i in range(1, min(num_features + 1, num_digits + 1)):
                mask = labeled == i
                if mask.sum() > 0:
                    y_coords, x_coords = np.mgrid[0:height, 0:width]
                    center_y = (y_coords * mask).sum() / mask.sum()
                    center_x = (x_coords * mask).sum() / mask.sum()
                    centers.append([center_x, center_y])
            
            # Fill in trajectories
            for i, center in enumerate(centers[:num_digits]):
                trajectories[t, i] = center
    
    return trajectories


def compute_path_signatures(trajectories: np.ndarray, signature_level: int = 3) -> np.ndarray:
    """
    Compute path signatures for the trajectories.
    
    Args:
        trajectories: Shape (seq_len, num_paths, 2) or (seq_len, 2)
        signature_level: Signature truncation level
        
    Returns:
        Signature features
    """
    try:
        # Try to use the signature computation from the project
        import iisignature
        
        if trajectories.ndim == 2:
            # Single path
            sig = iisignature.sig(trajectories, signature_level)
        else:
            # Multiple paths
            signatures = []
            for i in range(trajectories.shape[1]):
                path = trajectories[:, i, :]
                sig = iisignature.sig(path, signature_level)
                signatures.append(sig)
            sig = np.stack(signatures)
        
        return sig
        
    except ImportError:
        print("⚠️  iisignature not available, using simple features instead")
        
        # Fallback: compute simple geometric features
        if trajectories.ndim == 2:
            # Simple features: displacement, velocity, acceleration
            displacement = trajectories[-1] - trajectories[0]
            velocity = np.diff(trajectories, axis=0).mean(axis=0)
            acceleration = np.diff(trajectories, n=2, axis=0).mean(axis=0)
            return np.concatenate([displacement, velocity, acceleration])
        else:
            # Multiple paths
            features = []
            for i in range(trajectories.shape[1]):
                path = trajectories[:, i, :]
                displacement = path[-1] - path[0]
                velocity = np.diff(path, axis=0).mean(axis=0)
                acceleration = np.diff(path, n=2, axis=0).mean(axis=0)
                feat = np.concatenate([displacement, velocity, acceleration])
                features.append(feat)
            return np.stack(features)


def visualize_trajectories(sequence: np.ndarray, trajectories: np.ndarray, 
                         save_path: str, title: str = "Moving MNIST Trajectories"):
    """Visualize the extracted trajectories overlaid on the video frames."""
    seq_len = sequence.shape[0]
    
    # Create a figure with multiple frames
    n_frames_to_show = min(8, seq_len)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    frame_indices = np.linspace(0, seq_len-1, n_frames_to_show, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        
        # Show the frame
        frame = sequence[frame_idx, :, :, 0]
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
        
        # Overlay trajectory up to this frame
        if trajectories.ndim == 2:
            # Single trajectory
            traj_so_far = trajectories[:frame_idx+1]
            ax.plot(traj_so_far[:, 0], traj_so_far[:, 1], 'r-', linewidth=2, alpha=0.7)
            ax.plot(trajectories[frame_idx, 0], trajectories[frame_idx, 1], 'ro', markersize=8)
        else:
            # Multiple trajectories
            colors = ['red', 'blue', 'green']
            for j in range(trajectories.shape[1]):
                traj_so_far = trajectories[:frame_idx+1, j]
                ax.plot(traj_so_far[:, 0], traj_so_far[:, 1], 
                       color=colors[j % len(colors)], linewidth=2, alpha=0.7)
                ax.plot(trajectories[frame_idx, j, 0], trajectories[frame_idx, j, 1], 
                       'o', color=colors[j % len(colors)], markersize=8)
        
        ax.set_title(f'Frame {frame_idx + 1}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved trajectory visualization to {save_path}")


def demo_signature_analysis():
    """Run the Moving MNIST + Signature Analysis demo."""
    print("🎬 MOVING MNIST + SIGNATURE ANALYSIS DEMO")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("moving_mnist_signature_output")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Test configurations
    configs = [
        {'name': 'Single_Digit', 'num_digits': 1, 'seq_len': 20},
        {'name': 'Two_Digits', 'num_digits': 2, 'seq_len': 20},
    ]
    
    for config in configs:
        print(f"\n🔧 Testing configuration: {config['name']}")
        
        try:
            # Create dataset
            dataset = MovingMNIST(
                train=True,
                data_root='../../data',
                seq_len=config['seq_len'],
                num_digits=config['num_digits'],
                image_size=64,
                deterministic=True
            )
            
            print(f"   📊 Dataset: {len(dataset)} samples")
            
            # Get sample sequences
            n_samples = 3
            results = []
            
            for i in range(n_samples):
                sequence = dataset[i]
                print(f"   🎯 Processing sample {i+1}/{n_samples}...")
                
                # Extract trajectories
                if config['num_digits'] == 1:
                    trajectories = extract_digit_trajectories(sequence, 1)
                else:
                    # For multiple digits, try the more sophisticated approach
                    try:
                        trajectories = extract_multiple_trajectories(sequence, config['num_digits'])
                    except ImportError:
                        print("     ⚠️  scipy not available, using simple approach")
                        trajectories = extract_digit_trajectories(sequence, config['num_digits'])
                
                print(f"      Trajectories shape: {trajectories.shape}")
                
                # Compute signatures
                signatures = compute_path_signatures(trajectories, signature_level=3)
                print(f"      Signature features shape: {signatures.shape}")
                
                # Visualize
                config_name = config['name'].lower()
                viz_path = output_dir / f"{config_name}_sample_{i+1}_trajectories.png"
                visualize_trajectories(
                    sequence, trajectories, str(viz_path),
                    f"Moving MNIST {config['name']} - Sample {i+1}"
                )
                
                results.append({
                    'sequence': sequence,
                    'trajectories': trajectories,
                    'signatures': signatures
                })
            
            # Analyze signature features
            print(f"   📈 Signature Analysis:")
            all_signatures = [r['signatures'] for r in results]
            
            if all_signatures[0].ndim == 1:
                # Single path signatures
                sig_matrix = np.stack(all_signatures)
                print(f"      Signature matrix shape: {sig_matrix.shape}")
                print(f"      Feature statistics:")
                print(f"        Mean: {sig_matrix.mean(axis=0)[:5]}... (first 5)")
                print(f"        Std:  {sig_matrix.std(axis=0)[:5]}... (first 5)")
            else:
                # Multiple path signatures
                print(f"      Multiple path signatures - shape varies by sample")
                for j, sig in enumerate(all_signatures):
                    print(f"        Sample {j+1}: {sig.shape}")
            
            print(f"   ✅ Configuration {config['name']} completed")
            
        except Exception as e:
            print(f"   ❌ Error with configuration {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Demo completed! Check output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file_path in sorted(output_dir.glob("*.png")):
        print(f"   📄 {file_path.name}")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Use extracted trajectories with signature kernel methods")
    print(f"   • Train models on signature features for digit classification")
    print(f"   • Compare signature distances between different motion patterns")
    print(f"   • Integrate with existing D2/D3 distributional diffusion models")


if __name__ == "__main__":
    demo_signature_analysis()

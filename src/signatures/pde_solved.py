"""
PDE-Solved Signature Implementation using local sigkernel

This module implements PDE-solved signature transforms using the local sigkernel package
to avoid compatibility issues with the system-installed version.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
import sys
import os
from typing import Optional

# Add local sigkernel to path
sigkernel_path = os.path.join(os.path.dirname(__file__), 'sigkernel')
if sigkernel_path not in sys.path:
    sys.path.insert(0, sigkernel_path)

try:
    from sigkernel.sigkernel import SigKernel
    from sigkernel.static_kernels import RBFKernel, LinearKernel
    SIGKERNEL_AVAILABLE = True
    print("✅ Using local sigkernel for PDE-solved signatures")
except ImportError as e:
    SIGKERNEL_AVAILABLE = False
    warnings.warn(f"Local sigkernel not available: {e}. Using truncated signature fallback.")


class PDESolvedSignature(nn.Module):
    """
    PDE-Solved Signature Transform using local sigkernel.
    
    This implements advanced signature computation using the signature kernel
    method which solves PDEs to compute signatures efficiently.
    """
    
    def __init__(self, dyadic_order: int = 8, static_kernel_type: str = "RBF", 
                 sigma: float = 1.0, depth: int = 4):
        """
        Initialize PDE-solved signature transform.
        
        Args:
            dyadic_order: Dyadic order for PDE approximation
            static_kernel_type: Type of static kernel ('RBF' or 'Linear')
            sigma: RBF kernel bandwidth (if using RBF)
            depth: Fallback depth for truncated signatures
        """
        super().__init__()
        self.dyadic_order = dyadic_order
        self.static_kernel_type = static_kernel_type
        self.sigma = sigma
        self.depth = depth
        
        if SIGKERNEL_AVAILABLE:
            # Create static kernel
            if static_kernel_type.upper() == "RBF":
                self.static_kernel = RBFKernel(sigma=sigma)
            else:
                self.static_kernel = LinearKernel(scale=1.0)
            
            # Create signature kernel
            self.sig_kernel = SigKernel(
                static_kernel=self.static_kernel,
                dyadic_order=dyadic_order
            )
            
            print(f"✅ PDE-solved signature kernel created: {static_kernel_type} kernel, dyadic_order={dyadic_order}")
            
        else:
            # Fallback to truncated signatures
            from signatures.truncated import TruncatedSignature
            self.truncated_sig = TruncatedSignature(depth=depth)
            
            print(f"⚠️ Using truncated signature fallback: depth={depth}")
    
    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE-solved signatures of input paths.
        
        Args:
            paths: Input paths, shape (batch, channels, time)
            
        Returns:
            Signature features, shape (batch, signature_dim)
        """
        if SIGKERNEL_AVAILABLE:
            return self._compute_sigkernel_features(paths)
        else:
            return self.truncated_sig(paths)
    
    def _compute_sigkernel_features(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature features using sigkernel.
        
        This doesn't directly compute signatures, but computes signature kernel
        features that can be used for loss computation.
        """
        batch_size, channels, time_steps = paths.shape
        
        # Convert paths to sigkernel format: (batch, time, channels)
        paths_sigkernel = paths.transpose(1, 2)  # Shape: (batch, time, channels)
        
        # For feature extraction, we compute the signature kernel against a reference
        # This is a simplified approach - full implementation would need proper feature extraction
        
        # Use self-similarity as features (this is a simplification)
        try:
            # Compute signature kernel matrix against itself
            K = self.sig_kernel.compute_Gram(paths_sigkernel, paths_sigkernel, sym=True, max_batch=100)
            
            # Extract features from kernel matrix
            # Use diagonal and mean values as signature features
            diag_features = torch.diag(K)  # Self-similarities
            mean_features = K.mean(dim=1)  # Average similarities
            
            # Combine features
            features = torch.stack([diag_features, mean_features], dim=1)  # Shape: (batch, 2)
            
            return features
            
        except Exception as e:
            warnings.warn(f"Sigkernel computation failed: {e}. Using truncated fallback.")
            # Fallback to truncated signatures
            if not hasattr(self, 'truncated_sig'):
                from signatures.truncated import TruncatedSignature
                self.truncated_sig = TruncatedSignature(depth=self.depth)
            return self.truncated_sig(paths)


class SigKernelScoringLoss:
    """
    Signature Kernel Scoring Loss using local sigkernel.
    
    Implements the proper scoring rule using the signature kernel:
    S(P, Y) = E_P[k_sig(X, X)] - 2 * E_P[k_sig(X, Y)]
    """
    
    def __init__(self, dyadic_order: int = 8, static_kernel_type: str = "RBF", 
                 sigma: float = 1.0, max_batch: int = 128):
        """
        Initialize signature kernel scoring loss.
        
        Args:
            dyadic_order: Dyadic order for signature kernel
            static_kernel_type: Type of static kernel
            sigma: RBF kernel bandwidth
            max_batch: Maximum batch size for computation
        """
        self.dyadic_order = dyadic_order
        self.sigma = sigma
        self.max_batch = max_batch
        
        if SIGKERNEL_AVAILABLE:
            # Create static kernel
            if static_kernel_type.upper() == "RBF":
                self.static_kernel = RBFKernel(sigma=sigma)
            else:
                self.static_kernel = LinearKernel(scale=1.0)
            
            # Create signature kernel
            self.sig_kernel = SigKernel(
                static_kernel=self.static_kernel,
                dyadic_order=dyadic_order
            )
            
            self.use_sigkernel = True
            print(f"✅ Signature kernel scoring loss created: {static_kernel_type} kernel, sigma={sigma}, dyadic_order={dyadic_order}")
            
        else:
            self.use_sigkernel = False
            warnings.warn("Local sigkernel not available. Using simplified scoring implementation.")
    
    def __call__(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """
        Compute signature kernel scoring rule.
        
        Args:
            generated_paths: Generated paths, shape (batch, channels, time)
            real_paths: Real paths, shape (batch, channels, time)
            
        Returns:
            Scoring rule loss
        """
        if self.use_sigkernel:
            return self._compute_sigkernel_scoring(generated_paths, real_paths)
        else:
            return self._compute_simplified_scoring(generated_paths, real_paths)
    
    def _compute_sigkernel_scoring(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Compute scoring rule using signature kernel with aggressive memory optimization."""
        # Convert to sigkernel format: (batch, time, channels)
        gen_paths = generated_paths.transpose(1, 2).double()  # Ensure double precision
        real_paths = real_paths.transpose(1, 2).double()
        
        try:
            # AGGRESSIVE MEMORY OPTIMIZATION
            batch_size = gen_paths.shape[0]
            
            # Use very small chunks for memory efficiency
            chunk_size = min(4, self.max_batch, batch_size)  # Maximum 4 samples at a time
            
            if batch_size <= chunk_size:
                # Small batch - can process directly
                y = real_paths[:1]  # Shape: (1, time, channels)
                
                # Reduce sequence length for memory
                max_time = min(50, gen_paths.shape[1])  # Limit to 50 time steps
                gen_short = gen_paths[:, :max_time, :]
                y_short = y[:, :max_time, :]
                
                # Compute scoring rule with reduced data
                scoring_loss = self.sig_kernel.compute_scoring_rule(gen_short, y_short, max_batch=chunk_size)
                
                return scoring_loss.float()
            else:
                # Process in very small chunks
                total_loss = 0.0
                num_chunks = (batch_size + chunk_size - 1) // chunk_size
                
                # Use same reference for all chunks
                y = real_paths[:1]
                max_time = min(50, gen_paths.shape[1])
                y_short = y[:, :max_time, :]
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, batch_size)
                    
                    chunk_gen = gen_paths[start_idx:end_idx, :max_time, :]
                    
                    # Clear cache between chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    chunk_loss = self.sig_kernel.compute_scoring_rule(chunk_gen, y_short, max_batch=chunk_size)
                    total_loss += chunk_loss.item()
                
                return torch.tensor(total_loss / num_chunks, dtype=torch.float32)
            
        except Exception as e:
            warnings.warn(f"Sigkernel scoring computation failed: {e}. Using simplified fallback.")
            return self._compute_simplified_scoring(generated_paths, real_paths)
    
    def _compute_simplified_scoring(self, generated_paths: torch.Tensor, real_paths: torch.Tensor) -> torch.Tensor:
        """Simplified scoring rule computation."""
        # This is the same as our existing SimplifiedScoringLoss
        # Use truncated signatures for feature computation
        from signatures.truncated import TruncatedSignature
        
        sig_transform = TruncatedSignature(depth=4)
        
        gen_sigs = sig_transform(generated_paths)
        real_sigs = sig_transform(real_paths)
        
        # RBF kernel similarities
        similarities = self._rbf_kernel(gen_sigs, real_sigs)
        
        # Scoring rule
        score = -torch.log(similarities.mean() + 1e-8)
        
        return score
    
    def _rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel similarities."""
        X_norm = (X**2).sum(dim=1, keepdim=True)
        Y_norm = (Y**2).sum(dim=1, keepdim=True)
        
        distances = X_norm - 2 * torch.mm(X, Y.t()) + Y_norm.t()
        
        return torch.exp(-distances / (2 * self.sigma**2))


def test_local_sigkernel():
    """Test the local sigkernel implementation."""
    print("Testing Local Sigkernel Implementation")
    print("=" * 40)
    
    if not SIGKERNEL_AVAILABLE:
        print("❌ Local sigkernel not available")
        return False
    
    # Create test data
    batch_size = 4
    time_steps = 50
    channels = 2
    
    paths = torch.randn(batch_size, time_steps, channels, dtype=torch.float64)
    print(f"Test paths shape: {paths.shape}")
    print(f"Test paths dtype: {paths.dtype}")
    
    try:
        # Test static kernel
        static_kernel = RBFKernel(sigma=1.0)
        print(f"✅ Created RBFKernel")
        
        # Test signature kernel
        sig_kernel = SigKernel(static_kernel=static_kernel, dyadic_order=6)
        print(f"✅ Created SigKernel")
        
        # Test kernel computation
        K = sig_kernel.compute_kernel(paths, paths, max_batch=100)
        print(f"✅ Computed signature kernel: shape {K.shape}")
        
        # Test Gram matrix
        G = sig_kernel.compute_Gram(paths, paths, sym=True, max_batch=100)
        print(f"✅ Computed Gram matrix: shape {G.shape}")
        
        # Test scoring rule
        y = paths[:1]  # Single reference path
        scoring = sig_kernel.compute_scoring_rule(paths, y, max_batch=100)
        print(f"✅ Computed scoring rule: {scoring.item():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Local sigkernel test failed: {e}")
        return False


if __name__ == "__main__":
    test_local_sigkernel()

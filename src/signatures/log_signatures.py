"""
Log Signature Implementation

This module implements log signature transforms for path data.
Log signatures provide a more efficient representation than truncated signatures
by using the logarithm of the signature, which often has better convergence properties.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Optional, Union, Tuple

try:
    import iisignature
    IISIGNATURE_AVAILABLE = True
except ImportError:
    IISIGNATURE_AVAILABLE = False
    warnings.warn("iisignature not available. Log signature functionality will be limited.")


class LogSignature(nn.Module):
    """
    Log Signature Transform for path data.
    
    Log signatures provide a more efficient representation than truncated signatures
    by computing the logarithm of the signature, which often has better numerical
    properties and convergence.
    """
    
    def __init__(self, depth: int = 4, stream: bool = True, scalar_term: bool = True):
        """
        Initialize log signature transform.
        
        Args:
            depth: Maximum signature depth to compute
            stream: Whether to use streaming computation for efficiency
            scalar_term: Whether to include the scalar term (level 0)
        """
        super().__init__()
        self.depth = depth
        self.stream = stream
        self.scalar_term = scalar_term
        
        if not IISIGNATURE_AVAILABLE:
            warnings.warn("iisignature not available. Using simplified log signature implementation.")
    
    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Compute log signatures of input paths.
        
        Args:
            paths: Input paths, shape (batch, channels, time)
            
        Returns:
            Log signature features, shape (batch, signature_dim)
        """
        if IISIGNATURE_AVAILABLE:
            return self._compute_iisignature_logsig(paths)
        else:
            return self._compute_simplified_logsig(paths)
    
    def _compute_iisignature_logsig(self, paths: torch.Tensor) -> torch.Tensor:
        """Compute log signatures using iisignature library."""
        batch_size, channels, time_steps = paths.shape
        device = paths.device
        
        # Convert to numpy for iisignature
        paths_np = paths.detach().cpu().numpy()
        
        log_signatures = []
        
        for i in range(batch_size):
            # Extract path for this batch item
            path = paths_np[i].T  # Shape: (time, channels)
            
            try:
                # Compute log signature
                logsig = iisignature.logsig(path, self.depth)
                log_signatures.append(logsig)
                
            except Exception as e:
                warnings.warn(f"iisignature logsig failed for batch {i}: {e}. Using fallback.")
                # Fallback to simplified computation
                path_tensor = torch.from_numpy(path).unsqueeze(0).to(device)
                path_tensor = path_tensor.transpose(1, 2)  # Shape: (1, channels, time)
                fallback_logsig = self._compute_simplified_logsig(path_tensor)
                log_signatures.append(fallback_logsig.squeeze(0).cpu().numpy())
        
        # Convert back to tensor
        log_signatures = np.array(log_signatures)
        return torch.from_numpy(log_signatures).float().to(device)
    
    def _compute_simplified_logsig(self, paths: torch.Tensor) -> torch.Tensor:
        """
        Simplified log signature computation when iisignature is not available.
        
        This uses a simplified approximation based on path increments and their
        higher-order products, then takes the logarithm for stability.
        """
        batch_size, channels, time_steps = paths.shape
        device = paths.device
        
        # Compute path increments
        increments = paths[:, :, 1:] - paths[:, :, :-1]  # Shape: (batch, channels, time-1)
        
        features = []
        
        # Level 0: Scalar term (if requested)
        if self.scalar_term:
            scalar_features = torch.ones(batch_size, 1, device=device)
            features.append(scalar_features)
        
        # Level 1: Linear terms (path increments)
        level1_features = increments.sum(dim=2)  # Shape: (batch, channels)
        features.append(level1_features)
        
        # Level 2: Quadratic terms
        if self.depth >= 2:
            level2_features = []
            for i in range(channels):
                for j in range(channels):
                    # Compute iterated integral approximation
                    inc_i = increments[:, i, :]  # Shape: (batch, time-1)
                    inc_j = increments[:, j, :]  # Shape: (batch, time-1)
                    
                    # Simple approximation of iterated integral
                    if i == j:
                        # Diagonal terms: cumulative sum of squares
                        quad_term = (inc_i ** 2).sum(dim=1)
                    else:
                        # Off-diagonal terms: approximation using cumulative products
                        quad_term = (inc_i * inc_j).sum(dim=1)
                    
                    level2_features.append(quad_term)
            
            level2_tensor = torch.stack(level2_features, dim=1)  # Shape: (batch, channels^2)
            features.append(level2_tensor)
        
        # Level 3 and higher: Higher-order approximations
        for level in range(3, self.depth + 1):
            # Simplified higher-order terms using powers of increments
            higher_order_features = []
            
            for i in range(channels):
                # Use powers of individual increments as approximation
                inc_i = increments[:, i, :]  # Shape: (batch, time-1)
                power_term = (inc_i ** level).sum(dim=1)
                higher_order_features.append(power_term)
                
                # Cross terms with other channels (simplified)
                for j in range(i + 1, channels):
                    inc_j = increments[:, j, :]
                    cross_term = ((inc_i * inc_j) ** (level // 2)).sum(dim=1)
                    higher_order_features.append(cross_term)
            
            if higher_order_features:
                higher_order_tensor = torch.stack(higher_order_features, dim=1)
                features.append(higher_order_tensor)
        
        # Concatenate all features
        signature_features = torch.cat(features, dim=1)  # Shape: (batch, total_features)
        
        # Apply log transformation for stability
        # Use log(1 + |x|) * sign(x) to handle negative values
        log_features = torch.sign(signature_features) * torch.log(1 + torch.abs(signature_features))
        
        return log_features
    
    def get_signature_dim(self, channels: int) -> int:
        """
        Calculate the dimension of log signature features.
        
        Args:
            channels: Number of input channels
            
        Returns:
            Dimension of log signature vector
        """
        if IISIGNATURE_AVAILABLE:
            try:
                # Use iisignature to get exact dimension
                return iisignature.logsiglength(channels, self.depth)
            except:
                pass
        
        # Fallback calculation for simplified implementation
        total_dim = 0
        
        # Level 0: Scalar term
        if self.scalar_term:
            total_dim += 1
        
        # Level 1: Linear terms
        total_dim += channels
        
        # Level 2 and higher: Approximate calculation
        for level in range(2, self.depth + 1):
            if level == 2:
                # Quadratic terms: channels^2
                total_dim += channels ** 2
            else:
                # Higher-order terms: simplified approximation
                # Each channel contributes 1 term, plus cross terms
                total_dim += channels + (channels * (channels - 1)) // 2
        
        return total_dim


def test_log_signature():
    """Test log signature computation."""
    print("Testing Log Signature Implementation")
    print("=" * 40)
    
    # Create test data
    batch_size = 4
    channels = 2
    time_steps = 50
    depth = 3
    
    # Create sample paths
    t = torch.linspace(0, 1, time_steps)
    paths = torch.zeros(batch_size, channels, time_steps)
    
    for i in range(batch_size):
        # Create different path types
        if i % 4 == 0:
            # Linear path
            paths[i, 0, :] = t
            paths[i, 1, :] = 0.5 * t
        elif i % 4 == 1:
            # Quadratic path
            paths[i, 0, :] = t
            paths[i, 1, :] = t ** 2
        elif i % 4 == 2:
            # Sinusoidal path
            paths[i, 0, :] = t
            paths[i, 1, :] = torch.sin(2 * np.pi * t)
        else:
            # Random walk
            paths[i, 0, :] = t
            paths[i, 1, :] = torch.cumsum(torch.randn(time_steps) * 0.1, dim=0)
    
    print(f"Test paths shape: {paths.shape}")
    
    # Test log signature computation
    logsig = LogSignature(depth=depth, stream=True, scalar_term=True)
    
    try:
        log_features = logsig(paths)
        print(f"✅ Log signature computation successful")
        print(f"   Input shape: {paths.shape}")
        print(f"   Output shape: {log_features.shape}")
        print(f"   Feature dimension: {log_features.shape[1]}")
        print(f"   Depth: {depth}")
        
        # Check for reasonable values
        print(f"   Feature range: [{log_features.min().item():.4f}, {log_features.max().item():.4f}]")
        print(f"   Feature mean: {log_features.mean().item():.4f}")
        print(f"   Feature std: {log_features.std().item():.4f}")
        
        # Test signature dimension calculation
        expected_dim = logsig.get_signature_dim(channels)
        print(f"   Expected dimension: {expected_dim}")
        print(f"   Actual dimension: {log_features.shape[1]}")
        
        if IISIGNATURE_AVAILABLE:
            print(f"   Using iisignature: ✅")
        else:
            print(f"   Using simplified implementation: ⚠️")
        
        return True
        
    except Exception as e:
        print(f"❌ Log signature computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_log_signature()

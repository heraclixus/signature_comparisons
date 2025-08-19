"""
Truncated Signature Implementation (Extracted from deep_signature_transform)

This implements truncated signature computation using the iisignature library,
originally implemented in deep_signature_transform/siglayer/backend.py
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
import warnings
from typing import Union

# Try to import iisignature (may not be available in all environments)
try:
    import iisignature
    IISIGNATURE_AVAILABLE = True
except ImportError:
    IISIGNATURE_AVAILABLE = False
    warnings.warn("iisignature not available. TruncatedSignature will not work.")


def sig_dim(alphabet_size: int, depth: int) -> int:
    """Calculate the number of terms in a signature of given depth."""
    if alphabet_size == 1:
        return depth
    return int(alphabet_size * (1 - alphabet_size ** depth) / (1 - alphabet_size))


class path_sig_fn(autograd.Function):
    """
    Autograd function for signature computation with automatic differentiation.
    
    Uses iisignature library for forward pass and sigbackprop for backward pass.
    """
    
    @staticmethod
    def forward(ctx, path, depth):
        if not IISIGNATURE_AVAILABLE:
            raise ImportError("iisignature not available")
        
        device = path.device
        # Transpose for iisignature convention (channels last)
        path_np = path.detach().cpu().numpy().transpose()
        ctx.path = path_np
        ctx.depth = depth
        
        signature = iisignature.sig(path_np, depth)
        return torch.tensor(signature, dtype=torch.float, device=device)
    
    @staticmethod
    def backward(ctx, grad_output):
        if not IISIGNATURE_AVAILABLE:
            raise ImportError("iisignature not available")
        
        device = grad_output.device
        backprop = iisignature.sigbackprop(
            grad_output.cpu().numpy(), ctx.path, ctx.depth
        )
        # Transpose back to PyTorch convention (channels first)
        out = torch.tensor(backprop, dtype=torch.float, device=device).t()
        
        # Clean up context
        del ctx.path
        del ctx.depth
        
        return out, None


def path_sig(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Compute signature of a single path."""
    return path_sig_fn.apply(path, depth)


class TruncatedSignature(nn.Module):
    """
    Truncated Signature Layer.
    
    Computes the signature of input paths up to a specified depth.
    The signature is a fundamental object in rough path theory that
    characterizes the path up to tree-like equivalence.
    """
    
    def __init__(self, depth: int):
        """
        Initialize TruncatedSignature layer.
        
        Args:
            depth: Maximum depth of signature computation
        """
        super().__init__()
        
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"Depth must be an integer >= 1, got {depth}")
        
        self.depth = depth
    
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute signatures for a batch of paths.
        
        Args:
            path: Input paths, shape (batch, channels, length)
                 Each batch element is a path in R^channels
                 
        Returns:
            Signatures, shape (batch, signature_dim)
            where signature_dim = sig_dim(channels, depth)
        """
        if not IISIGNATURE_AVAILABLE:
            raise ImportError("iisignature not available")
        
        if len(path.shape) != 3:
            raise ValueError(f"Expected 3D tensor (batch, channels, length), got {path.shape}")
        
        if path.size(1) == 1:
            warnings.warn(
                f"{self.__class__.__name__} called on path with only one channel; "
                f"signature is just path moments, no cross-term information."
            )
        
        batch_size = path.size(0)
        signatures = []
        
        for i in range(batch_size):
            sig = path_sig(path[i], self.depth)
            signatures.append(sig)
        
        return torch.stack(signatures)
    
    def extra_repr(self) -> str:
        return f'depth={self.depth}'
    
    def output_size(self, input_channels: int) -> int:
        """
        Calculate output signature dimension.
        
        Args:
            input_channels: Number of input channels
            
        Returns:
            Signature dimension
        """
        return sig_dim(input_channels, self.depth)


class BatchTruncatedSignature(TruncatedSignature):
    """
    Batched version of TruncatedSignature for efficiency.
    
    Processes all paths in a batch simultaneously when possible.
    """
    
    def forward(self, path: torch.Tensor) -> torch.Tensor:
        """
        Compute signatures for a batch of paths efficiently.
        
        Args:
            path: Input paths, shape (batch, channels, length)
                 
        Returns:
            Signatures, shape (batch, signature_dim)
        """
        if not IISIGNATURE_AVAILABLE:
            raise ImportError("iisignature not available")
        
        if len(path.shape) != 3:
            raise ValueError(f"Expected 3D tensor (batch, channels, length), got {path.shape}")
        
        # For now, fall back to sequential processing
        # Could be optimized with batch processing in iisignature
        return super().forward(path)


# Convenience function for backward compatibility
def batch_path_sig(path: torch.Tensor, depth: int) -> torch.Tensor:
    """
    Compute signatures for a batch of paths.
    
    Args:
        path: Batch of paths, shape (batch, channels, length)
        depth: Signature depth
        
    Returns:
        Batch of signatures
    """
    sig_layer = TruncatedSignature(depth)
    return sig_layer(path)

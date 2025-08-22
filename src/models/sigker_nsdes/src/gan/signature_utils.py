"""
Utility module for signature computation with fallback from signatory to iisignature.
"""

import torch
import torch.autograd as autograd
import numpy as np
import warnings

# Try to import signatory first, fallback to iisignature if not available
try:
    import signatory
    SIGNATORY_AVAILABLE = True
except ImportError:
    SIGNATORY_AVAILABLE = False
    try:
        import iisignature
        IISIGNATURE_AVAILABLE = True
    except ImportError:
        IISIGNATURE_AVAILABLE = False
        warnings.warn(
            "Neither signatory nor iisignature is available. "
            "Please install one of these packages for signature computations."
        )


class IISignatureFunction(autograd.Function):
    """Custom autograd function for iisignature signature computation."""
    
    @staticmethod
    def forward(ctx, path, depth):
        device = path.device
        # Convert to numpy and transpose for iisignature (channels last)
        path_np = path.detach().cpu().numpy().transpose(0, 2, 1)  # (batch, channels, length) -> (batch, length, channels)
        ctx.path = path_np
        ctx.depth = depth
        
        # Compute signatures for each path in the batch
        batch_size = path_np.shape[0]
        signatures = []
        for i in range(batch_size):
            sig = iisignature.sig(path_np[i], depth)
            signatures.append(sig)
        
        # Convert to numpy array first to avoid warning
        signatures_array = np.array(signatures)
        result = torch.tensor(signatures_array, dtype=torch.float, device=device)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        batch_size = len(ctx.path)
        
        # Compute backpropagation for each path in the batch
        backprops = []
        for i in range(batch_size):
            backprop = iisignature.sigbackprop(
                grad_output[i].cpu().numpy(), 
                ctx.path[i], 
                ctx.depth
            )
            backprops.append(backprop)
        
        # Stack and transpose back to PyTorch convention (channels first)
        backprops_array = np.array(backprops)
        result = torch.tensor(backprops_array, dtype=torch.float, device=device).transpose(1, 2)
        
        # Clean up context
        del ctx.path
        del ctx.depth
        return result, None


class IILogSignatureFunction(autograd.Function):
    """Custom autograd function for iisignature log signature computation."""
    
    @staticmethod
    def forward(ctx, path, depth):
        device = path.device
        # Convert to numpy and transpose for iisignature (channels last)
        path_np = path.detach().cpu().numpy().transpose(0, 2, 1)
        ctx.path = path_np
        ctx.depth = depth
        
        # Compute log signatures for each path in the batch
        batch_size = path_np.shape[0]
        log_signatures = []
        for i in range(batch_size):
            # iisignature logsig expects (length, channels) shape and depth as second argument
            logsig = iisignature.logsig(path_np[i], depth)
            log_signatures.append(logsig)
        
        # Convert to numpy array first to avoid warning
        log_signatures_array = np.array(log_signatures)
        result = torch.tensor(log_signatures_array, dtype=torch.float, device=device)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        batch_size = len(ctx.path)
        
        # Compute backpropagation for each path in the batch
        backprops = []
        for i in range(batch_size):
            backprop = iisignature.logsigbackprop(
                grad_output[i].cpu().numpy(),
                ctx.path[i], 
                ctx.depth
            )
            backprops.append(backprop)
        
        # Stack and transpose back to PyTorch convention (channels first)
        backprops_array = np.array(backprops)
        result = torch.tensor(backprops_array, dtype=torch.float, device=device).transpose(1, 2)
        
        # Clean up context
        del ctx.path
        del ctx.depth
        return result, None


def signature(path, depth, scalar_term=True):
    """
    Compute signature with fallback from signatory to iisignature.
    
    Args:
        path: Tensor of shape (batch, channels, length) or (batch, length, channels)
        depth: Signature depth
        scalar_term: Whether to include scalar term (only for signatory)
        
    Returns:
        Signature tensor
    """
    if SIGNATORY_AVAILABLE:
        return signatory.signature(path, depth, scalar_term=scalar_term)
    elif IISIGNATURE_AVAILABLE:
        if not scalar_term:
            warnings.warn("scalar_term=False not supported with iisignature fallback, ignoring parameter")
        return IISignatureFunction.apply(path, depth)
    else:
        raise RuntimeError("Neither signatory nor iisignature is available")


def logsignature(path, depth):
    """
    Compute log signature with fallback from signatory to iisignature.
    
    Args:
        path: Tensor of shape (batch, channels, length) or (batch, length, channels)
        depth: Signature depth
        
    Returns:
        Log signature tensor
        
    Note:
        The iisignature fallback for log signature may have limitations and
        might not work in all cases due to differences in the API.
    """
    if SIGNATORY_AVAILABLE:
        return signatory.logsignature(path, depth)
    elif IISIGNATURE_AVAILABLE:
        warnings.warn(
            "Using iisignature fallback for log signature computation. "
            "This may have limitations compared to signatory."
        )
        return IILogSignatureFunction.apply(path, depth)
    else:
        raise RuntimeError("Neither signatory nor iisignature is available")


def all_words(channels, depth):
    """
    Get all words for signature computation.
    
    Args:
        channels: Number of channels
        depth: Maximum word length
        
    Returns:
        List of words (tuples)
    """
    if SIGNATORY_AVAILABLE:
        return signatory.all_words(channels, depth)
    elif IISIGNATURE_AVAILABLE:
        # iisignature doesn't have a direct equivalent, so we generate manually
        words = []
        
        # Add empty word (scalar term)
        words.append(())
        
        # Generate all words up to depth
        for d in range(1, depth + 1):
            # Generate all possible words of length d
            def generate_words(current_word, remaining_depth):
                if remaining_depth == 0:
                    words.append(tuple(current_word))
                    return
                
                for channel in range(channels):
                    generate_words(current_word + [channel], remaining_depth - 1)
            
            generate_words([], d)
        
        return words
    else:
        raise RuntimeError("Neither signatory nor iisignature is available")


def lyndon_words(channels, depth):
    """
    Get Lyndon words for log signature computation.
    
    Args:
        channels: Number of channels  
        depth: Maximum word length
        
    Returns:
        List of Lyndon words (tuples)
    """
    if SIGNATORY_AVAILABLE:
        return signatory.lyndon_words(channels, depth)
    elif IISIGNATURE_AVAILABLE:
        # iisignature doesn't have Lyndon words, so we implement a basic version
        def is_lyndon_word(word):
            """Check if a word is a Lyndon word."""
            if not word:
                return True
            
            n = len(word)
            for i in range(1, n):
                if word[i:] + word[:i] < word:
                    return False
            return True
        
        # Generate all words and filter for Lyndon words
        all_word_list = all_words(channels, depth)
        lyndon_word_list = [word for word in all_word_list if is_lyndon_word(word)]
        
        return lyndon_word_list
    else:
        raise RuntimeError("Neither signatory nor iisignature is available")

"""
Latent SDE Models for Signature-Based Deep Learning

This module implements Latent Stochastic Differential Equation models where:
- Z_t: Latent state following an SDE
- X_t: Observable data decoded from Z_t

Architecture:
1. Encoder: X → q(Z_0|X) (variational posterior)
2. Latent SDE: dZ_t = f(t,Z_t)dt + g(t,Z_t)dW_t  
3. Decoder: Z_t → X_t (observation model)

Training via ELBO (Evidence Lower BOund):
ELBO = E_q[log p(X|Z)] - KL(q(Z_0|X) || p(Z_0))
"""

from .implementations.v1_latent_sde import create_v1_model, V1TorchSDEModel, TorchSDELatentSDE

__all__ = [
    'create_v1_model',
    'V1TorchSDEModel',
    'TorchSDELatentSDE'
]

"""
Hybrid Latent SDE + Signature Models

This module implements hybrid models that combine:
- Latent SDE architectures (V1, V2) for generative power
- Signature-based losses for distributional quality

Model Series:
- C1-C3: V1 Latent SDE + Signature losses (T-Stat, Scoring, MMD)
- C4-C6: V2 SDE Matching + Signature losses (T-Stat, Scoring, MMD)

The hybrid approach aims to leverage the best of both worlds:
- Latent SDEs: Powerful generative modeling in latent space
- Signature losses: Strong distributional constraints and path-wise quality
"""

from .c1_latent_sde_tstat import C1Model, create_c1_model

__all__ = [
    'C1Model',
    'create_c1_model'
]

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
from .c2_latent_sde_scoring import C2Model, create_c2_model
from .c3_latent_sde_mmd import C3Model, create_c3_model
from .c4_sde_matching_tstat import C4Model, create_c4_model
from .c5_sde_matching_scoring import C5Model, create_c5_model
from .c6_sde_matching_mmd import C6Model, create_c6_model

__all__ = [
    'C1Model',
    'create_c1_model',
    'C2Model', 
    'create_c2_model',
    'C3Model',
    'create_c3_model',
    'C4Model',
    'create_c4_model',
    'C5Model',
    'create_c5_model',
    'C6Model',
    'create_c6_model'
]

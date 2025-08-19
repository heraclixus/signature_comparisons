"""
Discriminator implementations for adversarial training.

This module provides discriminator components that can be used with existing
generators to enable adversarial training variants of our baseline models.
"""

from .signature_discriminators import (
    SignatureMMDDiscriminator,
    SignatureScoringDiscriminator, 
    AdversarialDiscriminatorBase
)

__all__ = [
    'SignatureMMDDiscriminator',
    'SignatureScoringDiscriminator', 
    'AdversarialDiscriminatorBase'
]

# Note: TStatisticDiscriminator removed due to compatibility issues with adversarial training
# T-statistic models (A1, B3) should use non-adversarial training only

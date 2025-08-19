"""
Model Implementations using the Factory Pattern

This module contains specific implementations of truly generative signature-based models
that inherit from BaseSignatureModel and register themselves with the factory.

Only includes models that pass generator diversity tests for stochastic process modeling.
"""

from .a1_final import A1FinalModel, create_a1_final_model
from .a2_canned_scoring import A2Model, create_a2_model
from .a3_canned_mmd import A3Model, create_a3_model
from .a4_canned_logsig import A4Model, create_a4_model
from .b1_nsde_scoring import B1Model, create_b1_model
from .b2_nsde_mmd_pde import B2Model, create_b2_model
from .b3_nsde_tstatistic import B3Model, create_b3_model
from .b4_nsde_mmd import B4Model, create_b4_model
from .b5_nsde_scoring import B5Model, create_b5_model

def get_all_model_creators():
    """Get dictionary of all model creation functions."""
    return {
        'A1': create_a1_final_model,
        'A2': create_a2_model,
        'A3': create_a3_model,
        'A4': create_a4_model,
        'B1': create_b1_model,
        'B2': create_b2_model,
        'B3': create_b3_model,
        'B4': create_b4_model,
        'B5': create_b5_model
    }

__all__ = [
    'A1FinalModel',
    'create_a1_final_model',
    'A2Model', 
    'create_a2_model',
    'A3Model',
    'create_a3_model',
    'A4Model',
    'create_a4_model',
    'B1Model',
    'create_b1_model',
    'B2Model',
    'create_b2_model',
    'B3Model',
    'create_b3_model',
    'B4Model',
    'create_b4_model',
    'B5Model',
    'create_b5_model',
    'get_all_model_creators'
]

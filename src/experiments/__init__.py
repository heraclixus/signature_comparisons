"""
Experimental framework for signature-based deep learning comparisons.

This module provides tools for running systematic experiments across
different combinations of generators, losses, and signature methods.
"""

from .benchmarks import *
from .analysis import *

__all__ = [
    'run_benchmark',
    'analyze_results',
    'compare_methods'
]

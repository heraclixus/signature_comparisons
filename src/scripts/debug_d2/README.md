# D2 Debug Suite

Comprehensive debugging and analysis tools for D2 (Distributional Diffusion) models on 1D time series datasets.

## Overview

This debug suite investigates the D2 model's performance, training behavior, and scaling issues on simple 1D datasets like OU process and Fractional Brownian Motion.

## Key Issues Being Investigated

1. **Massive Scaling Problem**: Generated trajectories are ~1600x larger than ground truth
2. **Distribution Learning**: Whether D2 models can learn proper statistical distributions
3. **Training Effectiveness**: Does training improve generation quality?
4. **Architecture Comparison**: MLP vs Transformer generators

## Files

- `d2_comprehensive_debug.py` - Main comprehensive debug suite (CUDA-compatible)
- `run_debug.py` - Interactive runner script
- `run_server.py` - Non-interactive server/cluster runner
- `README.md` - This documentation

## Usage

### Interactive Mode (Local)
```bash
python src/scripts/debug_d2/run_debug.py
```

### Server/Cluster Mode (CUDA Optimized)
```bash
# Default run
python src/scripts/debug_d2/run_server.py

# Custom configuration
python src/scripts/debug_d2/run_server.py --dataset fbm_h03 --epochs 200 --output-dir my_results

# With arguments
python src/scripts/debug_d2/run_server.py --help
```

### Direct Run
```bash
python src/scripts/debug_d2/d2_comprehensive_debug.py
```

## CUDA Acceleration

The debug suite automatically detects and uses CUDA when available:

- **GPU Detection**: Automatically uses CUDA if available
- **Memory Optimization**: GPU memory management and cleanup
- **Batch Size Scaling**: Larger batches on GPU for efficiency  
- **Progress Monitoring**: GPU memory usage tracking
- **Server Optimizations**: CUDNN benchmarking for consistent workloads

### GPU Requirements
- CUDA-compatible GPU with sufficient memory (>4GB recommended)
- PyTorch with CUDA support installed
- For large models: 8GB+ GPU memory recommended

## Output

All results are saved to `debug_d2_results/` directory:

- **Plots**:
  - `d2_main_comparison_*.png` - Main trajectory comparisons
  - `d2_scaling_analysis_*.png` - Scaling issue analysis
  
- **Reports**:
  - `d2_debug_report_*.txt` - Comprehensive text report

## Key Findings

### Scaling Issue
- Ground truth: std ~0.6 (OU process), ~0.8 (FBM)  
- Generated: std ~900-1100
- **Scale factor: ~1600x too large**

### Visualization Problem
- Ground truth appears as "straight line at zero" in plots
- This is due to y-axis being dominated by massive generated values
- Actual ground truth has proper stochastic variation

### Training Investigation
- Tests both untrained and trained D2 models
- Uses proper `fit()` method with progress bars
- Compares MLP vs Transformer architectures

## Next Steps

1. **Root Cause Analysis**: Investigate D2 initialization and output scaling
2. **Architecture Fixes**: Consider normalization layers or output scaling
3. **Hyperparameter Tuning**: Test different population sizes, learning rates
4. **Alternative Training**: Explore different training approaches

# Diffusion Timestep Analysis Experiment

This experiment analyzes the impact of coarse-grained time step parameters on diffusion model sampling quality for D1-D4 models.

## Overview

The experiment:
1. **Loads or trains** D1, D2, D3, D4 models on a specified dataset
2. **Tests different numbers** of temporal discretizations during sampling
3. **Evaluates sample quality** using distributional metrics (RMSE, KS statistic, Wasserstein distance, Std RMSE)
4. **Generates plots** showing the relationship between time steps and quality

## Usage

### Basic Usage
```bash
# Run analysis on OU process with default settings
python src/experiments/diffusion_timestep_analysis.py --dataset ou_process

# Run on Heston model with custom timesteps
python src/experiments/diffusion_timestep_analysis.py --dataset heston --timesteps 5,10,15,20,25,30,40,50

# Force retraining of all models
python src/experiments/diffusion_timestep_analysis.py --dataset brownian --force-retrain --epochs 100
```

### Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--dataset` | Dataset to use for analysis | **Required** | `ou_process`, `heston`, `brownian`, `rbergomi`, `fbm_h03`, `fbm_h04`, `fbm_h06`, `fbm_h07` |
| `--timesteps` | Comma-separated timestep values | `5,10,15,20,25,30,40,50` | Any positive integers |
| `--force-retrain` | Force retraining of all models | `False` | Flag |
| `--epochs` | Number of training epochs | `50` | Positive integer |
| `--num-samples` | Number of samples for evaluation | `1000` | Positive integer |
| `--device` | Device to use | `auto` | `auto`, `cuda`, `cpu` |

### Example Commands

```bash
# Quick test with minimal parameters
python src/experiments/diffusion_timestep_analysis.py --dataset ou_process --timesteps 5,10,15 --epochs 10 --num-samples 100

# Comprehensive analysis on Heston model
python src/experiments/diffusion_timestep_analysis.py --dataset heston --timesteps 5,10,15,20,25,30,40,50,60,80,100 --epochs 200 --num-samples 2000

# Analysis on fractional Brownian motion
python src/experiments/diffusion_timestep_analysis.py --dataset fbm_h03 --timesteps 10,20,30,40,50 --epochs 100

# GPU training with forced retraining
python src/experiments/diffusion_timestep_analysis.py --dataset rbergomi --force-retrain --device cuda --epochs 150
```

## Model Availability

The experiment automatically detects available diffusion models:

| Model | Description | Status |
|-------|-------------|--------|
| **D1** | Time Series Diffusion (Transformer-based) | ⚠️ May have import issues |
| **D2** | Distributional Diffusion + Signature Kernels | ✅ Available |
| **D3** | Distributional Diffusion + PDE-Solved Signatures | ✅ Available |
| **D4** | Distributional Diffusion + Truncated Signatures | ✅ Available |

## Output Files

The experiment generates several output files in `results/{dataset}_diffusion_timestep_analysis/`:

### 1. Results Data
- **`timestep_analysis_results.json`**: Complete numerical results in JSON format
  - Contains metrics for each model at each timestep
  - Includes metadata (dataset, timesteps, timestamp)

### 2. Visualization Plots
- **`{dataset}_timestep_analysis.png`**: Combined 2x2 subplot showing all metrics
- **`{dataset}_rmse_timestep_analysis.png`**: Individual RMSE plot
- **`{dataset}_ks_statistic_timestep_analysis.png`**: Individual KS statistic plot  
- **`{dataset}_wasserstein_distance_timestep_analysis.png`**: Individual Wasserstein distance plot
- **`{dataset}_std_rmse_timestep_analysis.png`**: Individual empirical std RMSE plot

### Plot Features
- **Line plots with scatter points** for each model
- **Different colors and markers** for easy model identification
- **Log scale** for KS statistic and Wasserstein distance (when appropriate)
- **Grid lines** and **legends** for clarity
- **High-resolution** (300 DPI) for publication quality

## Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **RMSE** | Root Mean Square Error | Point-wise trajectory matching (lower = better) |
| **KS Statistic** | Kolmogorov-Smirnov test | Distribution similarity (lower = better) |
| **Wasserstein Distance** | Earth Mover's Distance | Distribution similarity (lower = better) |
| **Std RMSE** | Empirical Standard Deviation RMSE | Variance structure matching (lower = better) |

## Expected Results

### Typical Patterns
1. **Quality vs Speed Trade-off**: More timesteps generally improve quality but increase computation time
2. **Diminishing Returns**: Quality improvements plateau after a certain number of timesteps
3. **Model Differences**: 
   - **D2**: Base distributional diffusion
   - **D3**: Higher accuracy with PDE-solved signatures
   - **D4**: Fastest with truncated signatures

### Interpretation Guidelines
- **Low timesteps (5-10)**: Fast sampling, potentially lower quality
- **Medium timesteps (15-30)**: Good balance of speed and quality
- **High timesteps (40+)**: Best quality, slower sampling

## Troubleshooting

### Common Issues

1. **Model Import Errors**
   ```
   Warning: D1 model not available
   ```
   - **Solution**: Some models may have dependency issues. The experiment continues with available models.

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   - **Solution**: Use `--device cpu` or reduce `--num-samples`

3. **Training Failures**
   ```
   ❌ Failed to process D2: ...
   ```
   - **Solution**: Check model implementation, reduce complexity, or use `--force-retrain`

### Performance Tips

1. **Fast Testing**: Use `--epochs 10 --num-samples 100` for quick validation
2. **GPU Acceleration**: Use `--device cuda` if available
3. **Incremental Analysis**: Start with fewer timesteps, then expand range
4. **Batch Processing**: Run multiple datasets in separate commands

## Integration with Main Pipeline

This experiment complements the main evaluation pipeline:

1. **Training**: Uses the same model creation and training infrastructure
2. **Evaluation**: Uses the same metrics as `enhanced_model_evaluation.py`
3. **Persistence**: Integrates with dataset persistence system
4. **Checkpointing**: Can load existing trained models

## Example Analysis Workflow

```bash
# 1. Quick validation on small dataset
python src/experiments/diffusion_timestep_analysis.py --dataset ou_process --timesteps 5,10,15 --epochs 10 --num-samples 100

# 2. Comprehensive analysis on main dataset
python src/experiments/diffusion_timestep_analysis.py --dataset ou_process --timesteps 5,10,15,20,25,30,40,50 --epochs 100 --num-samples 1000

# 3. Compare across different datasets
python src/experiments/diffusion_timestep_analysis.py --dataset heston --timesteps 5,10,15,20,25,30,40,50 --epochs 100
python src/experiments/diffusion_timestep_analysis.py --dataset brownian --timesteps 5,10,15,20,25,30,40,50 --epochs 100

# 4. Analyze results and plots in results/{dataset}_diffusion_timestep_analysis/
```

## Future Enhancements

Potential improvements for the experiment:
1. **Computational Time Tracking**: Measure sampling time vs quality trade-offs
2. **Statistical Significance**: Add confidence intervals and significance tests
3. **Adaptive Timestep Selection**: Automatically find optimal timestep ranges
4. **Cross-Dataset Comparison**: Generate comparative plots across datasets
5. **Model Architecture Analysis**: Study how model complexity affects timestep sensitivity


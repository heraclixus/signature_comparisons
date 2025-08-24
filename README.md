# Signature-Based Deep Learning for Stochastic Process Generation

Comprehensive comparison of signature-based methods for time series generation across multiple stochastic processes.

## 🏆 Model Performance Rankings

### Top Models (17 Models Across 6 Datasets)

| Rank | Model | Type | Score | Best Use Case |
|------|-------|------|-------|---------------|
| 🥇 | **B4** | Neural SDE + MMD | **4.21** | **🏆 Champion - Distributional quality** |
| 🥈 | **B3** | Neural SDE + T-Stat | **5.21** | **Mean-reverting processes** |
| 🥉 | **V2** | 🧠 SDE Matching | **5.32** | **Latent SDE champion** |
| 4th | **D1** | 🌊 Diffusion | **6.46** | **Transformer-based generation** |
| 5th | **C4** | 🔬 Hybrid SDE + T-Stat | **7.22** | **Multi-objective training** |
| 6th | **C5** | 🔬 Hybrid SDE + Scoring | **7.59** | **Hybrid signature methods** |
| 7th | **C6** | 🔬 Hybrid SDE + MMD | **7.62** | **Advanced hybrid approach** |
| 8th | **B5** | Neural SDE + Scoring | **8.05** | **Fast Neural SDE** |
| 9th | **B2** | Neural SDE + MMD (PDE) | **8.47** | **PDE-solved signatures** |
| 10th | **C3** | 🔬 Latent SDE + MMD | **10.35** | **Hybrid latent approach** |

*Rankings based on weighted distributional metrics (KS statistic + Wasserstein distance) across 6 stochastic processes*

## 📋 Model Overview

### Model Types
- **A-Series**: CannedNet generators with signature losses
- **B-Series**: Neural SDE generators with signature methods  
- **C-Series**: 🔬 Hybrid models combining latent SDE + signatures
- **D1**: 🌊 Diffusion model with transformer architecture
- **V1/V2**: 🧠 Pure latent SDE models

### Key Architecture Components
- **Neural SDE**: Stochastic differential equation generators
- **Latent SDE**: SDE in latent space with variational inference
- **Signature Methods**: Truncated or PDE-solved signature computation
- **Diffusion**: Transformer + discrete diffusion process

## 📊 Evaluation Methodology

Evaluation emphasizes **distributional quality** over point-wise accuracy:
- **KS Statistic** (weight 3.0): Distribution similarity
- **Wasserstein Distance** (weight 2.5): Earth Mover's Distance  
- **Empirical Std RMSE** (weight 2.0): Variance structure matching
- **RMSE** (weight 1.0): Point-wise trajectory error

## 🚀 Quick Start

```bash
# Setup environment
conda activate sig19

# Train all models on all datasets
python src/experiments/train_and_save_models.py --epochs 1000

# Evaluate models and generate rankings
python src/experiments/enhanced_model_evaluation.py
python src/experiments/multi_dataset_evaluation.py

# Train single model (quick test)
python src/experiments/train_and_save_models.py --model B4 --dataset ou_process --epochs 100
```

### Available Datasets
- **ou_process**: Ornstein-Uhlenbeck (mean-reverting)
- **heston**: Heston stochastic volatility
- **rbergomi**: Rough Bergomi model  
- **brownian**: Standard Brownian motion
- **fbm_h03/h04/h06/h07**: Fractional Brownian Motion (various H values)

### Key Results Files

- `results/cross_dataset_analysis/overall_model_summary.csv` - Complete performance rankings
- `results/cross_dataset_analysis/cross_dataset_*_ranking.png` - Individual metric rankings
- `results/cross_dataset_analysis/rough_vs_nonrough_*.png` - Rough vs non-rough analysis
- `results/{dataset}/evaluation/enhanced_models_evaluation.csv` - Per-dataset results
- `results/{dataset}/evaluation/ultra_clear_trajectory_visualization.png` - Trajectory plots

## 📊 Key Results

### Performance Analysis
![Cross-Dataset Rankings](results/cross_dataset_analysis/cross_dataset_ks_statistic_mean_ranking.png)
*KS Statistic distribution quality ranking across all datasets*

![Rough vs Non-Rough](results/cross_dataset_analysis/rough_vs_nonrough_analysis.png)
*Performance comparison: rough vs non-rough stochastic processes*

### Evaluation Coverage
- **17 models** evaluated across **6 stochastic processes**
- **103 total evaluations** (A3 outlier excluded)
- **4 key metrics**: RMSE, KS statistic, Wasserstein distance, Std RMSE

## 🎯 Model Recommendations

### Top Choices
- **🏆 Overall Champion**: **B4** (Neural SDE + MMD) - Best distributional quality
- **Runner-up**: **B3** (Neural SDE + T-Statistic) - Excellent for mean-reverting processes  
- **Latent SDE Leader**: **V2** (SDE Matching) - Top latent approach
- **Advanced Option**: **D1** (Diffusion) - Transformer-based generation

### By Process Type
- **Mean-Reverting**: B3, B4, V2
- **Financial/Volatility**: B4, V2, C4
- **General Purpose**: B4, B3, V2

## 🔧 Usage Example

```python
import sys
sys.path.append('src')
from utils.model_checkpoint import create_checkpoint_manager

# Load trained model
checkpoint_manager = create_checkpoint_manager('results/ou_process')
model = checkpoint_manager.load_model('B4')  # Champion model

# Generate samples
samples = model.generate_samples(100)
print(f"Generated samples: {samples.shape}")
```

## 🎯 Key Findings

- **🏆 B4 (Neural SDE + MMD)** achieves best overall distributional performance
- **Neural SDE generators** significantly outperform CannedNet architectures  
- **Latent SDE models** (V1, V2) show competitive performance with fewer parameters
- **Multi-dataset validation** reveals true model capabilities vs single-dataset results
- **103 total evaluations** completed across 17 models and 6 stochastic processes

---

*Comprehensive systematic comparison of signature-based deep learning methods for stochastic process generation.*
# Signature-Based Deep Learning for Stochastic Process Generation

A comprehensive comparison of signature-based methods for time series generation, including both non-adversarial and adversarial training approaches across multiple stochastic processes.

## 📋 Model Overview

### Complete Model Architecture Descriptions

| Model ID | Generator Type | Loss Function | Signature Method | Training Type | Description |
|----------|----------------|---------------|------------------|---------------|-------------|
| **A1** | CannedNet | T-Statistic | Truncated | Non-Adversarial | Baseline model with simple generator and statistical loss |
| **A2** | CannedNet | Signature Scoring | Truncated | Non-Adversarial | CannedNet with signature-based scoring loss |
| **A3** | CannedNet | Signature MMD | Truncated | Non-Adversarial | CannedNet with Maximum Mean Discrepancy loss |
| **A4** | CannedNet | Log Signature | Truncated | Non-Adversarial | CannedNet with logarithmic signature features |
| **B1** | Neural SDE | Signature Scoring | PDE-Solved | Non-Adversarial | Neural SDE generator with signature scoring |
| **B2** | Neural SDE | Signature MMD | PDE-Solved | Non-Adversarial | Neural SDE with MMD loss and PDE-solved signatures |
| **B3** | Neural SDE | T-Statistic | PDE-Solved | Non-Adversarial | Neural SDE with statistical T-test loss |
| **B4** | Neural SDE | Signature MMD | Truncated | Non-Adversarial | Neural SDE with MMD and truncated signatures |
| **B5** | Neural SDE | Signature Scoring | Truncated | Non-Adversarial | Neural SDE with scoring and truncated signatures |
| **A2_ADV** | CannedNet | Signature Scoring | Truncated | ⚔️ Adversarial | A2 with adversarial discriminator |
| **A3_ADV** | CannedNet | Signature MMD | Truncated | ⚔️ Adversarial | A3 with adversarial discriminator |
| **B1_ADV** | Neural SDE | Signature Scoring | PDE-Solved | ⚔️ Adversarial | B1 with adversarial discriminator |
| **B2_ADV** | Neural SDE | Signature MMD | PDE-Solved | ⚔️ Adversarial | B2 with adversarial discriminator |
| **B4_ADV** | Neural SDE | Signature MMD | Truncated | ⚔️ Adversarial | B4 with adversarial discriminator |
| **B5_ADV** | Neural SDE | Signature Scoring | Truncated | ⚔️ Adversarial | B5 with adversarial discriminator |
| **V1** | Latent SDE | ELBO | N/A | 🧠 Latent SDE | TorchSDE with OU process prior + learned posterior |
| **V2** | Latent SDE | SDE Matching | N/A | 🧠 Latent SDE | Prior/posterior networks with 3-component loss |
| **C1** | Latent SDE | ELBO + T-Statistic | Truncated | 🔬 Hybrid | V1 Latent SDE + signature T-statistic loss |
| **C2** | Latent SDE | ELBO + Signature Scoring | Truncated | 🔬 Hybrid | V1 Latent SDE + signature scoring loss |
| **C3** | Latent SDE | ELBO + Signature MMD | Truncated | 🔬 Hybrid | V1 Latent SDE + signature MMD loss |
| **C4** | SDE Matching | SDE Matching + T-Statistic | Truncated | 🔬 Hybrid | V2 SDE Matching + signature T-statistic loss |
| **C5** | SDE Matching | SDE Matching + Signature Scoring | Truncated | 🔬 Hybrid | V2 SDE Matching + signature scoring loss |
| **C6** | SDE Matching | SDE Matching + Signature MMD | Truncated | 🔬 Hybrid | V2 SDE Matching + signature MMD loss |
| **D1** | Diffusion | Denoising Loss | N/A | 🌊 Diffusion | Transformer + discrete diffusion with GP-structured noise |

### Key Architecture Components

#### **Generator Types:**
- **CannedNet**: Simple feedforward neural network generator
- **Neural SDE**: Stochastic Differential Equation-based generator with drift/diffusion networks
- **Latent SDE**: SDE operating in latent space with encoder/decoder components
- **Diffusion**: Transformer-based denoising network with reverse diffusion process

#### **Loss Functions:**
- **T-Statistic**: Statistical hypothesis testing loss
- **Signature Scoring**: Signature-based scoring function loss
- **Signature MMD**: Maximum Mean Discrepancy using signature kernels
- **Log Signature**: Logarithmic signature feature matching
- **ELBO**: Evidence Lower Bound (variational inference)
- **SDE Matching**: 3-component loss (prior KL + SDE matching + reconstruction)
- **Denoising Loss**: Diffusion model noise prediction loss (MSE)

#### **Signature Methods:**
- **Truncated**: Standard truncated signature computation
- **PDE-Solved**: Signatures computed via PDE solving methods
- **N/A**: Not applicable (latent SDE models don't use signatures directly)

#### **Training Types:**
- **Non-Adversarial**: Direct loss optimization
- **⚔️ Adversarial**: Generator vs discriminator training
- **🧠 Latent SDE**: Variational inference in latent SDE space
- **🔬 Hybrid**: Multi-objective training combining SDE and signature losses
- **🌊 Diffusion**: Reverse diffusion process with denoising objective

### Quick Reference

#### **Best Models by Category:**
- **🏆 Overall Champion**: V2 (SDE Matching) - Rank #1
- **Traditional Signature Champion**: B4 (Neural SDE + MMD) - Rank #2  
- **Efficient Alternative**: V1 (TorchSDE Latent SDE) - Rank #5, only 4.5K parameters
- **Best Adversarial**: B5_ADV (Neural SDE + Adversarial Scoring) - Rank #6

#### **Model Families:**
- **A-Series (A1-A4)**: CannedNet-based models with different signature losses
- **B-Series (B1-B5)**: Neural SDE-based models with signature methods
- **V-Series (V1-V2)**: Latent SDE models with variational inference
- **_ADV Variants**: Adversarial training versions of compatible models

## 🏆 Model Performance Rankings

### Complete Model Performance (17 Models Across 8 Datasets)

| Rank | Model | Training Type | Weighted Score | Best Use Case |
|------|-------|---------------|----------------|---------------|
| 🥇 | **V2** | 🧠 Latent SDE | **4.35** | **🏆 NEW CHAMPION - SDE Matching** |
| 🥈 | **B4** | Non-Adversarial | **4.94** | **Traditional signature champion** |
| 🥉 | **B3** | Non-Adversarial | **5.38** | **Mean-reverting processes** |
| 4th | **B2** | Non-Adversarial | **5.85** | **Advanced PDE-solved signatures** |
| 5th | **V1** | 🧠 Latent SDE | **6.24** | **Efficient TorchSDE Latent SDE** |
| 6th | **B5_ADV** | ⚔️ Adversarial | **6.69** | **🏆 Best adversarial model** |
| 7th | **B1** | Non-Adversarial | **7.71** | **PDE-solved scoring** |
| 8th | **B2_ADV** | ⚔️ Adversarial | **8.24** | **Adversarial MMD (PDE)** |
| 9th | **B5** | Non-Adversarial | **9.74** | **Fast Neural SDE** |
| 10th | **A4** | Non-Adversarial | **10.18** | **Log signature experiments** |
| 11th | **B1_ADV** | ⚔️ Adversarial | **10.43** | **Adversarial scoring (PDE)** |
| 12th | **B4_ADV** | ⚔️ Adversarial | **10.66** | **Adversarial MMD** |
| 13th | **A1** | Non-Adversarial | **10.70** | **Baseline comparison** |
| 14th | **A2_ADV** | ⚔️ Adversarial | **11.75** | **Adversarial CannedNet scoring** |
| 15th | **A3_ADV** | ⚔️ Adversarial | **11.82** | **Adversarial CannedNet MMD** |
| 16th | **A2** | Non-Adversarial | **13.08** | **CannedNet scoring baseline** |
| 17th | **A3** | Non-Adversarial | **15.23** | **CannedNet MMD baseline** |

*Rankings based on distributional quality (KS statistic + Wasserstein distance) across 8 stochastic processes*

**Note**: T-statistic models (A1, B3) cannot be used with adversarial training due to signature dimension conflicts.

## 📊 Evaluation Methodology

### Distributional Metrics (Primary Focus)
Our evaluation prioritizes **distributional quality** over point-wise accuracy, as appropriate for stochastic processes:

| Metric | Weight | Description | Interpretation |
|--------|--------|-------------|----------------|
| **KS Statistic** | 3.0 | Kolmogorov-Smirnov test statistic | Lower = better distribution matching |
| **Wasserstein Distance** | 2.5 | Earth Mover's Distance | Lower = better distribution similarity |
| **Empirical Std RMSE** | 2.0 | Variance structure matching | Lower = better temporal variance matching |
| **RMSE** | 1.0 | Point-wise trajectory error | Lower = better trajectory matching |

### Weighted Ranking Formula
```
Weighted Score = (KS_Rank × 3.0 + Wasserstein_Rank × 2.5 + Std_RMSE_Rank × 2.0 + RMSE_Rank × 1.0) / 8.5
```

This weighting emphasizes **distributional quality** and **variance structure** over simple point-wise accuracy, which is more appropriate for evaluating stochastic process generators.

## 🚀 Quick Start

### Environment Setup
```bash
# Activate the conda environment
conda activate sig19

# Navigate to project directory
cd /path/to/signature_comparisons
```

n### 📊 Dataset Generation (Recommended First Step)

**Important**: As of the latest update, datasets are now pre-generated and saved to disk for faster training. This avoids regenerating data on every training run.

**New Dataset Specifications:**
- **Sample Paths**: 32,768 (128 × 256) for comprehensive training
- **Time Points**: 64 per path for efficient computation
- **Batch Size**: 128 for optimal training performance
- **Storage**: ~16 MB per dataset, organized in separate subdirectories
- **Implementation**: Uses proper mathematical models (HestonModel, rBergomi classes)

#### Generate All Datasets
```bash
# Generate all datasets with default parameters (32,768 samples, 64 time points)
python src/scripts/regenerate_all_datasets.py

# Generate with custom parameters
python src/scripts/generate_datasets.py --samples 1000 --points 50

# Generate specific datasets only
python src/scripts/generate_datasets.py --datasets ou_process,heston,brownian
```

#### Check Saved Datasets
```bash
# List all currently saved datasets
python src/scripts/generate_datasets.py --list

# Clean up old dataset versions (keep 3 most recent)
python src/scripts/generate_datasets.py --cleanup 3
```

#### Available Datasets
- **ou_process**: Ornstein-Uhlenbeck (mean-reverting)
- **heston**: Heston stochastic volatility model  
- **rbergomi**: Rough Bergomi model
- **brownian**: Standard Brownian motion
- **fbm_h03**: Fractional Brownian Motion (H=0.3, anti-persistent)
- **fbm_h04**: Fractional Brownian Motion (H=0.4, anti-persistent)
- **fbm_h06**: Fractional Brownian Motion (H=0.6, persistent)
- **fbm_h07**: Fractional Brownian Motion (H=0.7, persistent)

### 🎨 Dataset Visualization

Visualize sample paths and statistical properties of generated datasets:

```bash
# Visualize all datasets
python src/scripts/visualize_datasets.py

# Visualize specific datasets
python src/scripts/visualize_datasets.py --datasets ou_process,heston,fbm_h03

# Customize visualization
python src/scripts/visualize_datasets.py --num-paths 15 --save-format pdf --dpi 600

# List available datasets for visualization
python src/scripts/visualize_datasets.py --list
```

**Generated Visualizations:**
- **Sample Paths**: Multiple trajectory plots showing stochastic behavior
- **Statistical Summary**: Distribution analysis, confidence bands, variance evolution, and correlation matrices
- **Saved Location**: `data/{dataset_name}/visualizations/`

### Training Models

#### Non-Adversarial Training
```bash
# Train all models on all datasets (recommended)
python src/experiments/train_and_save_models.py --epochs 1000

# Train on specific dataset
python src/experiments/train_and_save_models.py --dataset ou_process --epochs 1000
python src/experiments/train_and_save_models.py --dataset heston --epochs 1000

# Force retrain existing models
python src/experiments/train_and_save_models.py --retrain-all --epochs 1000

# GPU training (if CUDA available)
python src/experiments/train_and_save_models.py --device cuda --epochs 1000

# Fast prototyping with small datasets
python src/experiments/train_and_save_models.py --test-mode --epochs 50

# Train single specific model (quick testing)
python src/experiments/train_and_save_models.py --model D1 --dataset ou_process --epochs 100
python src/experiments/train_and_save_models.py --model A3 --test-mode --epochs 10
python src/experiments/train_and_save_models.py --model C1 --retrain-all --epochs 200
```

#### Training Options
```bash
# Device selection
--device auto    # Auto-detect best device (default)
--device cuda    # Force GPU training
--device cpu     # Force CPU training

# Dataset size options
--test-mode      # Use 1,000 samples for fast prototyping
                 # (vs 32,768 samples in normal mode)

# Single model training
--model MODEL    # Train only specific model (A1, A2, A3, A4, B1, B2, B3, B4, B5, 
                 # C1, C2, C3, C4, C5, C6, D1) - great for quick testing

# Memory optimization
--memory-opt     # Enable memory-efficient training for B-type models

# Examples
python src/experiments/train_and_save_models.py --test-mode --device cpu --epochs 20
python src/experiments/train_and_save_models.py --device cuda --memory-opt --epochs 1000
```

#### Adversarial Training
```bash
# Train all working adversarial models (memory-efficient)
python src/experiments/adversarial_training.py --all --epochs 100 --memory-efficient

# Train specific adversarial models
python src/experiments/adversarial_training.py --models B4 B5 --epochs 50 --memory-efficient

# Force retrain adversarial models
python src/experiments/adversarial_training.py --all --force-retrain --epochs 50 --memory-efficient
```

#### Latent SDE Training 🧠
```bash
# Train all latent SDE models (V1 + V2)
python src/experiments/latent_sde_training.py --all --epochs 100

# Train specific latent SDE models
python src/experiments/latent_sde_training.py --models V1 V2 --epochs 50

# Train on all datasets
python src/experiments/latent_sde_training.py --all --all-datasets --epochs 50

# Force retrain latent SDE models
python src/experiments/latent_sde_training.py --all --force-retrain --epochs 50
```

### Model Evaluation

#### Complete Evaluation Pipeline
```bash
# Evaluate all models (both non-adversarial and adversarial)
python src/experiments/enhanced_model_evaluation.py

# Generate cross-dataset rankings and clean plots
python src/experiments/multi_dataset_evaluation.py
```

### Key Results Files

After evaluation, you'll find:

#### Cross-Dataset Analysis
- `results/cross_dataset_analysis/cross_dataset_rmse_mean_ranking.png` - ✨ **NEW** Individual RMSE performance ranking
- `results/cross_dataset_analysis/cross_dataset_ks_statistic_mean_ranking.png` - ✨ **NEW** Individual KS statistic ranking
- `results/cross_dataset_analysis/cross_dataset_wasserstein_distance_mean_ranking.png` - ✨ **NEW** Individual Wasserstein distance ranking
- `results/cross_dataset_analysis/cross_dataset_std_rmse_mean_ranking.png` - ✨ **NEW** Individual empirical std matching ranking
- `results/cross_dataset_analysis/distributional_quality_ranking.png` - Legacy aggregated distributional ranking
- `results/cross_dataset_analysis/rmse_accuracy_ranking.png` - Legacy RMSE ranking
- `results/cross_dataset_analysis/overall_model_summary.csv` - Complete performance data

#### Rough vs Non-Rough Process Analysis (Individual Metrics)
- `results/cross_dataset_analysis/rough_vs_nonrough_analysis.png` - ✨ **NEW** 2x4 individual metric comparison
- `results/cross_dataset_analysis/rough_vs_nonrough_rmse_comparison.png` - ✨ **NEW** Individual RMSE: rough vs non-rough
- `results/cross_dataset_analysis/rough_vs_nonrough_ks_statistic_comparison.png` - ✨ **NEW** Individual KS: rough vs non-rough  
- `results/cross_dataset_analysis/rough_vs_nonrough_wasserstein_distance_comparison.png` - ✨ **NEW** Individual Wasserstein: rough vs non-rough
- `results/cross_dataset_analysis/rough_vs_nonrough_std_rmse_comparison.png` - ✨ **NEW** Individual Std RMSE: rough vs non-rough
- `results/cross_dataset_analysis/rough_datasets_rankings.csv` - Rankings on rough processes only
- `results/cross_dataset_analysis/nonrough_datasets_rankings.csv` - Rankings on non-rough processes only

#### Individual Dataset Rankings (32 plots total: 8 datasets × 4 metrics)
- `results/cross_dataset_analysis/individual_dataset_rankings/{dataset}_{metric}_ranking.png` - ✨ **NEW** Per-dataset metric rankings
- `results/cross_dataset_analysis/individual_dataset_rankings/{dataset}_rankings.csv` - Per-dataset detailed results

#### Adversarial vs Non-Adversarial Comparison  
- `results/adversarial_comparison/adversarial_vs_non_adversarial_comparison.png` - Side-by-side comparison

#### Individual Dataset Results
- `results/{dataset}/evaluation/enhanced_models_evaluation.csv` - Non-adversarial metrics
- `results/{dataset}_adversarial/evaluation/enhanced_models_evaluation.csv` - Adversarial metrics
- `results/{dataset}/evaluation/ultra_clear_trajectory_visualization.png` - Trajectory plots
- `results/{dataset}/evaluation/rmse_ranking_{dataset}.png` - ✨ **NEW** Individual RMSE ranking per dataset
- `results/{dataset}/evaluation/ks_statistic_ranking_{dataset}.png` - ✨ **NEW** Individual KS ranking per dataset
- `results/{dataset}/evaluation/wasserstein_distance_ranking_{dataset}.png` - ✨ **NEW** Individual Wasserstein ranking per dataset
- `results/{dataset}/evaluation/std_rmse_ranking_{dataset}.png` - ✨ **NEW** Individual Std RMSE ranking per dataset

#### Latent SDE Results 🧠
- `results/{dataset}_latent_sde/evaluation/enhanced_models_evaluation.csv` - V1 + V2 metrics
- `results/{dataset}_latent_sde/evaluation/ultra_clear_trajectory_visualization.png` - V1 vs V2 trajectories
- `results/{dataset}_latent_sde/evaluation/enhanced_model_comparison.png` - V1 vs V2 comparison
- `results/{dataset}_latent_sde/evaluation/rmse_ranking_{dataset}_latent_sde.png` - ✨ **NEW** V1 vs V2 RMSE ranking
- `results/{dataset}_latent_sde/evaluation/ks_statistic_ranking_{dataset}_latent_sde.png` - ✨ **NEW** V1 vs V2 KS ranking
- `results/{dataset}_latent_sde/evaluation/wasserstein_distance_ranking_{dataset}_latent_sde.png` - ✨ **NEW** V1 vs V2 Wasserstein ranking
- `results/{dataset}_latent_sde/evaluation/std_rmse_ranking_{dataset}_latent_sde.png` - ✨ **NEW** V1 vs V2 Std RMSE ranking
- `results/{dataset}_latent_sde/training/{dataset}_latent_sde_training_summary.csv` - Training history

## 📊 Key Results

### Cross-Dataset Performance Analysis
Our evaluation system generates clean, publication-ready visualizations with individual distributional metrics:

#### Individual Distributional Metric Rankings

##### RMSE Performance Ranking
![RMSE Ranking](results/cross_dataset_analysis/cross_dataset_rmse_mean_ranking.png)
*Point-wise trajectory matching accuracy across all datasets*

##### KS Statistic Distribution Quality
![KS Statistic Ranking](results/cross_dataset_analysis/cross_dataset_ks_statistic_mean_ranking.png)
*Statistical distribution similarity ranking across all datasets*

##### Wasserstein Distance Distribution Quality  
![Wasserstein Distance Ranking](results/cross_dataset_analysis/cross_dataset_wasserstein_distance_mean_ranking.png)
*Earth Mover's Distance between distributions across all datasets*

##### Empirical Standard Deviation Matching
![Std RMSE Ranking](results/cross_dataset_analysis/cross_dataset_std_rmse_mean_ranking.png)
*Variance structure matching over time across all datasets*

#### Adversarial vs Non-Adversarial Comparison
![Adversarial Comparison](results/adversarial_comparison/adversarial_vs_non_adversarial_comparison.png)
*Direct side-by-side comparison of adversarial and non-adversarial training*

#### Example: Trajectory Quality Analysis
![Trajectory Visualization](results/ou_process/evaluation/ultra_clear_trajectory_visualization.png)
*Generated trajectories vs ground truth for OU Process dataset (20 samples per model)*

#### Latent SDE Model Performance 🧠
![Latent SDE Trajectories](results/ou_process_latent_sde/evaluation/ultra_clear_trajectory_visualization.png)
*V1 (TorchSDE) vs V2 (SDE Matching) trajectories on OU Process dataset*

**V1 (TorchSDE Latent SDE):**
- **Architecture**: OU process prior + learned neural SDE posterior
- **Parameters**: 4,483 (very efficient)
- **Performance**: RMSE 0.345, KS 0.183
- **Best for**: Mean-reverting processes, computational efficiency

**V2 (SDE Matching):**
- **Architecture**: Learnable prior + neural SDE + observation model
- **Parameters**: 23,133 (more complex)
- **Performance**: RMSE 0.549, KS 0.147
- **Best for**: Distributional quality, sophisticated architectures

#### Rough vs Non-Rough Process Analysis (Individual Metrics)
![Rough vs Non-Rough Analysis](results/cross_dataset_analysis/rough_vs_nonrough_analysis.png)
*2x4 comparison: Individual distributional metrics on rough vs non-rough processes*

##### Individual Rough vs Non-Rough Metric Comparisons
![RMSE: Rough vs Non-Rough](results/cross_dataset_analysis/rough_vs_nonrough_rmse_comparison.png)
*RMSE performance comparison between rough and non-rough stochastic processes*

![KS Statistic: Rough vs Non-Rough](results/cross_dataset_analysis/rough_vs_nonrough_ks_statistic_comparison.png)
*KS statistic distribution quality comparison between rough and non-rough processes*

![Wasserstein Distance: Rough vs Non-Rough](results/cross_dataset_analysis/rough_vs_nonrough_wasserstein_distance_comparison.png)
*Wasserstein distance distribution quality comparison between rough and non-rough processes*

![Std RMSE: Rough vs Non-Rough](results/cross_dataset_analysis/rough_vs_nonrough_std_rmse_comparison.png)
*Empirical standard deviation matching comparison between rough and non-rough processes*

### Evaluation Coverage
- **17 models** evaluated (9 non-adversarial + 6 adversarial + 2 latent SDE)
- **8 stochastic processes** tested
- **136 total evaluations** completed
- **4 key metrics** per evaluation (RMSE, KS statistic, Wasserstein distance, Std RMSE)
- **64 visualization files** created:
  - **4 cross-dataset** individual metric rankings
  - **4 rough vs non-rough** individual metric comparisons
  - **32 individual dataset** rankings (8 datasets × 4 metrics)
  - **8 legacy** aggregated plots

## 📋 Supported Datasets

### 🌊 Rough Processes (H < 0.5 or inherently rough)
- **rbergomi** - Rough Bergomi volatility model
- **fbm_h03** - Fractional Brownian Motion (H=0.3, anti-persistent)
- **fbm_h04** - Fractional Brownian Motion (H=0.4, anti-persistent)

### 🏔️ Non-Rough Processes (H ≥ 0.5 or smooth)
- **ou_process** - Ornstein-Uhlenbeck (mean-reverting)
- **heston** - Heston stochastic volatility
- **brownian** - Standard Brownian motion (H=0.5, neutral)
- **fbm_h06** - Fractional Brownian Motion (H=0.6, persistent)
- **fbm_h07** - Fractional Brownian Motion (H=0.7, persistent)

*Rough processes exhibit anti-persistence and irregular/jagged paths, while non-rough processes have smoother, more predictable behavior.*

## 🎯 Model Recommendations

### For General Use
- **🏆 NEW CHAMPION**: V2 (SDE Matching) - Best overall distributional performance
- **Traditional Champion**: B4 (Neural SDE + MMD) - Best signature-based model
- **Efficient Alternative**: V1 (TorchSDE Latent SDE) - High performance with fewer parameters
- **Best Adversarial**: B5_ADV (Neural SDE + Adversarial Scoring) - Top adversarial performance

### For Specific Process Types
- **Mean-Reverting (OU-like)**: **V1** 🧠, B3, B4
- **Financial/Volatility (Heston, rBergomi)**: **V2** 🧠, A1, A2  
- **Simple Diffusion (Brownian)**: **V2** 🧠 (champion on 5/8 datasets)
- **Unknown Process Type**: **V2** 🧠 (new overall champion)
- **Computational Efficiency**: **V1** 🧠 (4.5K parameters vs 23K for V2)
- **Distributional Quality**: **V2** 🧠 (best KS + Wasserstein performance)

## 🔧 Using Individual Models

```python
import sys
sys.path.append('src')
import torch
from utils.model_checkpoint import create_checkpoint_manager

# Load any trained model
checkpoint_manager = create_checkpoint_manager('results')
model = checkpoint_manager.load_model('V2')  # NEW CHAMPION! or B4, V1, etc.

# Generate samples
samples = model.generate_samples(100)
print(f"Generated samples: {samples.shape}")
```

#### Using Latent SDE Models 🧠
```python
# Load latent SDE models
v1_model = checkpoint_manager.load_model('V1')  # TorchSDE Latent SDE
v2_model = checkpoint_manager.load_model('V2')  # SDE Matching

# Generate samples
v1_samples = v1_model.generate_samples(100)  # Efficient OU-based generation
v2_samples = v2_model.generate_samples(100)  # High-quality SDE matching

print(f"V1 samples: {v1_samples.shape}")  # (100, 2, time_steps)
print(f"V2 samples: {v2_samples.shape}")  # (100, 2, time_steps)
```

## 📄 Individual Dataset Results

For detailed analysis of each stochastic process, see the individual dataset documentation:

### 🌊 Rough Processes (H < 0.5)
- **[rBergomi Results](rbergomi_results.md)** - Rough Bergomi volatility model analysis
- **[FBM H=0.3 Results](fbm_h03_results.md)** - Anti-persistent FBM analysis  
- **[FBM H=0.4 Results](fbm_h04_results.md)** - Anti-persistent FBM analysis

### 🏔️ Non-Rough Processes (H ≥ 0.5)
- **[OU Process Results](ou_process_results.md)** - Ornstein-Uhlenbeck mean-reverting analysis
- **[Heston Results](heston_results.md)** - Stochastic volatility model analysis
- **[Brownian Motion Results](brownian_results.md)** - Standard Brownian motion analysis
- **[FBM H=0.6 Results](fbm_h06_results.md)** - Persistent FBM analysis
- **[FBM H=0.7 Results](fbm_h07_results.md)** - Persistent FBM analysis

*Each dataset file contains comprehensive model performance analysis with visualizations for non-adversarial, adversarial, and latent SDE approaches.*

## 📁 Repository Structure

```
signature_comparisons/
├── src/
│   ├── models/                    # Model implementations
│   ├── experiments/               # Training and evaluation scripts
│   ├── losses/                    # Loss function implementations
│   └── signatures/                # Signature computation methods
├── results/                       # Generated results and plots
├── README.md                      # This file (main overview)
└── {dataset}_results.md           # Individual dataset analyses (×8)
```

## 🎯 Key Findings

- **🏆 NEW CHAMPION**: **V2 (SDE Matching)** achieves best overall distributional performance across all datasets
- **Latent SDE breakthrough**: Both **V1** and **V2** rank in top 5, showing latent SDE approaches are highly competitive
- **Neural SDE generators** significantly outperform CannedNet architectures
- **Individual metric analysis**: 64 separate visualizations (4 cross-dataset + 4 rough vs non-rough + 32 per-dataset rankings) provide granular insights
- **V2 dominates multiple datasets**: Champion on 5/8 datasets (Brownian, FBM H=0.3,0.4,0.6,0.7)
- **Efficiency vs Quality trade-off**: **V1** offers excellent performance with 5x fewer parameters than **V2**
- **Adversarial training** shows mixed results - only B5_ADV competitive with top models
- **Multi-dataset validation** is critical - single dataset results can be misleading
- **136 total evaluations** completed (17 models × 8 datasets)

---

*This framework provides the most comprehensive systematic comparison of signature-based deep learning methods for stochastic process generation, now including 17 models across 9 non-adversarial, 6 adversarial, and 2 latent SDE approaches.*
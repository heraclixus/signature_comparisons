# Fractional Brownian Motion (H=0.6) - Model Performance Results

## Dataset Overview
**Fractional Brownian Motion (H=0.6)** - Persistent with moderate positive correlations.

**Mathematical Definition:**
```
X_t = ∫₀ᵗ (t-s)^(H-1/2) dW_s
```
Where:
- H = 0.6 (Hurst parameter)
- W_s = standard Brownian motion

**Process Characteristics:**
- **Type**: Non-rough process (H = 0.6 > 0.5)
- **Behavior**: Persistent with moderate positive correlations
- **Properties**: Smooth paths with persistent increments
- **Applications**: Financial time series, climate data, network analysis

---

## 🏆 Dataset-Specific Model Rankings

### Cross-Dataset Ranking (All 17 Models)
![Fractional Brownian Motion (H=0.6) Cross-Dataset Rankings](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h06_rmse_ranking.png)

### Individual Distributional Metric Rankings

#### RMSE Performance Ranking
![RMSE Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h06_rmse_ranking.png)
*Point-wise trajectory matching accuracy on Fractional Brownian Motion (H=0.6) dataset*

#### KS Statistic Distribution Quality  
![KS Statistic Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h06_ks_statistic_ranking.png)
*Statistical distribution similarity ranking on Fractional Brownian Motion (H=0.6) dataset*

#### Wasserstein Distance Distribution Quality
![Wasserstein Distance Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h06_wasserstein_distance_ranking.png)
*Earth Mover's Distance between distributions on Fractional Brownian Motion (H=0.6) dataset*

#### Empirical Standard Deviation Matching
![Std RMSE Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h06_std_rmse_ranking.png)
*Variance structure matching over time on Fractional Brownian Motion (H=0.6) dataset*

---

## 📊 Model Performance Analysis

### Non-Adversarial Models
![Non-Adversarial Model Comparison](results/fbm_h06/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of non-adversarial models on Fractional Brownian Motion (H=0.6)*

#### Trajectory Visualization
![Non-Adversarial Trajectories](results/fbm_h06/evaluation/ultra_clear_trajectory_visualization.png)
*Generated vs ground truth trajectories for non-adversarial models*

#### Empirical Standard Deviation Analysis
![Non-Adversarial Std Analysis](results/fbm_h06/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution over time for non-adversarial models*

### Adversarial Models
![Adversarial Model Comparison](results/fbm_h06_adversarial/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of adversarial models on Fractional Brownian Motion (H=0.6)*

#### Trajectory Visualization
![Adversarial Trajectories](results/fbm_h06_adversarial/evaluation/ultra_clear_trajectory_visualization.png)
*Generated vs ground truth trajectories for adversarial models*

#### Empirical Standard Deviation Analysis
![Adversarial Std Analysis](results/fbm_h06_adversarial/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution over time for adversarial models*

### Latent SDE Models
![Latent SDE Model Comparison](results/fbm_h06_latent_sde/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of latent SDE models (V1 vs V2) on Fractional Brownian Motion (H=0.6)*

#### Trajectory Visualization
![Latent SDE Trajectories](results/fbm_h06_latent_sde/evaluation/ultra_clear_trajectory_visualization.png)
*V1 (TorchSDE) vs V2 (SDE Matching) trajectory comparison*

#### Empirical Standard Deviation Analysis
![Latent SDE Std Analysis](results/fbm_h06_latent_sde/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution for V1 vs V2 models*

---

## 🎯 Fractional Brownian Motion (H=0.6) Dataset Specific Insights

### Dataset Champion: **V2**
- **Weighted Rank**: 1.82 (best on this dataset)
- **KS Statistic**: 0.046 (distribution matching quality)
- **RMSE**: 0.540 (trajectory matching accuracy)
- **Std RMSE**: 0.211 (variance structure matching)

### Key Findings for Fractional Brownian Motion (H=0.6):
- **Dataset champion** demonstrates effectiveness on H=0.6 processes
- **Process characteristics** (H=0.6) influence model performance rankings
- **Distributional quality** varies significantly across different model architectures
- **Individual metric analysis** reveals model strengths and weaknesses

---

*This analysis demonstrates model performance on non-rough process (h = 0.6 > 0.5) with H=0.6, showing how different architectures handle persistent processes.*
# Fractional Brownian Motion (H=0.4) - Model Performance Results

## Dataset Overview
**Fractional Brownian Motion (H=0.4)** - Anti-persistent with moderate negative correlations.

**Mathematical Definition:**
```
X_t = ∫₀ᵗ (t-s)^(H-1/2) dW_s
```
Where:
- H = 0.4 (Hurst parameter)
- W_s = standard Brownian motion

**Process Characteristics:**
- **Type**: Rough process (H = 0.4 < 0.5)
- **Behavior**: Anti-persistent with moderate negative correlations
- **Properties**: Irregular paths with anti-persistent increments
- **Applications**: Rough volatility modeling, financial time series analysis

---

## 🏆 Dataset-Specific Model Rankings

### Cross-Dataset Ranking (All 17 Models)
![Fractional Brownian Motion (H=0.4) Cross-Dataset Rankings](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h04_rmse_ranking.png)

### Individual Distributional Metric Rankings

#### RMSE Performance Ranking
![RMSE Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h04_rmse_ranking.png)
*Point-wise trajectory matching accuracy on Fractional Brownian Motion (H=0.4) dataset*

#### KS Statistic Distribution Quality  
![KS Statistic Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h04_ks_statistic_ranking.png)
*Statistical distribution similarity ranking on Fractional Brownian Motion (H=0.4) dataset*

#### Wasserstein Distance Distribution Quality
![Wasserstein Distance Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h04_wasserstein_distance_ranking.png)
*Earth Mover's Distance between distributions on Fractional Brownian Motion (H=0.4) dataset*

#### Empirical Standard Deviation Matching
![Std RMSE Ranking](results/cross_dataset_analysis/individual_dataset_rankings/fbm_h04_std_rmse_ranking.png)
*Variance structure matching over time on Fractional Brownian Motion (H=0.4) dataset*

---

## 📊 Model Performance Analysis

### Non-Adversarial Models
![Non-Adversarial Model Comparison](results/fbm_h04/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of non-adversarial models on Fractional Brownian Motion (H=0.4)*

#### Trajectory Visualization
![Non-Adversarial Trajectories](results/fbm_h04/evaluation/ultra_clear_trajectory_visualization.png)
*Generated vs ground truth trajectories for non-adversarial models*

#### Empirical Standard Deviation Analysis
![Non-Adversarial Std Analysis](results/fbm_h04/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution over time for non-adversarial models*

### Adversarial Models
![Adversarial Model Comparison](results/fbm_h04_adversarial/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of adversarial models on Fractional Brownian Motion (H=0.4)*

#### Trajectory Visualization
![Adversarial Trajectories](results/fbm_h04_adversarial/evaluation/ultra_clear_trajectory_visualization.png)
*Generated vs ground truth trajectories for adversarial models*

#### Empirical Standard Deviation Analysis
![Adversarial Std Analysis](results/fbm_h04_adversarial/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution over time for adversarial models*

### Latent SDE Models
![Latent SDE Model Comparison](results/fbm_h04_latent_sde/evaluation/enhanced_model_comparison.png)
*Enhanced comparison of latent SDE models (V1 vs V2) on Fractional Brownian Motion (H=0.4)*

#### Trajectory Visualization
![Latent SDE Trajectories](results/fbm_h04_latent_sde/evaluation/ultra_clear_trajectory_visualization.png)
*V1 (TorchSDE) vs V2 (SDE Matching) trajectory comparison*

#### Empirical Standard Deviation Analysis
![Latent SDE Std Analysis](results/fbm_h04_latent_sde/evaluation/empirical_std_analysis.png)
*Empirical standard deviation evolution for V1 vs V2 models*

---

## 🎯 Fractional Brownian Motion (H=0.4) Dataset Specific Insights

### Dataset Champion: **V2**
- **Weighted Rank**: 1.59 (best on this dataset)
- **KS Statistic**: 0.066 (distribution matching quality)
- **RMSE**: 0.608 (trajectory matching accuracy)
- **Std RMSE**: 0.089 (variance structure matching)

### Key Findings for Fractional Brownian Motion (H=0.4):
- **Dataset champion** demonstrates effectiveness on H=0.4 processes
- **Process characteristics** (H=0.4) influence model performance rankings
- **Distributional quality** varies significantly across different model architectures
- **Individual metric analysis** reveals model strengths and weaknesses

---

*This analysis demonstrates model performance on rough process (h = 0.4 < 0.5) with H=0.4, showing how different architectures handle anti-persistent processes.*
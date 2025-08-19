"""
Clean Distributional Analysis

Creates clean, sorted visualizations focusing only on the key distribution metrics
without path correlation, training performance, or text boxes with rankings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def create_clean_distributional_visualizations():
    """Create clean distributional visualizations."""
    print("🎨 Creating Clean Distributional Visualizations")
    print("=" * 50)
    
    # Load results
    results_path = "results/evaluation/enhanced_models_evaluation.csv"
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    print(f"Loaded results for {len(df)} models")
    
    # Sort by KS statistic (best distribution matching first)
    df_sorted = df.sort_values('ks_statistic').reset_index(drop=True)
    models = df_sorted['model_id'].tolist()
    
    # Create figure with 3 clean subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distributional Performance Analysis (Sorted by Distribution Quality)', 
                 fontsize=16, fontweight='bold')
    
    # 1. RMSE (sorted by KS statistic)
    ax1 = axes[0]
    rmse_values = df_sorted['rmse'].tolist()
    bars1 = ax1.bar(range(len(models)), rmse_values, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    ax1.set_title('RMSE by Model', fontweight='bold', fontsize=14)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(rmse_values):
        ax1.text(i, v + max(rmse_values) * 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. KS Statistic (sorted)
    ax2 = axes[1]
    ks_values = df_sorted['ks_statistic'].tolist()
    bars2 = ax2.bar(range(len(models)), ks_values, color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)
    ax2.set_title('KS Statistic (Lower = Better Distribution)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('KS Statistic', fontsize=12)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(ks_values):
        ax2.text(i, v + max(ks_values) * 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Empirical Std RMSE (sorted by KS statistic)
    ax3 = axes[2]
    std_rmse_values = df_sorted['std_rmse'].tolist()
    bars3 = ax3.bar(range(len(models)), std_rmse_values, color='gold', alpha=0.8, edgecolor='orange', linewidth=1)
    ax3.set_title('Empirical Std RMSE (Lower = Better)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Std RMSE', fontsize=12)
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(std_rmse_values):
        ax3.text(i, v + max(std_rmse_values) * 0.02, f'{v:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight the best performer in each metric
    # Best RMSE
    best_rmse_idx = np.argmin(rmse_values)
    bars1[best_rmse_idx].set_color('gold')
    bars1[best_rmse_idx].set_edgecolor('darkorange')
    bars1[best_rmse_idx].set_linewidth(3)
    
    # Best KS (first in sorted order)
    bars2[0].set_color('lightgreen')
    bars2[0].set_edgecolor('darkgreen')
    bars2[0].set_linewidth(3)
    
    # Best Std RMSE
    best_std_idx = np.argmin(std_rmse_values)
    bars3[best_std_idx].set_color('lightgreen')
    bars3[best_std_idx].set_edgecolor('darkgreen')
    bars3[best_std_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    # Save visualization
    save_dir = "results/evaluation"
    plt.savefig(os.path.join(save_dir, 'clean_distributional_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Clean distributional analysis saved to: {os.path.join(save_dir, 'clean_distributional_analysis.png')}")
    
    plt.close()
    
    # Create summary table
    create_summary_table(df_sorted, save_dir)


def create_summary_table(df_sorted: pd.DataFrame, save_dir: str):
    """Create a clean summary table of results."""
    print("Creating summary table...")
    
    # Select key columns for summary
    summary_cols = ['model_id', 'rmse', 'ks_statistic', 'wasserstein_distance', 'std_rmse', 'std_correlation']
    summary_df = df_sorted[summary_cols].copy()
    
    # Round values for readability
    summary_df['rmse'] = summary_df['rmse'].round(4)
    summary_df['ks_statistic'] = summary_df['ks_statistic'].round(4)
    summary_df['wasserstein_distance'] = summary_df['wasserstein_distance'].round(4)
    summary_df['std_rmse'] = summary_df['std_rmse'].round(4)
    summary_df['std_correlation'] = summary_df['std_correlation'].round(4)
    
    # Add rank column
    summary_df.insert(0, 'rank', range(1, len(summary_df) + 1))
    
    # Save summary
    summary_path = os.path.join(save_dir, 'clean_model_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Clean summary table saved to: {summary_path}")
    
    # Print summary to console
    print(f"\n📊 CLEAN MODEL SUMMARY (Sorted by Distribution Quality):")
    print("=" * 80)
    print(summary_df.to_string(index=False))


def main():
    """Main function."""
    create_clean_distributional_visualizations()
    print(f"\n🎉 CLEAN VISUALIZATIONS COMPLETE!")
    print("   - 3 focused plots without unnecessary metrics")
    print("   - Sorted by distribution quality (KS statistic)")
    print("   - Best performers highlighted")
    print("   - Clean summary table included")


if __name__ == "__main__":
    main()

"""
Multi-Dataset Analysis System

Process and analyze results from enhanced_model_evaluation.py across all datasets:
1. Ornstein-Uhlenbeck Process (original)
2. Heston Stochastic Volatility Model
3. Simplified rBergomi Model
4. Standard Brownian Motion

Creates comprehensive cross-dataset performance analysis and rankings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_dataset_results():
    """Load evaluation results from all datasets, including adversarial variants."""
    print("📂 Loading evaluation results from all datasets (including adversarial)...")
    
    base_datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
    all_results = []
    
    for dataset_name in base_datasets:
        # Load non-adversarial results
        results_path = f'results/{dataset_name}/evaluation/enhanced_models_evaluation.csv'
        if os.path.exists(results_path):
            print(f"   ✅ Loading {dataset_name} non-adversarial results...")
            df = pd.read_csv(results_path)
            df['dataset'] = dataset_name
            df['training_type'] = 'non_adversarial'
            all_results.append(df)
        
        # Load adversarial results
        adv_results_path = f'results/{dataset_name}_adversarial/evaluation/enhanced_models_evaluation.csv'
        if os.path.exists(adv_results_path):
            print(f"   ⚔️ Loading {dataset_name} adversarial results...")
            adv_df = pd.read_csv(adv_results_path)
            adv_df['dataset'] = dataset_name
            adv_df['training_type'] = 'adversarial'
            # Add base model ID for comparison
            adv_df['base_model_id'] = adv_df['model_id'].str.replace('_ADV', '')
            all_results.append(adv_df)
        else:
            print(f"   ⚠️ No adversarial results found for {dataset_name}")
    
    if not all_results:
        print("❌ No evaluation results found!")
        print("   Run enhanced_model_evaluation.py first to generate results")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"✅ Loaded results for {len(base_datasets)} datasets, {len(combined_df)} total evaluations")
    print(f"   Non-adversarial models: {len(combined_df[combined_df['training_type'] == 'non_adversarial'])}")
    print(f"   Adversarial models: {len(combined_df[combined_df['training_type'] == 'adversarial'])}")
    
    return combined_df


def compute_cross_dataset_rankings(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Compute model rankings across all datasets with weighted emphasis on distributional metrics."""
    print("📊 Computing cross-dataset rankings...")
    print("   🎯 Emphasizing distributional metrics for stochastic processes")
    
    datasets = combined_df['dataset'].unique()
    models = combined_df['model_id'].unique()
    
    # Define ranking metrics with weights (lower is better for all)
    # Distributional metrics get higher weights since we're dealing with stochastic processes
    ranking_metrics = {
        'ks_statistic': 3.0,        # Primary: Distribution similarity (highest weight)
        'wasserstein_distance': 2.5, # Primary: Distribution distance (high weight)
        'rmse': 1.0,                 # Secondary: Point-wise accuracy (lower weight)
        'std_rmse': 2.0             # Important: Variance structure matching (medium-high weight)
    }
    
    print(f"   Metric weights: KS({ranking_metrics['ks_statistic']}), "
          f"Wasserstein({ranking_metrics['wasserstein_distance']}), "
          f"RMSE({ranking_metrics['rmse']}), StdRMSE({ranking_metrics['std_rmse']})")
    
    rankings_data = []
    
    for dataset in datasets:
        dataset_results = combined_df[combined_df['dataset'] == dataset].copy()
        
        # Compute ranks for each metric (1 = best)
        weighted_ranks = []
        for metric, weight in ranking_metrics.items():
            if metric in dataset_results.columns:
                dataset_results[f'{metric}_rank'] = dataset_results[metric].rank(method='min')
                # Apply weight to the rank (lower ranks get more benefit from higher weights)
                weighted_ranks.append(dataset_results[f'{metric}_rank'] * weight)
        
        # Compute weighted average rank
        if weighted_ranks:
            total_weight = sum(ranking_metrics.values())
            dataset_results['weighted_average_rank'] = sum(weighted_ranks) / total_weight
            
            # Also keep unweighted average for comparison
            rank_columns = [f'{metric}_rank' for metric in ranking_metrics.keys() 
                          if f'{metric}_rank' in dataset_results.columns]
            dataset_results['unweighted_average_rank'] = dataset_results[rank_columns].mean(axis=1)
            
            # Use weighted rank as the primary ranking
            dataset_results['average_rank'] = dataset_results['weighted_average_rank']
        
        rankings_data.append(dataset_results)
    
    # Combine rankings
    rankings_df = pd.concat(rankings_data, ignore_index=True)
    
    # Compute overall statistics
    overall_stats = []
    
    for model in models:
        model_data = rankings_df[rankings_df['model_id'] == model]
        
        if len(model_data) > 0:
            stats_row = {
                'model_id': model,
                'datasets_evaluated': len(model_data),
                'weighted_average_rank_overall': model_data['weighted_average_rank'].mean(),
                'unweighted_average_rank_overall': model_data['unweighted_average_rank'].mean(),
                'best_dataset': model_data.loc[model_data['weighted_average_rank'].idxmin(), 'dataset'] if len(model_data) > 0 else None,
                'worst_dataset': model_data.loc[model_data['weighted_average_rank'].idxmax(), 'dataset'] if len(model_data) > 0 else None,
                
                # Distributional metrics (primary focus)
                'ks_statistic_mean': model_data['ks_statistic'].mean(),
                'ks_statistic_std': model_data['ks_statistic'].std(),
                'ks_rank_mean': model_data['ks_statistic_rank'].mean(),
                'wasserstein_distance_mean': model_data['wasserstein_distance'].mean(),
                'wasserstein_distance_std': model_data['wasserstein_distance'].std(),
                'wasserstein_rank_mean': model_data['wasserstein_distance_rank'].mean(),
                
                # Variance structure metrics (important)
                'std_rmse_mean': model_data['std_rmse'].mean(),
                'std_rmse_std': model_data['std_rmse'].std(),
                'std_rank_mean': model_data['std_rmse_rank'].mean(),
                
                # Point-wise accuracy (secondary)
                'rmse_mean': model_data['rmse'].mean(),
                'rmse_std': model_data['rmse'].std(),
                'rmse_rank_mean': model_data['rmse_rank'].mean(),
                
                # Overall distributional quality score (lower is better)
                'distributional_score': (model_data['ks_statistic_rank'].mean() * 3.0 + 
                                       model_data['wasserstein_distance_rank'].mean() * 2.5) / 5.5
            }
            overall_stats.append(stats_row)
    
    overall_df = pd.DataFrame(overall_stats)
    # Sort by weighted average rank (which emphasizes distributional metrics)
    overall_df = overall_df.sort_values('weighted_average_rank_overall').reset_index(drop=True)
    
    # Add ranking position
    overall_df['overall_rank_position'] = range(1, len(overall_df) + 1)
    
    print(f"✅ Rankings computed for {len(models)} models across {len(datasets)} datasets")
    
    return rankings_df, overall_df


def create_comprehensive_visualizations(rankings_df: pd.DataFrame, overall_df: pd.DataFrame, save_dir: str):
    """Create comprehensive cross-dataset analysis visualizations."""
    print("🎨 Creating comprehensive cross-dataset visualizations...")
    
    datasets = rankings_df['dataset'].unique()
    models = overall_df['model_id'].tolist()
    
    # Create main ranking plots
    create_distributional_ranking_plot(overall_df, save_dir)
    create_rmse_ranking_plot(overall_df, save_dir)
    
    # Create rough vs non-rough analysis
    create_rough_vs_nonrough_analysis(rankings_df, save_dir)
    
    print(f"✅ Comprehensive analysis saved to: {save_dir}/")


def create_distributional_ranking_plot(overall_df: pd.DataFrame, save_dir: str):
    """Create a clean distributional quality ranking plot."""
    print("   📊 Creating distributional quality ranking plot...")
    
    # Sort models by distributional score
    dist_sorted_df = overall_df.sort_values('distributional_score').reset_index(drop=True)
    dist_sorted_models = dist_sorted_df['model_id'].tolist()
    dist_values = dist_sorted_df['distributional_score'].tolist()
    
    # Create figure with more space
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create gradient colors from best (dark green) to worst (light green)
    colors = plt.cm.Greens_r(np.linspace(0.3, 0.9, len(dist_sorted_models)))
    
    bars = ax.bar(range(len(dist_sorted_models)), dist_values, 
                  color=colors, edgecolor='darkgreen', linewidth=2, alpha=0.8)
    
    # Title and labels with better spacing
    ax.set_title('Cross-Dataset Distributional Quality Ranking\n(KS Statistic + Wasserstein Distance)', 
                 fontsize=16, fontweight='bold', pad=30)
    ax.set_ylabel('Average Distributional Score (Lower = Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models (Sorted by Performance)', fontsize=14, fontweight='bold')
    
    # X-axis labels with better spacing
    ax.set_xticks(range(len(dist_sorted_models)))
    ax.set_xticklabels(dist_sorted_models, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars with better positioning
    max_val = max(dist_values)
    for i, (bar, value) in enumerate(zip(bars, dist_values)):
        # Value above bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                f'{value:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Rank number inside bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'#{i+1}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Add legend explaining the metrics
    ax.text(0.02, 0.98, 'Lower scores = Better distributional matching\nBased on KS statistic & Wasserstein distance', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distributional_quality_ranking.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Distributional ranking saved to: distributional_quality_ranking.png")


def create_rmse_ranking_plot(overall_df: pd.DataFrame, save_dir: str):
    """Create a clean RMSE accuracy ranking plot."""
    print("   📊 Creating RMSE accuracy ranking plot...")
    
    # Sort models by RMSE rank
    rmse_sorted_df = overall_df.sort_values('rmse_rank_mean').reset_index(drop=True)
    rmse_sorted_models = rmse_sorted_df['model_id'].tolist()
    rmse_values = rmse_sorted_df['rmse_rank_mean'].tolist()
    
    # Create figure with more space
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create gradient colors from best (dark blue) to worst (light blue)
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(rmse_sorted_models)))
    
    bars = ax.bar(range(len(rmse_sorted_models)), rmse_values, 
                  color=colors, edgecolor='darkblue', linewidth=2, alpha=0.8)
    
    # Title and labels with better spacing
    ax.set_title('Cross-Dataset Point-wise Accuracy Ranking\n(RMSE Trajectory Matching)', 
                 fontsize=16, fontweight='bold', pad=30)
    ax.set_ylabel('Average RMSE Rank (Lower = Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Models (Sorted by Performance)', fontsize=14, fontweight='bold')
    
    # X-axis labels with better spacing
    ax.set_xticks(range(len(rmse_sorted_models)))
    ax.set_xticklabels(rmse_sorted_models, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars with better positioning
    max_val = max(rmse_values)
    for i, (bar, value) in enumerate(zip(bars, rmse_values)):
        # Value above bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        # Rank number inside bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'#{i+1}', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Add legend explaining the metrics
    ax.text(0.02, 0.98, 'Lower ranks = Better trajectory matching\nBased on RMSE across all datasets', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rmse_accuracy_ranking.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ RMSE ranking saved to: rmse_accuracy_ranking.png")


def create_rough_vs_nonrough_analysis(rankings_df: pd.DataFrame, save_dir: str):
    """Create analysis comparing model performance on rough vs non-rough datasets."""
    print("   🌊 Creating rough vs non-rough dataset analysis...")
    
    # Define rough vs non-rough datasets
    rough_datasets = ['rbergomi', 'fbm_h03', 'fbm_h04']  # H < 0.5 or inherently rough
    nonrough_datasets = ['ou_process', 'heston', 'brownian', 'fbm_h06', 'fbm_h07']  # H >= 0.5 or smooth
    
    # Filter data for each category
    rough_data = rankings_df[rankings_df['dataset'].isin(rough_datasets)]
    nonrough_data = rankings_df[rankings_df['dataset'].isin(nonrough_datasets)]
    
    if rough_data.empty or nonrough_data.empty:
        print("   ⚠️ Insufficient data for rough vs non-rough analysis")
        return
    
    # Compute average rankings for each category
    rough_rankings = compute_category_rankings(rough_data, "Rough Processes")
    nonrough_rankings = compute_category_rankings(nonrough_data, "Non-Rough Processes")
    
    # Create comparison visualizations
    create_rough_nonrough_comparison_plots(rough_rankings, nonrough_rankings, save_dir)
    
    # Save detailed results
    rough_rankings.to_csv(os.path.join(save_dir, 'rough_datasets_rankings.csv'), index=False)
    nonrough_rankings.to_csv(os.path.join(save_dir, 'nonrough_datasets_rankings.csv'), index=False)
    
    print(f"   ✅ Rough vs non-rough analysis saved")


def compute_category_rankings(data: pd.DataFrame, category_name: str) -> pd.DataFrame:
    """Compute average rankings for a category of datasets."""
    # Group by model and compute average metrics
    model_stats = data.groupby('model_id').agg({
        'weighted_average_rank': 'mean',
        'ks_statistic': 'mean',
        'wasserstein_distance': 'mean',
        'rmse': 'mean',
        'std_rmse': 'mean',
        'dataset': 'count'  # Count how many datasets each model was evaluated on
    }).reset_index()
    
    # Rename count column
    model_stats.rename(columns={'dataset': 'num_datasets'}, inplace=True)
    
    # Compute distributional score (lower is better)
    model_stats['distributional_score'] = (model_stats['ks_statistic'] + model_stats['wasserstein_distance']) / 2
    
    # Sort by weighted average rank
    model_stats = model_stats.sort_values('weighted_average_rank').reset_index(drop=True)
    model_stats['category_rank'] = range(1, len(model_stats) + 1)
    model_stats['category'] = category_name
    
    return model_stats


def create_rough_nonrough_comparison_plots(rough_rankings: pd.DataFrame, nonrough_rankings: pd.DataFrame, save_dir: str):
    """Create comparison plots for rough vs non-rough dataset performance."""
    print("   🎨 Creating rough vs non-rough comparison plots...")
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance: Rough vs Non-Rough Stochastic Processes', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Distributional Quality on Rough Processes
    ax1 = axes[0, 0]
    create_category_ranking_plot(ax1, rough_rankings, 'distributional_score', 
                                'Rough Processes: Distributional Quality',
                                'Distributional Score (Lower = Better)', 'Reds_r')
    
    # 2. Distributional Quality on Non-Rough Processes  
    ax2 = axes[0, 1]
    create_category_ranking_plot(ax2, nonrough_rankings, 'distributional_score',
                                'Non-Rough Processes: Distributional Quality', 
                                'Distributional Score (Lower = Better)', 'Blues_r')
    
    # 3. RMSE on Rough Processes
    ax3 = axes[1, 0]
    create_category_ranking_plot(ax3, rough_rankings, 'rmse',
                                'Rough Processes: RMSE Accuracy',
                                'RMSE (Lower = Better)', 'Oranges_r')
    
    # 4. RMSE on Non-Rough Processes
    ax4 = axes[1, 1]
    create_category_ranking_plot(ax4, nonrough_rankings, 'rmse',
                                'Non-Rough Processes: RMSE Accuracy',
                                'RMSE (Lower = Better)', 'Greens_r')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rough_vs_nonrough_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Rough vs non-rough comparison saved to: rough_vs_nonrough_analysis.png")
    
    # Create side-by-side ranking comparison
    create_side_by_side_ranking_comparison(rough_rankings, nonrough_rankings, save_dir)


def create_category_ranking_plot(ax, rankings_df: pd.DataFrame, metric: str, title: str, ylabel: str, colormap: str):
    """Create a ranking plot for a specific category and metric."""
    models = rankings_df['model_id'].tolist()
    values = rankings_df[metric].tolist()
    
    # Create gradient colors
    colors = plt.colormaps[colormap](np.linspace(0.3, 0.9, len(models)))
    
    bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1)
    
    ax.set_title(title, fontweight='bold', fontsize=12, pad=15)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels and rank numbers
    max_val = max(values)
    for i, (bar, value) in enumerate(zip(bars, values)):
        # Value above bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Rank number inside bar
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'#{i+1}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')


def create_side_by_side_ranking_comparison(rough_rankings: pd.DataFrame, nonrough_rankings: pd.DataFrame, save_dir: str):
    """Create side-by-side comparison of model rankings on rough vs non-rough datasets."""
    print("   📊 Creating side-by-side ranking comparison...")
    
    # Find models that appear in both categories
    rough_models = set(rough_rankings['model_id'])
    nonrough_models = set(nonrough_rankings['model_id'])
    common_models = rough_models.intersection(nonrough_models)
    
    if not common_models:
        print("   ⚠️ No common models found for side-by-side comparison")
        return
    
    common_models = sorted(common_models)
    
    # Create figure with single plot for clarity
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Focus on distributional quality (most important for stochastic processes)
    create_paired_category_plot(ax, rough_rankings, nonrough_rankings, common_models, 
                               'distributional_score', 'Rough vs Non-Rough Process Performance',
                               'Distributional Score (Lower = Better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rough_vs_nonrough_side_by_side.png'),
                dpi=300, bbox_inches='tight')
    print(f"   ✅ Side-by-side comparison saved to: rough_vs_nonrough_side_by_side.png")


def create_paired_category_plot(ax, rough_df: pd.DataFrame, nonrough_df: pd.DataFrame, 
                               models: list, metric: str, title: str, ylabel: str):
    """Create paired bar plot comparing rough vs non-rough performance."""
    rough_values = []
    nonrough_values = []
    
    for model in models:
        rough_val = rough_df[rough_df['model_id'] == model][metric].iloc[0]
        nonrough_val = nonrough_df[nonrough_df['model_id'] == model][metric].iloc[0]
        rough_values.append(rough_val)
        nonrough_values.append(nonrough_val)
    
    x = np.arange(len(models))
    width = 0.35
    
    # Calculate max value for layout
    all_values = rough_values + nonrough_values
    max_val = max(all_values)
    
    bars1 = ax.bar(x - width/2, rough_values, width, label='Rough Processes', 
                   color='coral', alpha=0.8, edgecolor='darkred')
    bars2 = ax.bar(x + width/2, nonrough_values, width, label='Non-Rough Processes',
                   color='skyblue', alpha=0.8, edgecolor='darkblue')
    
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Extend y-axis to provide space for labels
    ax.set_ylim(0, max_val * 1.15)
    
    # Add value labels with better positioning and formatting
    
    for bars, values, color in [(bars1, rough_values, 'darkred'), (bars2, nonrough_values, 'darkblue')]:
        for bar, value in zip(bars, values):
            # Position label above bar with more spacing
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max_val * 0.05,  # Increased spacing
                   f'{value:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color=color)  # Larger, bolder, colored text


def run_cross_dataset_analysis():
    """Run comprehensive cross-dataset analysis using existing evaluation results."""
    print("🚀 Cross-Dataset Model Performance Analysis")
    print("=" * 70)
    print("Processing results from enhanced_model_evaluation.py")
    
    # Load all dataset results
    combined_df = load_dataset_results()
    
    if combined_df is None:
        return
    
    # Compute rankings and statistics
    rankings_df, overall_df = compute_cross_dataset_rankings(combined_df)
    
    # Create output directory
    save_dir = 'results/cross_dataset_analysis'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed results
    rankings_path = os.path.join(save_dir, 'detailed_rankings.csv')
    overall_path = os.path.join(save_dir, 'overall_model_summary.csv')
    combined_path = os.path.join(save_dir, 'all_datasets_combined_results.csv')
    
    rankings_df.to_csv(rankings_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   Detailed rankings: {rankings_path}")
    print(f"   Overall summary: {overall_path}")
    print(f"   Combined raw data: {combined_path}")
    
    # Create comprehensive visualizations
    create_comprehensive_visualizations(rankings_df, overall_df, save_dir)
    
    # Print summary to console
    print_analysis_summary(overall_df, rankings_df)
    
    return overall_df, rankings_df


def print_analysis_summary(overall_df: pd.DataFrame, rankings_df: pd.DataFrame):
    """Print a comprehensive analysis summary to console."""
    print(f"\n{'='*70}")
    print("📊 CROSS-DATASET ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    datasets = rankings_df['dataset'].unique()
    print(f"Datasets analyzed: {', '.join(datasets)}")
    print(f"Total models evaluated: {len(overall_df)}")
    
    print(f"\n🏆 OVERALL MODEL RANKINGS (Distributional-Weighted):")
    print("-" * 70)
    print(f"{'Rank':<4} {'Model':<6} {'Weighted':<9} {'Unweighted':<11} {'KS Rank':<8} {'Wasserstein':<11} {'Best Dataset'}")
    print("-" * 70)
    
    for i, row in overall_df.iterrows():
        model_id = row['model_id']
        weighted_rank = row['weighted_average_rank_overall']
        unweighted_rank = row['unweighted_average_rank_overall']
        ks_rank = row['ks_rank_mean']
        wasserstein_rank = row['wasserstein_rank_mean']
        best_dataset = row['best_dataset']
        
        print(f"{i+1:2d}.  {model_id:<6} {weighted_rank:7.2f}   {unweighted_rank:9.2f}   "
              f"{ks_rank:6.1f}    {wasserstein_rank:9.1f}   {best_dataset}")
    
    print(f"\n📈 KEY INSIGHTS (Stochastic Process Focus):")
    print("-" * 50)
    
    # Champion model
    champion = overall_df.iloc[0]
    print(f"🥇 Distributional Champion: {champion['model_id']}")
    print(f"   Weighted rank: {champion['weighted_average_rank_overall']:.2f} (vs unweighted: {champion['unweighted_average_rank_overall']:.2f})")
    print(f"   KS statistic rank: {champion['ks_rank_mean']:.1f} (avg: {champion['ks_statistic_mean']:.4f})")
    print(f"   Wasserstein rank: {champion['wasserstein_rank_mean']:.1f} (avg: {champion['wasserstein_distance_mean']:.4f})")
    print(f"   Best dataset: {champion['best_dataset']}")
    
    # Best pure distributional model
    best_distributional = overall_df.loc[overall_df['distributional_score'].idxmin()]
    if best_distributional['model_id'] != champion['model_id']:
        print(f"🎯 Pure Distributional Leader: {best_distributional['model_id']}")
        print(f"   Distributional score: {best_distributional['distributional_score']:.2f}")
        print(f"   (Focuses only on KS + Wasserstein metrics)")
    
    # Most consistent model
    consistency_scores = []
    for model in overall_df['model_id']:
        model_data = rankings_df[rankings_df['model_id'] == model]
        if len(model_data) > 1:
            rank_std = model_data['weighted_average_rank'].std()
            consistency_scores.append((model, rank_std))
    
    if consistency_scores:
        most_consistent = min(consistency_scores, key=lambda x: x[1])
        print(f"🎯 Most Consistent: {most_consistent[0]} (weighted rank std: {most_consistent[1]:.2f})")
    
    # Dataset-specific insights
    print(f"\n📊 DATASET-SPECIFIC CHAMPIONS (Distributional Focus):")
    print("-" * 60)
    for dataset in datasets:
        dataset_data = rankings_df[rankings_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            best_model = dataset_data.loc[dataset_data['weighted_average_rank'].idxmin()]
            print(f"{dataset.upper():12s}: {best_model['model_id']} "
                  f"(weighted rank: {best_model['weighted_average_rank']:.2f}, "
                  f"KS: {best_model['ks_statistic']:.3f})")
    
    print(f"\n✅ Complete analysis saved to: results/cross_dataset_analysis/")
    print(f"   View comprehensive_cross_dataset_analysis.png for detailed visualizations")


def main():
    """Main function - run cross-dataset analysis."""
    run_cross_dataset_analysis()


if __name__ == "__main__":
    main()

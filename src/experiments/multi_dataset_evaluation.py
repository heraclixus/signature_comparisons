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
    """Load evaluation results from all datasets."""
    print("📂 Loading evaluation results from all datasets...")
    
    datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
    all_results = []
    
    for dataset_name in datasets:
        results_path = f'results/{dataset_name}/evaluation/enhanced_models_evaluation.csv'
        
        if os.path.exists(results_path):
            print(f"   ✅ Loading {dataset_name} results...")
            df = pd.read_csv(results_path)
            df['dataset'] = dataset_name
            all_results.append(df)
        else:
            print(f"   ⚠️ No results found for {dataset_name} at {results_path}")
    
    if not all_results:
        print("❌ No evaluation results found!")
        print("   Run enhanced_model_evaluation.py first to generate results")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    print(f"✅ Loaded results for {len(all_results)} datasets, {len(combined_df)} total evaluations")
    
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
    
    # Create figure with 1x2 layout for clean comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Overall title
    fig.suptitle('Cross-Dataset Model Performance: Distributional vs Point-wise Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Distributional Quality Ranking (Primary Focus)
    ax1 = axes[0]
    # Sort models by distributional score for this plot
    dist_sorted_df = overall_df.sort_values('distributional_score').reset_index(drop=True)
    dist_sorted_models = dist_sorted_df['model_id'].tolist()
    dist_values = dist_sorted_df['distributional_score'].tolist()
    
    # Create gradient colors from best (dark green) to worst (light green)
    colors = plt.cm.Greens_r(np.linspace(0.3, 0.9, len(dist_sorted_models)))
    
    bars1 = ax1.bar(range(len(dist_sorted_models)), dist_values, 
                    color=colors, edgecolor='darkgreen', linewidth=1.5)
    ax1.set_title('Distributional Quality Ranking\n(KS Statistic + Wasserstein Distance)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('Distributional Score (Lower = Better)', fontsize=12)
    ax1.set_xticks(range(len(dist_sorted_models)))
    ax1.set_xticklabels(dist_sorted_models, rotation=0, fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars with better spacing
    for i, (bar, value) in enumerate(zip(bars1, dist_values)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dist_values) * 0.03,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add rank number inside bars
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # 2. RMSE Accuracy Ranking (Secondary Focus)  
    ax2 = axes[1]
    # Sort models by RMSE for this plot
    rmse_sorted_df = overall_df.sort_values('rmse_rank_mean').reset_index(drop=True)
    rmse_sorted_models = rmse_sorted_df['model_id'].tolist()
    rmse_values = rmse_sorted_df['rmse_rank_mean'].tolist()
    
    # Create gradient colors from best (dark red) to worst (light red)
    colors_rmse = plt.cm.Reds_r(np.linspace(0.3, 0.9, len(rmse_sorted_models)))
    
    bars2 = ax2.bar(range(len(rmse_sorted_models)), rmse_values, 
                    color=colors_rmse, edgecolor='darkred', linewidth=1.5)
    ax2.set_title('Point-wise Accuracy Ranking\n(RMSE Trajectory Matching)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('Average RMSE Rank (Lower = Better)', fontsize=12)
    ax2.set_xticks(range(len(rmse_sorted_models)))
    ax2.set_xticklabels(rmse_sorted_models, rotation=0, fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels above bars with better spacing
    for i, (bar, value) in enumerate(zip(bars2, rmse_values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.03,
                f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add rank number inside bars
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.5,
                f'#{i+1}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_cross_dataset_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive analysis saved to: {save_dir}/comprehensive_cross_dataset_analysis.png")


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

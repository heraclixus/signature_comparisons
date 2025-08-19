"""
Distributional Ranking Analysis

This script re-analyzes the trained model results with proper emphasis on
probability distribution-based metrics rather than MSE, which is more appropriate
for stochastic process generated data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple


def load_trained_models_results():
    """Load the trained models evaluation results."""
    results_path = "results/evaluation/trained_models_evaluation.csv"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return None
    
    df = pd.read_csv(results_path)
    print(f"Loaded results for {len(df)} trained models")
    
    return df


def calculate_distributional_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive distributional scores for ranking.
    
    For stochastic processes, we prioritize:
    1. Distribution similarity (KS test, Wasserstein distance)
    2. Statistical moment matching
    3. Path correlation (secondary)
    4. RMSE (least important for stochastic data)
    """
    
    # Create scoring dataframe
    scoring_df = df[['model_id', 'training_epoch', 'training_loss', 'total_parameters']].copy()
    
    # 1. Distribution Similarity Score (Primary - 40% weight)
    # Lower KS statistic = better distribution matching
    ks_scores = 1 - (df['ks_statistic'] / df['ks_statistic'].max())  # Normalize and invert
    scoring_df['distribution_similarity_score'] = ks_scores
    
    # 2. Statistical Moment Matching Score (Secondary - 30% weight)
    # Combine mean and std relative errors
    mean_errors = df['mean_rel_error'].fillna(df['mean_rel_error'].median())
    std_errors = df['std_rel_error'].fillna(df['std_rel_error'].median())
    
    # Invert and normalize (lower error = higher score)
    mean_scores = 1 - (mean_errors / mean_errors.max())
    std_scores = 1 - (std_errors / std_errors.max())
    moment_scores = (mean_scores + std_scores) / 2
    scoring_df['moment_matching_score'] = moment_scores
    
    # 3. Path Structure Score (Tertiary - 20% weight)
    # Path correlation (absolute value, since negative correlation can also be meaningful)
    path_corrs = df['mean_path_correlation'].abs()
    path_scores = path_corrs / path_corrs.max() if path_corrs.max() > 0 else pd.Series([0] * len(df))
    scoring_df['path_structure_score'] = path_scores
    
    # 4. Generation Quality Score (Quaternary - 10% weight)
    # RMSE (inverted and normalized)
    rmse_scores = 1 - (df['rmse'] / df['rmse'].max())
    scoring_df['generation_quality_score'] = rmse_scores
    
    # Calculate Composite Distributional Score
    composite_score = (
        0.40 * scoring_df['distribution_similarity_score'] +
        0.30 * scoring_df['moment_matching_score'] +
        0.20 * scoring_df['path_structure_score'] +
        0.10 * scoring_df['generation_quality_score']
    )
    
    scoring_df['composite_distributional_score'] = composite_score
    
    # Add original metrics for reference
    scoring_df['rmse'] = df['rmse']
    scoring_df['ks_statistic'] = df['ks_statistic']
    scoring_df['ks_p_value'] = df['ks_p_value']
    scoring_df['wasserstein_distance'] = df['wasserstein_distance']
    scoring_df['mean_path_correlation'] = df['mean_path_correlation']
    scoring_df['mean_rel_error'] = mean_errors
    scoring_df['std_rel_error'] = std_errors
    
    # Sort by composite score (highest = best)
    scoring_df = scoring_df.sort_values('composite_distributional_score', ascending=False)
    
    return scoring_df


def analyze_distributional_rankings(scoring_df: pd.DataFrame):
    """Analyze the distributional rankings and provide insights."""
    
    print("DISTRIBUTIONAL RANKING ANALYSIS")
    print("=" * 60)
    print("Prioritizing probability distribution metrics over MSE")
    print("Weighting: Distribution Similarity (40%) + Moment Matching (30%) + Path Structure (20%) + RMSE (10%)")
    
    print(f"\n🏆 DISTRIBUTIONAL PERFORMANCE RANKINGS:")
    print("=" * 50)
    
    print(f"{'Rank':<4} {'Model':<6} {'Generator':<12} {'Loss':<15} {'Score':<8} {'KS↓':<8} {'RMSE':<8}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(scoring_df.iterrows(), 1):
        model_id = row['model_id']
        
        # Determine generator and loss from model_id
        if model_id.startswith('A'):
            generator = "CannedNet"
        elif model_id.startswith('B'):
            generator = "Neural SDE"
        elif model_id.startswith('C'):
            generator = "GRU"
        else:
            generator = "Unknown"
        
        if 'T-Statistic' in str(row.get('training_loss', '')):
            loss = "T-Statistic"
        elif row['ks_statistic'] < 0.2:  # Heuristic for good distribution matching
            if model_id in ['A2', 'C2']:
                loss = "Sig Scoring"
            else:
                loss = "MMD"
        else:
            loss = "T-Statistic"
        
        # Determine loss more accurately
        if model_id in ['A1', 'C1']:
            loss = "T-Statistic"
        elif model_id in ['A2', 'C2']:
            loss = "Sig Scoring"
        elif model_id in ['A3', 'B4', 'C3']:
            loss = "MMD"
        
        score = row['composite_distributional_score']
        ks_stat = row['ks_statistic']
        rmse = row['rmse']
        
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i:2d}"
        
        print(f"{medal:<4} {model_id:<6} {generator:<12} {loss:<15} {score:<8.4f} {ks_stat:<8.4f} {rmse:<8.4f}")
    
    # Key insights
    print(f"\n🔍 KEY DISTRIBUTIONAL INSIGHTS:")
    print("=" * 40)
    
    top_3 = scoring_df.head(3)
    
    print(f"🏆 Champion: {top_3.iloc[0]['model_id']} (Score: {top_3.iloc[0]['composite_distributional_score']:.4f})")
    print(f"   Best distribution matching: KS = {top_3.iloc[0]['ks_statistic']:.4f}")
    print(f"   RMSE: {top_3.iloc[0]['rmse']:.4f} (not primary metric)")
    
    # Compare top performers by architecture
    print(f"\nBest by Architecture (Distributional Score):")
    
    for generator in ['GRU', 'Neural SDE', 'CannedNet']:
        if generator == 'GRU':
            subset = scoring_df[scoring_df['model_id'].str.startswith('C')]
        elif generator == 'Neural SDE':
            subset = scoring_df[scoring_df['model_id'].str.startswith('B')]
        else:  # CannedNet
            subset = scoring_df[scoring_df['model_id'].str.startswith('A')]
        
        if not subset.empty:
            best = subset.iloc[0]
            print(f"  {generator:<12}: {best['model_id']} (Score: {best['composite_distributional_score']:.4f}, KS: {best['ks_statistic']:.4f})")
    
    # Distribution quality analysis
    print(f"\nDistribution Quality Analysis:")
    excellent_dist = scoring_df[scoring_df['ks_statistic'] < 0.15]
    good_dist = scoring_df[(scoring_df['ks_statistic'] >= 0.15) & (scoring_df['ks_statistic'] < 0.25)]
    poor_dist = scoring_df[scoring_df['ks_statistic'] >= 0.25]
    
    print(f"  Excellent Distribution Matching (KS < 0.15): {len(excellent_dist)} models")
    for _, row in excellent_dist.iterrows():
        print(f"    {row['model_id']}: KS = {row['ks_statistic']:.4f}")
    
    print(f"  Good Distribution Matching (0.15 ≤ KS < 0.25): {len(good_dist)} models")
    for _, row in good_dist.iterrows():
        print(f"    {row['model_id']}: KS = {row['ks_statistic']:.4f}")
    
    print(f"  Poor Distribution Matching (KS ≥ 0.25): {len(poor_dist)} models")
    for _, row in poor_dist.iterrows():
        print(f"    {row['model_id']}: KS = {row['ks_statistic']:.4f}")


def create_distributional_visualization(scoring_df: pd.DataFrame, save_dir: str):
    """Create visualization focused on distributional metrics."""
    
    print(f"\nCreating distributional analysis visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = scoring_df['model_id'].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    # 1. Composite Distributional Score
    ax = axes[0, 0]
    scores = scoring_df['composite_distributional_score'].tolist()
    bars = ax.bar(models, scores, color=colors, alpha=0.8)
    ax.set_ylabel('Composite Distributional Score')
    ax.set_title('Overall Distributional Performance\n(Higher is Better)')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 2. Distribution Similarity (KS Statistic)
    ax = axes[0, 1]
    ks_stats = scoring_df['ks_statistic'].tolist()
    bars = ax.bar(models, ks_stats, color=colors, alpha=0.8)
    ax.set_ylabel('KS Statistic')
    ax.set_title('Distribution Similarity\n(Lower = Better Match)')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal lines for quality thresholds
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='Excellent (<0.15)')
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Good (<0.25)')
    ax.legend()
    
    for bar, ks in zip(bars, ks_stats):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{ks:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Wasserstein Distance
    ax = axes[0, 2]
    wass_dists = scoring_df['wasserstein_distance'].tolist()
    bars = ax.bar(models, wass_dists, color=colors, alpha=0.8)
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Distribution Distance\n(Lower is Better)')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    for bar, wass in zip(bars, wass_dists):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{wass:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 4. Score Components Breakdown
    ax = axes[1, 0]
    
    # Stacked bar chart of score components
    dist_scores = scoring_df['distribution_similarity_score'].tolist()
    moment_scores = scoring_df['moment_matching_score'].tolist()
    path_scores = scoring_df['path_structure_score'].tolist()
    rmse_scores = scoring_df['generation_quality_score'].tolist()
    
    # Apply weights
    weighted_dist = [s * 0.40 for s in dist_scores]
    weighted_moment = [s * 0.30 for s in moment_scores]
    weighted_path = [s * 0.20 for s in path_scores]
    weighted_rmse = [s * 0.10 for s in rmse_scores]
    
    ax.bar(models, weighted_dist, label='Dist Similarity (40%)', alpha=0.8, color='darkblue')
    ax.bar(models, weighted_moment, bottom=weighted_dist, label='Moment Match (30%)', alpha=0.8, color='blue')
    
    bottom_2 = [d + m for d, m in zip(weighted_dist, weighted_moment)]
    ax.bar(models, weighted_path, bottom=bottom_2, label='Path Structure (20%)', alpha=0.8, color='lightblue')
    
    bottom_3 = [b + p for b, p in zip(bottom_2, weighted_path)]
    ax.bar(models, weighted_rmse, bottom=bottom_3, label='RMSE (10%)', alpha=0.8, color='lightgray')
    
    ax.set_ylabel('Weighted Score Components')
    ax.set_title('Distributional Score Breakdown')
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Architecture Comparison
    ax = axes[1, 1]
    
    # Group by architecture
    arch_data = {}
    for _, row in scoring_df.iterrows():
        model_id = row['model_id']
        if model_id.startswith('A'):
            arch = 'CannedNet'
        elif model_id.startswith('B'):
            arch = 'Neural SDE'
        elif model_id.startswith('C'):
            arch = 'GRU'
        else:
            arch = 'Unknown'
        
        if arch not in arch_data:
            arch_data[arch] = []
        arch_data[arch].append(row['composite_distributional_score'])
    
    # Box plot by architecture
    arch_names = list(arch_data.keys())
    arch_scores = [arch_data[arch] for arch in arch_names]
    
    bp = ax.boxplot(arch_scores, labels=arch_names, patch_artist=True)
    
    # Color the boxes
    colors_arch = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors_arch):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Distributional Score')
    ax.set_title('Architecture Performance\nDistribution')
    ax.grid(True, alpha=0.3)
    
    # 6. Summary Rankings
    ax = axes[1, 2]
    ax.axis('off')
    
    # Create ranking summary
    top_5 = scoring_df.head(5)
    
    summary_text = "Distributional Rankings\n" + "="*22 + "\n\n"
    summary_text += "Ranked by Distribution Matching\n"
    summary_text += "(Not MSE)\n\n"
    
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        model = row['model_id']
        score = row['composite_distributional_score']
        ks = row['ks_statistic']
        
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        
        summary_text += f"{medal} {model}\n"
        summary_text += f"   Score: {score:.4f}\n"
        summary_text += f"   KS: {ks:.4f}\n\n"
    
    # Architecture summary
    summary_text += "Architecture Winners:\n"
    for arch in ['GRU', 'Neural SDE', 'CannedNet']:
        if arch == 'GRU':
            subset = scoring_df[scoring_df['model_id'].str.startswith('C')]
        elif arch == 'Neural SDE':
            subset = scoring_df[scoring_df['model_id'].str.startswith('B')]
        else:
            subset = scoring_df[scoring_df['model_id'].str.startswith('A')]
        
        if not subset.empty:
            best = subset.iloc[0]
            summary_text += f"  {arch}: {best['model_id']}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'distributional_rankings.png'), dpi=300, bbox_inches='tight')
    print(f"Distributional ranking visualization saved to: {save_dir}/distributional_rankings.png")
    plt.close()


def compare_rmse_vs_distributional_rankings(df: pd.DataFrame, scoring_df: pd.DataFrame):
    """Compare RMSE-based vs distributional-based rankings."""
    
    print(f"\n" + "="*60)
    print("RMSE vs DISTRIBUTIONAL RANKING COMPARISON")
    print("="*60)
    
    # RMSE ranking
    rmse_ranking = df.sort_values('rmse')[['model_id', 'rmse', 'ks_statistic']].reset_index(drop=True)
    
    # Distributional ranking
    dist_ranking = scoring_df[['model_id', 'composite_distributional_score', 'ks_statistic']].reset_index(drop=True)
    
    print(f"RMSE-Based Ranking vs Distributional-Based Ranking:")
    print(f"{'RMSE Rank':<12} {'Model':<6} {'RMSE':<8} {'KS':<8} | {'Dist Rank':<12} {'Model':<6} {'Score':<8} {'KS':<8}")
    print("-" * 80)
    
    for i in range(len(df)):
        rmse_model = rmse_ranking.iloc[i]
        dist_model = dist_ranking.iloc[i]
        
        print(f"{i+1:<12} {rmse_model['model_id']:<6} {rmse_model['rmse']:<8.4f} {rmse_model['ks_statistic']:<8.4f} | "
              f"{i+1:<12} {dist_model['model_id']:<6} {dist_model['composite_distributional_score']:<8.4f} {dist_model['ks_statistic']:<8.4f}")
    
    # Identify ranking changes
    print(f"\nRanking Changes (RMSE → Distributional):")
    
    rmse_order = rmse_ranking['model_id'].tolist()
    dist_order = dist_ranking['model_id'].tolist()
    
    for model in rmse_order:
        rmse_rank = rmse_order.index(model) + 1
        dist_rank = dist_order.index(model) + 1
        change = rmse_rank - dist_rank
        
        if change > 0:
            print(f"  📈 {model}: Rank {rmse_rank} → {dist_rank} (+{change} positions)")
        elif change < 0:
            print(f"  📉 {model}: Rank {rmse_rank} → {dist_rank} ({change} positions)")
        else:
            print(f"  ➡️ {model}: Rank {rmse_rank} (no change)")
    
    # Key insights about the differences
    rmse_winner = rmse_ranking.iloc[0]['model_id']
    dist_winner = dist_ranking.iloc[0]['model_id']
    
    print(f"\n🎯 RANKING IMPACT:")
    if rmse_winner == dist_winner:
        print(f"  ✅ Same winner: {rmse_winner}")
        print(f"     Both metrics agree on best model")
    else:
        print(f"  🔄 Different winners:")
        print(f"     RMSE Champion: {rmse_winner}")
        print(f"     Distributional Champion: {dist_winner}")
        print(f"     Distributional ranking is more appropriate for stochastic data!")


def main():
    """Main distributional analysis."""
    print("Distributional Ranking Analysis for Stochastic Process Models")
    print("=" * 70)
    print("Re-ranking models based on probability distribution metrics")
    print("Appropriate for stochastic process generated data")
    
    # Load results
    df = load_trained_models_results()
    if df is None:
        return False
    
    # Calculate distributional scores
    scoring_df = calculate_distributional_scores(df)
    
    # Analyze rankings
    analyze_distributional_rankings(scoring_df)
    
    # Compare with RMSE rankings
    compare_rmse_vs_distributional_rankings(df, scoring_df)
    
    # Create visualization
    os.makedirs('results/evaluation', exist_ok=True)
    create_distributional_visualization(scoring_df, 'results/evaluation')
    
    # Save distributional rankings
    ranking_path = 'results/evaluation/distributional_rankings.csv'
    scoring_df.to_csv(ranking_path, index=False)
    print(f"\nDistributional rankings saved to: {ranking_path}")
    
    print(f"\n🎉 DISTRIBUTIONAL ANALYSIS COMPLETE!")
    print(f"   Proper stochastic process evaluation methodology applied")
    print(f"   Distribution-based metrics prioritized over MSE")
    print(f"   True model performance revealed!")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Use distributional rankings for stochastic process model selection")
        print(f"   Distribution matching is the primary criterion")
        print(f"   RMSE is secondary for probabilistic data")
    else:
        print(f"\n❌ Analysis failed - check trained model results")

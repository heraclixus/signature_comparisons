"""
Quick Runtime Summary

Generate a quick runtime analysis summary from existing training data.
Perfect for getting immediate insights after training sessions.
"""

import pandas as pd
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def analyze_training_runtime(results_dir: str = "results") -> pd.DataFrame:
    """Quick analysis of training runtime from existing data."""
    runtime_data = []
    
    datasets = ['ou_process', 'heston', 'rbergomi', 'brownian']
    
    for dataset_name in datasets:
        dataset_dir = os.path.join(results_dir, dataset_name, 'trained_models')
        if not os.path.exists(dataset_dir):
            continue
        
        for model_dir in os.listdir(dataset_dir):
            model_path = os.path.join(dataset_dir, model_dir)
            if not os.path.isdir(model_path):
                continue
            
            history_file = os.path.join(model_path, 'training_history.csv')
            checkpoint_file = os.path.join(model_path, 'checkpoint_info.json')
            
            if os.path.exists(history_file) and os.path.exists(checkpoint_file):
                try:
                    # Load data
                    history_df = pd.read_csv(history_file)
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_info = json.load(f)
                    
                    # Calculate metrics
                    total_time = history_df['cumulative_time'].iloc[-1] if 'cumulative_time' in history_df.columns else 0
                    avg_epoch_time = history_df['epoch_time'].mean() if 'epoch_time' in history_df.columns else 0
                    total_epochs = len(history_df)
                    best_loss = history_df['loss'].min()
                    final_loss = history_df['loss'].iloc[-1]
                    
                    # Architecture classification
                    arch_type = "CannedNet" if model_dir.startswith('A') else "Neural SDE"
                    
                    runtime_data.append({
                        'dataset': dataset_name,
                        'model_id': model_dir,
                        'architecture': arch_type,
                        'parameters': checkpoint_info.get('total_parameters', 0),
                        'epochs': total_epochs,
                        'total_time': total_time,
                        'avg_epoch_time': avg_epoch_time,
                        'best_loss': best_loss,
                        'final_loss': final_loss,
                        'epochs_per_minute': 60 / avg_epoch_time if avg_epoch_time > 0 else 0,
                        'training_date': checkpoint_info.get('timestamp', 'Unknown')[:10]
                    })
                    
                except Exception as e:
                    continue
    
    return pd.DataFrame(runtime_data)


def print_runtime_summary(df: pd.DataFrame):
    """Print a concise runtime summary."""
    if df.empty:
        print("❌ No training data found!")
        return
    
    print("🚀 TRAINING RUNTIME SUMMARY")
    print("=" * 50)
    
    # Overall stats
    total_models = len(df)
    total_datasets = df['dataset'].nunique()
    total_training_time = df['total_time'].sum()
    
    print(f"📊 Overview:")
    print(f"   • {total_models} trained models across {total_datasets} datasets")
    print(f"   • Total training time: {total_training_time/3600:.1f} hours")
    print(f"   • Average training time per model: {total_training_time/total_models/60:.1f} minutes")
    
    # Speed rankings
    print(f"\n⚡ Speed Rankings (Fastest to Slowest):")
    speed_ranking = df.groupby('model_id')['avg_epoch_time'].mean().sort_values()
    for i, (model, time_val) in enumerate(speed_ranking.head(5).items(), 1):
        print(f"   {i}. {model}: {time_val:.2f}s per epoch")
    
    # Architecture comparison
    arch_comparison = df.groupby('architecture')['avg_epoch_time'].mean().sort_values()
    print(f"\n🏗️ Architecture Performance:")
    for arch, avg_time in arch_comparison.items():
        count = len(df[df['architecture'] == arch])
        print(f"   • {arch}: {avg_time:.2f}s avg epoch time ({count} models)")
    
    # Dataset-specific performance
    print(f"\n📊 Dataset Performance:")
    dataset_perf = df.groupby('dataset').agg({
        'avg_epoch_time': 'mean',
        'model_id': 'count'
    }).round(2)
    
    for dataset, row in dataset_perf.iterrows():
        print(f"   • {dataset}: {row['avg_epoch_time']:.2f}s avg ({int(row['model_id'])} models)")
    
    # Recent training activity
    if 'training_date' in df.columns:
        recent_training = df.sort_values('training_date', ascending=False).head(3)
        print(f"\n📅 Recent Training Activity:")
        for _, row in recent_training.iterrows():
            print(f"   • {row['model_id']} on {row['dataset']}: {row['training_date']}")
    
    # Performance insights
    print(f"\n💡 Key Insights:")
    
    # Find fastest and slowest
    fastest = speed_ranking.index[0]
    slowest = speed_ranking.index[-1]
    speed_ratio = speed_ranking.iloc[-1] / speed_ranking.iloc[0]
    
    print(f"   • Fastest model: {fastest} ({speed_ranking.iloc[0]:.2f}s/epoch)")
    print(f"   • Slowest model: {slowest} ({speed_ranking.iloc[-1]:.2f}s/epoch)")
    print(f"   • Speed difference: {speed_ratio:.1f}x")
    
    # Parameter efficiency
    if 'parameters' in df.columns:
        df['param_efficiency'] = df['parameters'] / df['avg_epoch_time']
        most_efficient = df.loc[df['param_efficiency'].idxmax()]
        print(f"   • Most parameter-efficient: {most_efficient['model_id']} ({most_efficient['param_efficiency']:.0f} params/sec)")
    
    print(f"\n🔍 For detailed analysis, run: python src/experiments/runtime_analysis.py")


def main():
    """Quick runtime summary from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick training runtime summary")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory path")
    parser.add_argument("--csv", action="store_true",
                       help="Save summary to CSV file")
    
    args = parser.parse_args()
    
    # Analyze runtime
    df = analyze_training_runtime(args.results_dir)
    
    # Print summary
    print_runtime_summary(df)
    
    # Save CSV if requested
    if args.csv and not df.empty:
        output_path = os.path.join(args.results_dir, 'quick_runtime_summary.csv')
        df.to_csv(output_path, index=False)
        print(f"\n💾 Summary saved to: {output_path}")


if __name__ == "__main__":
    main()

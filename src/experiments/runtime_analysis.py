"""
Runtime Analysis for Signature-Based Models

This module analyzes training speed, memory usage, and computational efficiency
across different model architectures and datasets.
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
import sys
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.multi_dataset import MultiDatasetManager
from utils.model_checkpoint import create_checkpoint_manager


class RuntimeProfiler:
    """Profile training runtime and resource usage."""
    
    def __init__(self):
        self.profiles = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict:
        """Get system specifications."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__
        }
    
    def profile_model_creation(self, model_id: str, create_fn, train_data: torch.Tensor) -> Dict:
        """Profile model creation time and memory usage."""
        # Memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Time model creation
        start_time = time.time()
        model = create_fn(train_data, train_data)
        creation_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_delta = memory_after - memory_before
        
        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model_id': model_id,
            'creation_time': creation_time,
            'memory_delta': memory_delta,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameters_per_second': total_params / creation_time if creation_time > 0 else 0
        }
    
    def profile_training_epoch(self, model, train_data: torch.Tensor, batch_size: int = 16) -> Dict:
        """Profile a single training epoch."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # Time single epoch
        start_time = time.time()
        
        # Get random batch
        indices = torch.randperm(train_data.shape[0])[:batch_size]
        batch_data = train_data[indices]
        
        # Forward pass
        optimizer.zero_grad()
        forward_start = time.time()
        generated_output = model(batch_data)
        forward_time = time.time() - forward_start
        
        # Loss computation
        loss_start = time.time()
        loss = model.compute_loss(generated_output)
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start
        
        total_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_delta = memory_after - memory_before
        
        # Clean up
        del generated_output, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'total_time': total_time,
            'forward_time': forward_time,
            'loss_time': loss_time,
            'backward_time': backward_time,
            'memory_delta': memory_delta,
            'loss_value': loss.item() if 'loss' in locals() else 0.0,
            'samples_per_second': batch_size / total_time if total_time > 0 else 0,
            'forward_fraction': forward_time / total_time if total_time > 0 else 0,
            'loss_fraction': loss_time / total_time if total_time > 0 else 0,
            'backward_fraction': backward_time / total_time if total_time > 0 else 0
        }


class RuntimeAnalyzer:
    """Analyze training runtime across models and datasets."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.profiler = RuntimeProfiler()
        self.dataset_manager = MultiDatasetManager()
        
    def analyze_existing_training_history(self) -> pd.DataFrame:
        """Analyze runtime from existing training history files."""
        runtime_data = []
        
        datasets = ['ou_process', 'heston', 'rbergomi', 'brownian']
        
        for dataset_name in datasets:
            dataset_dir = os.path.join(self.results_dir, dataset_name, 'trained_models')
            if not os.path.exists(dataset_dir):
                continue
            
            for model_dir in os.listdir(dataset_dir):
                model_path = os.path.join(dataset_dir, model_dir)
                if not os.path.isdir(model_path):
                    continue
                
                # Read training history
                history_file = os.path.join(model_path, 'training_history.csv')
                checkpoint_file = os.path.join(model_path, 'checkpoint_info.json')
                
                if os.path.exists(history_file) and os.path.exists(checkpoint_file):
                    try:
                        # Load training history
                        history_df = pd.read_csv(history_file)
                        
                        # Load checkpoint info
                        with open(checkpoint_file, 'r') as f:
                            checkpoint_info = json.load(f)
                        
                        # Calculate runtime metrics
                        total_epochs = len(history_df)
                        total_time = history_df['cumulative_time'].iloc[-1] if 'cumulative_time' in history_df.columns else 0
                        avg_epoch_time = history_df['epoch_time'].mean() if 'epoch_time' in history_df.columns else 0
                        final_loss = history_df['loss'].iloc[-1]
                        best_loss = history_df['loss'].min()
                        convergence_epoch = history_df['loss'].idxmin() + 1
                        
                        # Time to convergence
                        if 'cumulative_time' in history_df.columns:
                            time_to_convergence = history_df['cumulative_time'].iloc[convergence_epoch - 1]
                        else:
                            time_to_convergence = avg_epoch_time * convergence_epoch
                        
                        runtime_data.append({
                            'dataset': dataset_name,
                            'model_id': model_dir,
                            'total_epochs': total_epochs,
                            'total_time': total_time,
                            'avg_epoch_time': avg_epoch_time,
                            'time_to_convergence': time_to_convergence,
                            'convergence_epoch': convergence_epoch,
                            'final_loss': final_loss,
                            'best_loss': best_loss,
                            'total_parameters': checkpoint_info.get('total_parameters', 0),
                            'trainable_parameters': checkpoint_info.get('trainable_parameters', 0),
                            'epochs_per_minute': 60 / avg_epoch_time if avg_epoch_time > 0 else 0,
                            'parameters_per_second': checkpoint_info.get('total_parameters', 0) / total_time if total_time > 0 else 0,
                            'loss_improvement_rate': (history_df['loss'].iloc[0] - best_loss) / time_to_convergence if time_to_convergence > 0 else 0
                        })
                        
                    except Exception as e:
                        print(f"⚠️ Error processing {dataset_name}/{model_dir}: {e}")
                        continue
        
        return pd.DataFrame(runtime_data)
    
    def profile_model_architectures(self, dataset_name: str = 'ou_process', num_samples: int = 64) -> pd.DataFrame:
        """Profile model creation and single epoch performance."""
        # Get available models
        from models.implementations import get_all_model_creators
        
        # Generate test data
        if dataset_name == 'ou_process':
            from dataset import generative_model
            dataset = generative_model.get_signal(num_samples=num_samples)
            train_data = torch.stack([dataset[i][0] for i in range(num_samples)])
        else:
            dataset_data = self.dataset_manager.get_dataset(dataset_name, num_samples=num_samples)
            train_data = torch.stack([dataset_data[i][0] for i in range(num_samples)])
        
        profile_data = []
        model_creators = get_all_model_creators()
        
        for model_id, create_fn in model_creators.items():
            try:
                print(f"Profiling {model_id}...")
                
                # Profile model creation
                creation_profile = self.profiler.profile_model_creation(model_id, create_fn, train_data)
                
                # Create model for epoch profiling
                model = create_fn(train_data, train_data)
                
                # Profile training epoch (multiple runs for stability)
                epoch_profiles = []
                for run in range(3):  # Average over 3 runs
                    try:
                        batch_size = 8 if model_id.startswith('B') else 16  # Smaller batches for Neural SDE
                        epoch_profile = self.profiler.profile_training_epoch(model, train_data, batch_size)
                        epoch_profiles.append(epoch_profile)
                    except Exception as e:
                        print(f"  ⚠️ Epoch profiling failed for {model_id} run {run + 1}: {e}")
                        continue
                
                if epoch_profiles:
                    # Average epoch metrics
                    avg_epoch_profile = {}
                    for key in epoch_profiles[0].keys():
                        if isinstance(epoch_profiles[0][key], (int, float)):
                            avg_epoch_profile[key] = np.mean([p[key] for p in epoch_profiles])
                        else:
                            avg_epoch_profile[key] = epoch_profiles[0][key]
                    
                    # Combine profiles
                    combined_profile = {**creation_profile, **avg_epoch_profile}
                    combined_profile['dataset'] = dataset_name
                    combined_profile['num_samples'] = num_samples
                    combined_profile['successful_runs'] = len(epoch_profiles)
                    
                    profile_data.append(combined_profile)
                else:
                    print(f"  ❌ All epoch profiling runs failed for {model_id}")
                
            except Exception as e:
                print(f"❌ Failed to profile {model_id}: {e}")
                continue
        
        return pd.DataFrame(profile_data)
    
    def create_runtime_visualizations(self, runtime_df: pd.DataFrame, profile_df: pd.DataFrame = None):
        """Create comprehensive runtime analysis visualizations."""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Training Time Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Runtime Analysis', fontsize=16, fontweight='bold')
        
        # Average epoch time by model
        if not runtime_df.empty:
            model_times = runtime_df.groupby('model_id')['avg_epoch_time'].mean().sort_values()
            axes[0, 0].bar(range(len(model_times)), model_times.values)
            axes[0, 0].set_xticks(range(len(model_times)))
            axes[0, 0].set_xticklabels(model_times.index, rotation=45)
            axes[0, 0].set_ylabel('Average Epoch Time (s)')
            axes[0, 0].set_title('Average Training Speed by Model')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Time to convergence
            convergence_times = runtime_df.groupby('model_id')['time_to_convergence'].mean().sort_values()
            axes[0, 1].bar(range(len(convergence_times)), convergence_times.values, color='orange')
            axes[0, 1].set_xticks(range(len(convergence_times)))
            axes[0, 1].set_xticklabels(convergence_times.index, rotation=45)
            axes[0, 1].set_ylabel('Time to Convergence (s)')
            axes[0, 1].set_title('Time to Best Loss by Model')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Training efficiency (loss improvement per second)
            efficiency = runtime_df.groupby('model_id')['loss_improvement_rate'].mean().sort_values(ascending=False)
            axes[1, 0].bar(range(len(efficiency)), efficiency.values, color='green')
            axes[1, 0].set_xticks(range(len(efficiency)))
            axes[1, 0].set_xticklabels(efficiency.index, rotation=45)
            axes[1, 0].set_ylabel('Loss Improvement Rate (Δloss/s)')
            axes[1, 0].set_title('Training Efficiency by Model')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Parameters vs Speed scatter
            axes[1, 1].scatter(runtime_df['total_parameters'], runtime_df['avg_epoch_time'], 
                             alpha=0.7, s=60)
            for i, model in enumerate(runtime_df['model_id']):
                axes[1, 1].annotate(model, 
                                   (runtime_df['total_parameters'].iloc[i], 
                                    runtime_df['avg_epoch_time'].iloc[i]),
                                   xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1, 1].set_xlabel('Total Parameters')
            axes[1, 1].set_ylabel('Average Epoch Time (s)')
            axes[1, 1].set_title('Model Complexity vs Training Speed')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        runtime_dir = os.path.join(self.results_dir, 'runtime_analysis')
        os.makedirs(runtime_dir, exist_ok=True)
        plt.savefig(os.path.join(runtime_dir, 'training_runtime_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed Performance Breakdown (if profile data available)
        if profile_df is not None and not profile_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Detailed Performance Profiling', fontsize=16, fontweight='bold')
            
            # Training phase breakdown
            phases = ['forward_fraction', 'loss_fraction', 'backward_fraction']
            phase_data = profile_df[phases + ['model_id']].set_index('model_id')
            phase_data.plot(kind='bar', stacked=True, ax=axes[0, 0])
            axes[0, 0].set_title('Training Phase Time Distribution')
            axes[0, 0].set_ylabel('Fraction of Total Time')
            axes[0, 0].legend(['Forward', 'Loss', 'Backward'])
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Memory usage
            axes[0, 1].bar(range(len(profile_df)), profile_df['memory_delta'])
            axes[0, 1].set_xticks(range(len(profile_df)))
            axes[0, 1].set_xticklabels(profile_df['model_id'], rotation=45)
            axes[0, 1].set_ylabel('Memory Delta (MB)')
            axes[0, 1].set_title('Memory Usage per Training Step')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Samples per second
            throughput = profile_df.set_index('model_id')['samples_per_second'].sort_values(ascending=False)
            axes[1, 0].bar(range(len(throughput)), throughput.values, color='purple')
            axes[1, 0].set_xticks(range(len(throughput)))
            axes[1, 0].set_xticklabels(throughput.index, rotation=45)
            axes[1, 0].set_ylabel('Samples/Second')
            axes[1, 0].set_title('Training Throughput by Model')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Model creation time
            creation_times = profile_df.set_index('model_id')['creation_time'].sort_values()
            axes[1, 1].bar(range(len(creation_times)), creation_times.values, color='red')
            axes[1, 1].set_xticks(range(len(creation_times)))
            axes[1, 1].set_xticklabels(creation_times.index, rotation=45)
            axes[1, 1].set_ylabel('Creation Time (s)')
            axes[1, 1].set_title('Model Initialization Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(runtime_dir, 'detailed_performance_profiling.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"📊 Runtime analysis visualizations saved to {runtime_dir}/")
    
    def generate_runtime_report(self, runtime_df: pd.DataFrame, profile_df: pd.DataFrame = None) -> str:
        """Generate comprehensive runtime analysis report."""
        
        report_lines = [
            "# Training Runtime Analysis Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## System Information",
            f"- CPU: {self.profiler.system_info['cpu_count']} cores @ {self.profiler.system_info['cpu_freq']:.1f} MHz",
            f"- Memory: {self.profiler.system_info['memory_total']:.1f} GB",
            f"- CUDA: {'Available' if self.profiler.system_info['cuda_available'] else 'Not Available'}",
        ]
        
        if self.profiler.system_info['cuda_device']:
            report_lines.append(f"- GPU: {self.profiler.system_info['cuda_device']}")
        
        report_lines.extend([
            f"- PyTorch: {self.profiler.system_info['pytorch_version']}",
            "",
            "## Training Speed Analysis"
        ])
        
        if not runtime_df.empty:
            # Speed rankings
            speed_ranking = runtime_df.groupby('model_id')['avg_epoch_time'].mean().sort_values()
            report_lines.extend([
                "",
                "### Training Speed Ranking (Fastest to Slowest):",
                ""
            ])
            
            for i, (model, time_val) in enumerate(speed_ranking.items(), 1):
                report_lines.append(f"{i}. **{model}**: {time_val:.2f}s per epoch")
            
            # Efficiency analysis
            efficiency_ranking = runtime_df.groupby('model_id')['loss_improvement_rate'].mean().sort_values(ascending=False)
            report_lines.extend([
                "",
                "### Training Efficiency Ranking (Best Loss Improvement Rate):",
                ""
            ])
            
            for i, (model, eff_val) in enumerate(efficiency_ranking.items(), 1):
                if eff_val > 0:
                    report_lines.append(f"{i}. **{model}**: {eff_val:.4f} loss improvement per second")
            
            # Model complexity analysis
            complexity_analysis = runtime_df.groupby('model_id').agg({
                'total_parameters': 'mean',
                'avg_epoch_time': 'mean',
                'time_to_convergence': 'mean'
            }).round(2)
            
            report_lines.extend([
                "",
                "### Model Complexity vs Performance:",
                "",
                "| Model | Parameters | Avg Epoch Time | Time to Convergence |",
                "|-------|------------|----------------|---------------------|"
            ])
            
            for model, row in complexity_analysis.iterrows():
                report_lines.append(
                    f"| {model} | {row['total_parameters']:,} | {row['avg_epoch_time']:.2f}s | {row['time_to_convergence']:.1f}s |"
                )
        
        # Detailed profiling results
        if profile_df is not None and not profile_df.empty:
            report_lines.extend([
                "",
                "## Detailed Performance Profiling",
                "",
                "### Training Phase Breakdown:",
                ""
            ])
            
            for _, row in profile_df.iterrows():
                model_id = row['model_id']
                forward_pct = row['forward_fraction'] * 100
                loss_pct = row['loss_fraction'] * 100
                backward_pct = row['backward_fraction'] * 100
                
                report_lines.extend([
                    f"**{model_id}**:",
                    f"- Forward pass: {forward_pct:.1f}%",
                    f"- Loss computation: {loss_pct:.1f}%",
                    f"- Backward pass: {backward_pct:.1f}%",
                    f"- Throughput: {row['samples_per_second']:.1f} samples/sec",
                    f"- Memory usage: {row['memory_delta']:.1f} MB per step",
                    ""
                ])
        
        # Recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if not runtime_df.empty:
            fastest_model = speed_ranking.index[0]
            most_efficient = efficiency_ranking.index[0] if len(efficiency_ranking) > 0 else None
            
            report_lines.extend([
                f"### For Speed-Critical Applications:",
                f"- **Fastest Training**: {fastest_model} ({speed_ranking.iloc[0]:.2f}s per epoch)",
                ""
            ])
            
            if most_efficient and efficiency_ranking.iloc[0] > 0:
                report_lines.extend([
                    f"### For Efficiency-Critical Applications:",
                    f"- **Best Efficiency**: {most_efficient} ({efficiency_ranking.iloc[0]:.4f} improvement rate)",
                    ""
                ])
            
            # Architecture-specific recommendations
            canned_models = runtime_df[runtime_df['model_id'].str.startswith('A')]
            nsde_models = runtime_df[runtime_df['model_id'].str.startswith('B')]
            
            if not canned_models.empty and not nsde_models.empty:
                avg_canned_time = canned_models['avg_epoch_time'].mean()
                avg_nsde_time = nsde_models['avg_epoch_time'].mean()
                
                report_lines.extend([
                    "### Architecture Comparison:",
                    f"- **CannedNet models**: {avg_canned_time:.2f}s average epoch time",
                    f"- **Neural SDE models**: {avg_nsde_time:.2f}s average epoch time",
                    f"- **Speed ratio**: Neural SDE is {avg_nsde_time/avg_canned_time:.1f}x slower than CannedNet",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def run_comprehensive_analysis(self, include_profiling: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Run comprehensive runtime analysis."""
        print("🚀 Starting Comprehensive Runtime Analysis")
        print("=" * 60)
        
        # 1. Analyze existing training history
        print("📊 Analyzing existing training history...")
        runtime_df = self.analyze_existing_training_history()
        print(f"   Found {len(runtime_df)} model-dataset combinations")
        
        # 2. Profile model architectures (optional)
        profile_df = None
        if include_profiling and len(runtime_df) > 0:
            print("\n🔬 Profiling model architectures...")
            try:
                profile_df = self.profile_model_architectures('ou_process', num_samples=32)
                print(f"   Profiled {len(profile_df)} models")
            except Exception as e:
                print(f"⚠️ Profiling failed: {e}")
        
        # 3. Create visualizations
        print("\n📈 Creating runtime visualizations...")
        self.create_runtime_visualizations(runtime_df, profile_df)
        
        # 4. Generate report
        print("\n📝 Generating runtime analysis report...")
        report = self.generate_runtime_report(runtime_df, profile_df)
        
        # Save report
        runtime_dir = os.path.join(self.results_dir, 'runtime_analysis')
        os.makedirs(runtime_dir, exist_ok=True)
        report_path = os.path.join(runtime_dir, 'runtime_analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save data
        runtime_df.to_csv(os.path.join(runtime_dir, 'training_runtime_data.csv'), index=False)
        if profile_df is not None:
            profile_df.to_csv(os.path.join(runtime_dir, 'performance_profiling_data.csv'), index=False)
        
        print(f"\n✅ Runtime analysis complete!")
        print(f"📁 Results saved to: {runtime_dir}/")
        print(f"📊 Runtime data: training_runtime_data.csv")
        if profile_df is not None:
            print(f"🔬 Profiling data: performance_profiling_data.csv")
        print(f"📝 Full report: runtime_analysis_report.md")
        print(f"📈 Visualizations: *.png files")
        
        return runtime_df, profile_df


def main():
    """Run runtime analysis from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze training runtime and performance")
    parser.add_argument("--no-profiling", action="store_true", 
                       help="Skip detailed model profiling (faster)")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory path")
    
    args = parser.parse_args()
    
    analyzer = RuntimeAnalyzer(args.results_dir)
    runtime_df, profile_df = analyzer.run_comprehensive_analysis(
        include_profiling=not args.no_profiling
    )
    
    # Print summary
    if not runtime_df.empty:
        print(f"\n📊 RUNTIME ANALYSIS SUMMARY")
        print("=" * 40)
        
        speed_ranking = runtime_df.groupby('model_id')['avg_epoch_time'].mean().sort_values()
        print("🏃 Fastest Models:")
        for i, (model, time_val) in enumerate(speed_ranking.head(3).items(), 1):
            print(f"  {i}. {model}: {time_val:.2f}s per epoch")
        
        if 'loss_improvement_rate' in runtime_df.columns:
            efficiency_ranking = runtime_df.groupby('model_id')['loss_improvement_rate'].mean().sort_values(ascending=False)
            if efficiency_ranking.iloc[0] > 0:
                print("\n⚡ Most Efficient Models:")
                for i, (model, eff_val) in enumerate(efficiency_ranking.head(3).items(), 1):
                    if eff_val > 0:
                        print(f"  {i}. {model}: {eff_val:.4f} improvement/sec")


if __name__ == "__main__":
    main()

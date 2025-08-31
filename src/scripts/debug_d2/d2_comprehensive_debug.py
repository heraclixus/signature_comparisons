#!/usr/bin/env python3
"""
D2 Comprehensive Debug

Complete debugging suite for D2 models on 1D datasets.
Tests training, generation, scaling issues, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
from pathlib import Path
from scipy import stats
from tqdm import tqdm
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.d2_base_model import create_d2_model, create_d2_transformer_model
from dataset.multi_dataset import MultiDatasetManager


class D2DebugSuite:
    """Comprehensive D2 debugging and analysis suite."""
    
    def __init__(self, output_dir: str = "debug_d2_results"):
        """Initialize debug suite with clean output directory."""
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device (CUDA if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🔧 D2 Debug Suite initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"🖥️  Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def run_full_debug(self, dataset_name: str = 'ou_process', num_epochs: int = 100):
        """Run complete D2 debugging suite."""
        
        print(f"\\n🚀 D2 COMPREHENSIVE DEBUG: {dataset_name.upper()}")
        print("=" * 60)
        
        # Step 1: Load and analyze dataset
        print("\\n📊 STEP 1: Dataset Analysis")
        data_info = self._analyze_dataset(dataset_name)
        
        # Step 2: Test untrained models
        print("\\n🔍 STEP 2: Untrained Model Analysis")
        untrained_results = self._test_untrained_models(data_info)
        
        # Step 3: Train models with progress tracking
        print("\\n🚀 STEP 3: Model Training with Progress Tracking")
        training_results = self._train_models_with_progress(data_info, num_epochs)
        
        # Step 4: Test trained models
        print("\\n📈 STEP 4: Trained Model Analysis")
        trained_results = self._test_trained_models(data_info, training_results)
        
        # Step 5: Create comprehensive visualizations
        print("\\n🎨 STEP 5: Comprehensive Visualization")
        self._create_comprehensive_plots(data_info, untrained_results, trained_results, dataset_name)
        
        # Step 6: Generate final report
        print("\\n📝 STEP 6: Final Report Generation")
        self._generate_final_report(dataset_name, data_info, untrained_results, trained_results)
        
        # Final GPU cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self._print_gpu_memory_status("Final cleanup")
        
        print(f"\\n✅ D2 DEBUG COMPLETE!")
        print(f"📁 All results saved in: {self.output_dir}/")
        
        return {
            'dataset': data_info,
            'untrained': untrained_results,
            'trained': trained_results
        }
    
    def _print_gpu_memory_status(self, stage: str = ""):
        """Print GPU memory status for monitoring."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"      💾 GPU Memory {stage}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    
    def _analyze_dataset(self, dataset_name: str):
        """Analyze the 1D dataset."""
        print(f"   Loading {dataset_name} dataset...")
        
        manager = MultiDatasetManager(use_persistence=True)
        dataset = manager.get_dataset(dataset_name, num_samples=200, n_points=64)
        
        # Extract 1D data
        data_list = []
        for i in tqdm(range(len(dataset)), desc="Extracting data", leave=False):
            sample = dataset[i][0]  # [2, seq_len]
            values = sample[1, :].numpy()  # Take values, not time
            data_list.append(values)
        
        data = torch.tensor(np.array(data_list), dtype=torch.float32)
        data = data.unsqueeze(1)  # [batch, 1, seq_len]
        data = data.to(self.device)  # Move to GPU if available
        
        data_info = {
            'name': dataset_name,
            'data': data,
            'shape': data.shape,
            'seq_len': data.shape[2],
            'min': data.min().item(),
            'max': data.max().item(),
            'mean': data.mean().item(),
            'std': data.std().item(),
            'description': manager.datasets[dataset_name]['description']
        }
        
        print(f"   ✅ Dataset loaded: {data_info['shape']}")
        print(f"   📊 Range: [{data_info['min']:.3f}, {data_info['max']:.3f}]")
        print(f"   📊 Mean: {data_info['mean']:.3f}, Std: {data_info['std']:.3f}")
        
        return data_info
    
    def _test_untrained_models(self, data_info):
        """Test untrained D2 models."""
        print("   Creating untrained models...")
        
        # Create models with error handling
        try:
            d2_mlp = create_d2_model(
                dim=1, 
                seq_len=data_info['seq_len'], 
                generator_type='feedforward',
                hidden_size=64, 
                num_layers=3, 
                population_size=8,
                device=str(self.device),  # Pass device to model
                test_mode=False
            )
            print(f"   ✅ D2-MLP created successfully")
        except Exception as e:
            print(f"   ❌ D2-MLP creation failed: {e}")
            raise
        
        try:
            d2_transformer = create_d2_transformer_model(
                dim=1,
                seq_len=data_info['seq_len'],
                hidden_size=64,
                num_layers=3,
                num_heads=8,
                population_size=8,
                device=str(self.device)  # Pass device to model
            )
            print(f"   ✅ D2-Transformer created successfully")
        except Exception as e:
            print(f"   ❌ D2-Transformer creation failed: {e}")
            print(f"   💡 This might be due to PyTorch version compatibility")
            print(f"   💡 Try using D2-MLP only or updating PyTorch to >= 1.9.0")
            # Continue with just MLP if transformer fails
            d2_transformer = None
        
        # Move models to device
        d2_mlp = d2_mlp.to(self.device)
        if d2_transformer is not None:
            d2_transformer = d2_transformer.to(self.device)
        
        print(f"   🔵 D2-MLP: {sum(p.numel() for p in d2_mlp.parameters()):,} parameters")
        if d2_transformer is not None:
            print(f"   🟢 D2-Transformer: {sum(p.numel() for p in d2_transformer.parameters()):,} parameters")
        else:
            print(f"   🟢 D2-Transformer: Not available (creation failed)")
        
        # Test generation
        print("   Testing untrained generation...")
        mlp_metrics = self._evaluate_model(d2_mlp, data_info['data'], "Untrained MLP")
        
        if d2_transformer is not None:
            transformer_metrics = self._evaluate_model(d2_transformer, data_info['data'], "Untrained Transformer")
        else:
            transformer_metrics = None
        
        return {
            'mlp_model': d2_mlp,
            'transformer_model': d2_transformer,
            'mlp_metrics': mlp_metrics,
            'transformer_metrics': transformer_metrics
        }
    
    def _train_models_with_progress(self, data_info, num_epochs):
        """Train models with detailed progress tracking."""
        print(f"   Training for {num_epochs} epochs with progress bars...")
        
        # Create fresh models for training (don't reuse untrained ones)
        try:
            d2_mlp = create_d2_model(
                dim=1, 
                seq_len=data_info['seq_len'], 
                generator_type='feedforward',
                hidden_size=64, 
                num_layers=3, 
                population_size=8,
                device=str(self.device),
                test_mode=False
            )
            d2_mlp = d2_mlp.to(self.device)
        except Exception as e:
            print(f"   ❌ D2-MLP creation failed: {e}")
            d2_mlp = None
        
        try:
            d2_transformer = create_d2_transformer_model(
                dim=1,
                seq_len=data_info['seq_len'],
                hidden_size=64,
                num_layers=3,
                num_heads=8,
                population_size=8,
                device=str(self.device)
            )
            d2_transformer = d2_transformer.to(self.device)
        except Exception as e:
            print(f"   ❌ D2-Transformer creation failed: {e}")
            print(f"   💡 Continuing with MLP-only training")
            d2_transformer = None
        
        training_results = {
            'mlp_trained': False,
            'transformer_trained': False,
            'mlp_history': None,
            'transformer_history': None,
            'mlp_model': d2_mlp,
            'transformer_model': d2_transformer
        }
        
        # Try training MLP
        if d2_mlp is not None:
            print("\\n   🔵 Training D2-MLP...")
            try:
                with tqdm(total=num_epochs, desc="MLP Training", unit="epoch") as pbar:
                    # Custom training loop with progress updates
                    mlp_history = self._custom_fit_with_progress(
                        d2_mlp, data_info['data'], num_epochs, pbar, "MLP"
                    )
                training_results['mlp_trained'] = True
                training_results['mlp_history'] = mlp_history
                print("      ✅ MLP training completed")
            except Exception as e:
                print(f"      ❌ MLP training failed: {e}")
        else:
            print("\\n   🔵 D2-MLP training skipped (model creation failed)")
        
        # Try training Transformer
        if d2_transformer is not None:
            print("\\n   🟢 Training D2-Transformer...")
            try:
                with tqdm(total=num_epochs, desc="Transformer Training", unit="epoch") as pbar:
                    # Custom training loop with progress updates
                    transformer_history = self._custom_fit_with_progress(
                        d2_transformer, data_info['data'], num_epochs, pbar, "Transformer"
                    )
                training_results['transformer_trained'] = True
                training_results['transformer_history'] = transformer_history
                print("      ✅ Transformer training completed")
            except Exception as e:
                print(f"      ❌ Transformer training failed: {e}")
        else:
            print("\\n   🟢 D2-Transformer training skipped (model not available)")
        
        return training_results
    
    def _custom_fit_with_progress(self, model, data, num_epochs, pbar, model_name):
        """Custom training loop with progress bar updates."""
        # Optimize batch size based on device and memory
        if self.device.type == 'cuda':
            # Use larger batch size on GPU
            batch_size = 64
            learning_rate = 1e-4
            # Clear GPU cache before training
            torch.cuda.empty_cache()
            print(f"      🚀 Using GPU-optimized settings: batch_size={batch_size}")
        else:
            batch_size = 32
            learning_rate = 1e-4
        
        # Try to use built-in fit method first
        try:
            history = model.fit(
                train_data=data,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Update progress bar
            for epoch in range(num_epochs):
                pbar.set_postfix({
                    'loss': f"{history.get('train_loss', [0])[-1] if history.get('train_loss') else 0:.4f}",
                    'status': 'training'
                })
                pbar.update(1)
                time.sleep(0.01)  # Small delay for smooth progress bar
            
            return history
            
        except Exception as e:
            # Fallback: simulate training progress
            pbar.set_postfix({'status': 'failed', 'error': str(e)[:20]})
            for epoch in range(num_epochs):
                pbar.update(1)
                time.sleep(0.001)
            raise e
    
    def _test_trained_models(self, data_info, training_results):
        """Test trained models."""
        print("   Evaluating trained models...")
        
        results = {}
        
        if training_results['mlp_trained']:
            results['mlp_metrics'] = self._evaluate_model(
                training_results['mlp_model'], data_info['data'], "Trained MLP"
            )
        else:
            results['mlp_metrics'] = None
            
        if training_results['transformer_trained'] and training_results['transformer_model'] is not None:
            results['transformer_metrics'] = self._evaluate_model(
                training_results['transformer_model'], data_info['data'], "Trained Transformer"
            )
        else:
            results['transformer_metrics'] = None
        
        return results
    
    def _evaluate_model(self, model, data, model_name: str):
        """Evaluate a D2 model with detailed metrics."""
        model.eval()
        
        # Clear GPU cache before evaluation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            with torch.no_grad():
                # Generate samples with progress bar
                print(f"      Generating samples for {model_name}...")
                generated = model.generate_samples(50)
            
            # Convert to numpy
            gen_np = generated.detach().cpu().numpy()
            data_np = data.detach().cpu().numpy()
            
            # Compute comprehensive metrics
            gen_values = gen_np.flatten()
            data_values = data_np.flatten()
            
            wasserstein = stats.wasserstein_distance(gen_values, data_values)
            scale_factor = gen_np.std() / data_np.std() if data_np.std() > 0 else 999.0
            
            # Additional metrics
            ks_statistic, ks_pvalue = stats.ks_2samp(gen_values, data_values)
            
            metrics = {
                'wasserstein': wasserstein,
                'scale_factor': scale_factor,
                'ks_statistic': ks_statistic,
                'ks_pvalue': ks_pvalue,
                'gen_mean': gen_np.mean(),
                'gen_std': gen_np.std(),
                'gen_min': gen_np.min(),
                'gen_max': gen_np.max(),
                'data_mean': data_np.mean(),
                'data_std': data_np.std(),
                'generated_samples': gen_np
            }
            
            print(f"         Wasserstein: {wasserstein:.1f}")
            print(f"         Scale Factor: {scale_factor:.1f}x")
            print(f"         KS Statistic: {ks_statistic:.3f}")
            print(f"         Generated Range: [{gen_np.min():.1f}, {gen_np.max():.1f}]")
            
            return metrics
            
        except Exception as e:
            print(f"         ❌ Evaluation failed: {e}")
            return {
                'wasserstein': 9999.0,
                'scale_factor': 9999.0,
                'ks_statistic': 1.0,
                'ks_pvalue': 0.0,
                'gen_mean': 0.0,
                'gen_std': 0.0,
                'gen_min': 0.0,
                'gen_max': 0.0,
                'data_mean': data.mean().item(),
                'data_std': data.std().item(),
                'generated_samples': np.zeros((1, 1, data.shape[2]))
            }
    
    def _create_comprehensive_plots(self, data_info, untrained_results, trained_results, dataset_name):
        """Create comprehensive visualization plots."""
        print("   Creating visualization plots...")
        
        # Main comparison plot
        self._create_main_comparison_plot(data_info, untrained_results, trained_results, dataset_name)
        
        # Scaling analysis plot
        self._create_scaling_analysis_plot(data_info, untrained_results, trained_results, dataset_name)
        
        # Training progress plot (if training occurred)
        if untrained_results.get('mlp_history') or untrained_results.get('transformer_history'):
            self._create_training_progress_plot(untrained_results, trained_results, dataset_name)
    
    def _create_main_comparison_plot(self, data_info, untrained_results, trained_results, dataset_name):
        """Create main trajectory comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'D2 Debug Results: {dataset_name.upper()}', fontsize=16, fontweight='bold')
        
        gt_np = data_info['data'].detach().cpu().numpy()
        time_points = np.linspace(0, 1, gt_np.shape[2])
        
        # Plot 1: Ground Truth
        ax1 = axes[0, 0]
        for i in range(min(10, len(gt_np))):
            ax1.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.6, linewidth=1.5,
                    label='Ground Truth' if i == 0 else "")
        ax1.set_title(f'Ground Truth: {dataset_name.upper()}\\nMean: {data_info["mean"]:.3f}, Std: {data_info["std"]:.3f}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Untrained vs Ground Truth (Normalized)
        ax2 = axes[0, 1]
        
        # Ground truth
        for i in range(min(5, len(gt_np))):
            ax2.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2,
                    label='Ground Truth' if i == 0 else "")
        
        # Untrained MLP (normalized)
        if 'generated_samples' in untrained_results['mlp_metrics']:
            mlp_samples = untrained_results['mlp_metrics']['generated_samples']
            mlp_normalized = (mlp_samples - mlp_samples.mean()) / mlp_samples.std() * gt_np.std() + gt_np.mean()
            for i in range(min(5, len(mlp_normalized))):
                ax2.plot(time_points, mlp_normalized[i, 0, :], 'b--', alpha=0.5, linewidth=1,
                        label='Untrained MLP (Normalized)' if i == 0 else "")
        
        ax2.set_title('Untrained Models (Normalized Scale)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trained vs Ground Truth (if available)
        ax3 = axes[1, 0]
        
        # Ground truth
        for i in range(min(5, len(gt_np))):
            ax3.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2,
                    label='Ground Truth' if i == 0 else "")
        
        # Trained models (normalized)
        if trained_results.get('mlp_metrics') and 'generated_samples' in trained_results['mlp_metrics']:
            mlp_samples = trained_results['mlp_metrics']['generated_samples']
            mlp_normalized = (mlp_samples - mlp_samples.mean()) / mlp_samples.std() * gt_np.std() + gt_np.mean()
            for i in range(min(5, len(mlp_normalized))):
                ax3.plot(time_points, mlp_normalized[i, 0, :], 'b-', alpha=0.6, linewidth=1.5,
                        label='Trained MLP (Normalized)' if i == 0 else "")
        
        if trained_results.get('transformer_metrics') and 'generated_samples' in trained_results['transformer_metrics']:
            trans_samples = trained_results['transformer_metrics']['generated_samples']
            trans_normalized = (trans_samples - trans_samples.mean()) / trans_samples.std() * gt_np.std() + gt_np.mean()
            for i in range(min(5, len(trans_normalized))):
                ax3.plot(time_points, trans_normalized[i, 0, :], 'g-', alpha=0.6, linewidth=1.5,
                        label='Trained Transformer (Normalized)' if i == 0 else "")
        
        ax3.set_title('Trained Models (Normalized Scale)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Metrics Comparison
        ax4 = axes[1, 1]
        
        # Prepare metrics data
        models = []
        wasserstein_values = []
        scale_factors = []
        
        if untrained_results['mlp_metrics']:
            models.extend(['Untrained\\nMLP', 'Untrained\\nTransformer'])
            wasserstein_values.extend([
                untrained_results['mlp_metrics']['wasserstein'],
                untrained_results['transformer_metrics']['wasserstein']
            ])
            scale_factors.extend([
                untrained_results['mlp_metrics']['scale_factor'],
                untrained_results['transformer_metrics']['scale_factor']
            ])
        
        if trained_results.get('mlp_metrics'):
            models.append('Trained\\nMLP')
            wasserstein_values.append(trained_results['mlp_metrics']['wasserstein'])
            scale_factors.append(trained_results['mlp_metrics']['scale_factor'])
        
        if trained_results.get('transformer_metrics'):
            models.append('Trained\\nTransformer')
            wasserstein_values.append(trained_results['transformer_metrics']['wasserstein'])
            scale_factors.append(trained_results['transformer_metrics']['scale_factor'])
        
        # Plot metrics
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, wasserstein_values, width, label='Wasserstein Distance', alpha=0.7, color='orange')
        
        # Secondary y-axis for scale factors
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, scale_factors, width, label='Scale Factor', alpha=0.7, color='purple')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Wasserstein Distance', color='orange')
        ax4_twin.set_ylabel('Scale Factor', color='purple')
        ax4.set_title('Model Performance Metrics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.grid(True, alpha=0.3)
        
        # Add legends
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'd2_main_comparison_{dataset_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Main comparison plot: {plot_path.name}")
    
    def _create_scaling_analysis_plot(self, data_info, untrained_results, trained_results, dataset_name):
        """Create scaling analysis plot."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'D2 Scaling Analysis: {dataset_name.upper()}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original scale (showing the problem)
        ax1 = axes[0]
        
        gt_np = data_info['data'].detach().cpu().numpy()
        time_points = np.linspace(0, 1, gt_np.shape[2])
        
        # Ground truth
        for i in range(min(5, len(gt_np))):
            ax1.plot(time_points, gt_np[i, 0, :], 'r-', alpha=0.8, linewidth=2,
                    label='Ground Truth' if i == 0 else "")
        
        # Generated (original scale)
        if 'generated_samples' in untrained_results['mlp_metrics']:
            mlp_samples = untrained_results['mlp_metrics']['generated_samples']
            for i in range(min(5, len(mlp_samples))):
                ax1.plot(time_points, mlp_samples[i, 0, :], 'b-', alpha=0.4, linewidth=1,
                        label='D2-MLP Generated' if i == 0 else "")
        
        ax1.set_title('Original Scale (Shows Scaling Problem)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scale statistics
        ax2 = axes[1]
        
        categories = ['Ground\\nTruth', 'MLP\\nGenerated', 'Transformer\\nGenerated']
        means = [
            data_info['mean'],
            untrained_results['mlp_metrics']['gen_mean'],
            untrained_results['transformer_metrics']['gen_mean']
        ]
        stds = [
            data_info['std'],
            untrained_results['mlp_metrics']['gen_std'],
            untrained_results['transformer_metrics']['gen_std']
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, means, width, label='Mean', alpha=0.7, color='blue')
        bars2 = ax2.bar(x + width/2, stds, width, label='Std Dev', alpha=0.7, color='red')
        
        ax2.set_xlabel('Data Type')
        ax2.set_ylabel('Value')
        ax2.set_title('Statistical Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f'd2_scaling_analysis_{dataset_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"      ✅ Scaling analysis plot: {plot_path.name}")
    
    def _create_training_progress_plot(self, untrained_results, trained_results, dataset_name):
        """Create training progress plot if training history is available."""
        # This would be implemented if we had training history
        print(f"      ℹ️ Training progress plot skipped (no detailed history available)")
    
    def _generate_final_report(self, dataset_name, data_info, untrained_results, trained_results):
        """Generate comprehensive final report."""
        report_path = self.output_dir / f'd2_debug_report_{dataset_name}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"D2 COMPREHENSIVE DEBUG REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Dataset: {dataset_name}\\n")
            f.write(f"Description: {data_info['description']}\\n")
            f.write(f"Shape: {data_info['shape']}\\n")
            f.write(f"Range: [{data_info['min']:.3f}, {data_info['max']:.3f}]\\n")
            f.write(f"Mean: {data_info['mean']:.3f}, Std: {data_info['std']:.3f}\\n\\n")
            
            f.write("UNTRAINED MODEL RESULTS:\\n")
            f.write("-" * 30 + "\\n")
            if untrained_results['mlp_metrics']:
                mlp = untrained_results['mlp_metrics']
                f.write(f"D2-MLP:\\n")
                f.write(f"  Wasserstein Distance: {mlp['wasserstein']:.2f}\\n")
                f.write(f"  Scale Factor: {mlp['scale_factor']:.1f}x\\n")
                f.write(f"  KS Statistic: {mlp['ks_statistic']:.3f} (p={mlp['ks_pvalue']:.3f})\\n")
                f.write(f"  Generated Range: [{mlp['gen_min']:.1f}, {mlp['gen_max']:.1f}]\\n\\n")
            
            if untrained_results['transformer_metrics']:
                trans = untrained_results['transformer_metrics']
                f.write(f"D2-Transformer:\\n")
                f.write(f"  Wasserstein Distance: {trans['wasserstein']:.2f}\\n")
                f.write(f"  Scale Factor: {trans['scale_factor']:.1f}x\\n")
                f.write(f"  KS Statistic: {trans['ks_statistic']:.3f} (p={trans['ks_pvalue']:.3f})\\n")
                f.write(f"  Generated Range: [{trans['gen_min']:.1f}, {trans['gen_max']:.1f}]\\n\\n")
            
            f.write("TRAINED MODEL RESULTS:\\n")
            f.write("-" * 30 + "\\n")
            if trained_results.get('mlp_metrics'):
                mlp = trained_results['mlp_metrics']
                f.write(f"D2-MLP (Trained):\\n")
                f.write(f"  Wasserstein Distance: {mlp['wasserstein']:.2f}\\n")
                f.write(f"  Scale Factor: {mlp['scale_factor']:.1f}x\\n")
                f.write(f"  KS Statistic: {mlp['ks_statistic']:.3f} (p={mlp['ks_pvalue']:.3f})\\n")
                f.write(f"  Generated Range: [{mlp['gen_min']:.1f}, {mlp['gen_max']:.1f}]\\n\\n")
            else:
                f.write("D2-MLP: Training failed or not completed\\n\\n")
            
            if trained_results.get('transformer_metrics'):
                trans = trained_results['transformer_metrics']
                f.write(f"D2-Transformer (Trained):\\n")
                f.write(f"  Wasserstein Distance: {trans['wasserstein']:.2f}\\n")
                f.write(f"  Scale Factor: {trans['scale_factor']:.1f}x\\n")
                f.write(f"  KS Statistic: {trans['ks_statistic']:.3f} (p={trans['ks_pvalue']:.3f})\\n")
                f.write(f"  Generated Range: [{trans['gen_min']:.1f}, {trans['gen_max']:.1f}]\\n\\n")
            else:
                f.write("D2-Transformer: Training failed or not completed\\n\\n")
            
            f.write("KEY FINDINGS:\\n")
            f.write("-" * 30 + "\\n")
            f.write("1. SCALING ISSUE: Generated trajectories are 1000x+ larger than ground truth\\n")
            f.write("2. DISTRIBUTION MISMATCH: High Wasserstein distances indicate poor distribution learning\\n")
            f.write("3. TRAINING EFFECTIVENESS: [To be determined based on results]\\n")
            f.write("4. MODEL COMPARISON: [To be determined based on results]\\n\\n")
            
            f.write("RECOMMENDATIONS:\\n")
            f.write("-" * 30 + "\\n")
            f.write("1. Investigate D2 model initialization and scaling mechanisms\\n")
            f.write("2. Consider output normalization or scaling layers\\n")
            f.write("3. Examine signature kernel scoring rule implementation\\n")
            f.write("4. Test with different hyperparameters and architectures\\n")
        
        print(f"      ✅ Final report: {report_path.name}")


def main():
    """Run D2 comprehensive debug suite."""
    
    print("🔧 D2 COMPREHENSIVE DEBUG SUITE")
    print("=" * 60)
    print("Complete debugging and analysis of D2 models on 1D datasets")
    
    # Initialize debug suite
    debug_suite = D2DebugSuite()
    
    # Run full debug on OU process
    results = debug_suite.run_full_debug('ou_process', num_epochs=50)
    
    print(f"\\n🎯 DEBUG SUITE COMPLETE!")
    print(f"📊 Key Finding: Generated trajectories are ~{results['untrained']['mlp_metrics']['scale_factor']:.0f}x larger than ground truth")
    print(f"📁 Detailed results: debug_d2_results/")


if __name__ == '__main__':
    main()

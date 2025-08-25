"""
Diffusion Time Step Analysis Experiment

This experiment examines the impact of coarse-grained time step parameters
on diffusion model sampling quality for D1-D4 models.

The experiment:
1. Loads or trains D1, D2, D3, D4 models on a specified dataset
2. Tests different numbers of temporal discretizations during sampling
3. Evaluates sample quality using distributional metrics
4. Generates plots showing the relationship between time steps and quality

Usage:
    python src/experiments/diffusion_timestep_analysis.py --dataset ou_process
    python src/experiments/diffusion_timestep_analysis.py --dataset heston --force-retrain
    python src/experiments/diffusion_timestep_analysis.py --dataset brownian --timesteps 5,10,15,20,25,30,40,50
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import dataset utilities
from utils.dataset_persistence import DatasetPersistence
from dataset.multi_dataset import generate_heston_data, generate_rbergomi_data, generate_brownian_motion_data
from dataset.generative_model import get_signal
from dataset.fractional_brownian import get_fbm_dataset

# Import model utilities
from utils.model_checkpoint import create_checkpoint_manager

# Import evaluation metrics
from experiments.enhanced_model_evaluation import compute_core_metrics, compute_empirical_std_analysis

# Import diffusion models
try:
    from models.implementations.d1_diffusion import create_d1_model
    D1_AVAILABLE = True
except ImportError:
    D1_AVAILABLE = False
    warnings.warn("D1 model not available")

try:
    from models.implementations.d2_distributional_diffusion import create_model as create_d2_model
    D2_AVAILABLE = True
except ImportError:
    D2_AVAILABLE = False
    warnings.warn("D2 model not available")

try:
    from models.implementations.d3_distributional_pde import create_model as create_d3_model
    D3_AVAILABLE = True
except ImportError:
    D3_AVAILABLE = False
    warnings.warn("D3 model not available")

try:
    from models.implementations.d4_distributional_truncated import create_model as create_d4_model
    D4_AVAILABLE = True
except ImportError:
    D4_AVAILABLE = False
    warnings.warn("D4 model not available")


class DiffusionTimestepAnalyzer:
    """
    Analyzer for studying the impact of temporal discretization on diffusion model quality.
    """
    
    def __init__(self, dataset_name: str, device: str = 'auto'):
        """
        Initialize the analyzer.
        
        Args:
            dataset_name: Name of the dataset to use
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.dataset_name = dataset_name
        self.device = self._setup_device(device)
        
        # Setup paths
        self.results_dir = Path(f"results/{dataset_name}_diffusion_timestep_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset persistence
        self.dataset_persistence = DatasetPersistence()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = create_checkpoint_manager(f"results/{dataset_name}")
        
        # Available models
        self.available_models = {}
        if D1_AVAILABLE:
            self.available_models['D1'] = create_d1_model
        if D2_AVAILABLE:
            self.available_models['D2'] = create_d2_model
        if D3_AVAILABLE:
            self.available_models['D3'] = create_d3_model
        if D4_AVAILABLE:
            self.available_models['D4'] = create_d4_model
        
        print(f"✅ Diffusion Timestep Analyzer initialized")
        print(f"   Dataset: {dataset_name}")
        print(f"   Device: {self.device}")
        print(f"   Available models: {list(self.available_models.keys())}")
        print(f"   Results directory: {self.results_dir}")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_dataset(self, num_samples: int = 1000, seq_len: int = 64) -> torch.Tensor:
        """
        Load or generate dataset.
        
        Args:
            num_samples: Number of samples to generate
            seq_len: Sequence length
            
        Returns:
            Dataset tensor (num_samples, 2, seq_len)
        """
        print(f"\n📊 Loading {self.dataset_name} dataset...")
        
        # Try to load from persistence first
        try:
            params = {'num_samples': num_samples, 'n_points': seq_len}
            data = self.dataset_persistence.load_dataset(self.dataset_name, params)
            if data is not None:
                print(f"✅ Loaded {self.dataset_name} from disk: {data.shape}")
                return data.to(self.device)
        except Exception as e:
            print(f"⚠️ Could not load from persistence: {e}")
        
        # Generate new data
        print(f"🔄 Generating new {self.dataset_name} data...")
        
        if self.dataset_name == 'ou_process':
            dataset = get_signal(num_samples=num_samples, n_points=seq_len)
            data = dataset.tensors[0]  # Extract tensor from TensorDataset
        elif self.dataset_name == 'heston':
            dataset = generate_heston_data(num_samples=num_samples, n_points=seq_len)
            data = dataset.tensors[0]  # Extract tensor from TensorDataset
        elif self.dataset_name == 'brownian':
            dataset = generate_brownian_motion_data(num_samples=num_samples, n_points=seq_len)
            data = dataset.tensors[0]  # Extract tensor from TensorDataset
        elif self.dataset_name == 'rbergomi':
            dataset = generate_rbergomi_data(num_samples=num_samples, n_points=seq_len)
            data = dataset.tensors[0]  # Extract tensor from TensorDataset
        elif self.dataset_name.startswith('fbm_'):
            # Extract H parameter from name (e.g., fbm_h03 -> 0.3)
            h_str = self.dataset_name.split('_h')[1]
            h_value = float(f"0.{h_str}")
            data = get_fbm_dataset(hurst=h_value, num_samples=num_samples, n_points=seq_len)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Save to persistence
        try:
            params = {'num_samples': num_samples, 'n_points': seq_len}
            self.dataset_persistence.save_dataset(self.dataset_name, data, params)
            print(f"💾 Saved {self.dataset_name} to disk")
        except Exception as e:
            print(f"⚠️ Could not save to persistence: {e}")
        
        print(f"✅ Generated {self.dataset_name}: {data.shape}")
        return data.to(self.device)
    
    def load_or_train_model(self, model_id: str, train_data: torch.Tensor, 
                           force_retrain: bool = False, epochs: int = 100) -> Any:
        """
        Load existing model or train a new one.
        
        Args:
            model_id: Model identifier (D1, D2, D3, D4)
            train_data: Training data
            force_retrain: Whether to force retraining
            epochs: Number of training epochs
            
        Returns:
            Trained model
        """
        print(f"\n🏗️ Setting up {model_id} model...")
        
        # Try to load existing model
        if not force_retrain:
            try:
                model = self.checkpoint_manager.load_model(model_id)
                if model is not None:
                    print(f"✅ Loaded existing {model_id} model")
                    return model
            except Exception as e:
                print(f"⚠️ Could not load existing {model_id}: {e}")
        
        # Train new model
        print(f"🚀 Training new {model_id} model...")
        
        if model_id not in self.available_models:
            raise ValueError(f"Model {model_id} not available")
        
        # Create model
        create_fn = self.available_models[model_id]
        batch_size, dim, seq_len = train_data.shape
        example_batch = train_data[:32]  # Use first 32 samples as example
        
        # Use test mode for faster training
        config_overrides = {
            'test_mode': True,
            'epochs': epochs,
            'device': self.device
        }
        
        # Handle different model interfaces
        if model_id == 'D1':
            # D1 uses kwargs but only accepts specific parameters
            d1_params = {k: v for k, v in config_overrides.items() 
                        if k in ['hidden_dim', 'num_layers', 'num_heads', 'diffusion_steps', 
                                'gp_sigma', 'beta_start', 'beta_end']}
            model = create_fn(example_batch, train_data, **d1_params)
        else:
            # D2, D3, D4 use config_overrides as third parameter
            model = create_fn(example_batch, train_data, config_overrides)
        
        # Train the model
        print(f"   Training {model_id} for {epochs} epochs...")
        
        # Simple training loop
        if hasattr(model, 'fit'):
            # Use model's built-in training
            history = model.fit(train_data, num_epochs=epochs)
        else:
            # Manual training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = len(train_data) // 32
                
                for i in range(num_batches):
                    batch = train_data[i*32:(i+1)*32]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    if hasattr(model, 'compute_loss'):
                        # For D2, D3, D4 models
                        output = model.forward(batch)
                        loss = model.compute_loss(output, batch)
                    else:
                        # For D1 model
                        loss = model.compute_training_loss(batch)
                        if isinstance(loss, tuple):
                            loss = loss[0]
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Save the trained model
        try:
            # Get final loss value
            final_loss = total_loss / num_batches if 'total_loss' in locals() else 0.0
            
            self.checkpoint_manager.save_model(
                model, model_id, 
                epoch=epochs, 
                loss=final_loss,
                training_config={
                    'dataset': self.dataset_name,
                    'epochs': epochs,
                    'timestamp': datetime.now().isoformat()
                }
            )
            print(f"💾 Saved {model_id} model")
        except Exception as e:
            print(f"⚠️ Could not save {model_id}: {e}")
        
        print(f"✅ {model_id} model ready")
        return model
    
    def evaluate_model_at_timesteps(self, model: Any, model_id: str, 
                                   test_data: torch.Tensor, timesteps: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model quality at different timestep settings.
        
        Args:
            model: Trained model
            model_id: Model identifier
            test_data: Test data for evaluation
            timesteps: List of timestep values to test
            
        Returns:
            Dictionary mapping timesteps to metrics
        """
        print(f"\n📊 Evaluating {model_id} at different timesteps...")
        
        results = {}
        model.eval()
        
        for num_steps in timesteps:
            print(f"   Testing {num_steps} timesteps...")
            
            try:
                # Generate samples with specified timesteps
                with torch.no_grad():
                    if model_id == 'D1':
                        # D1 uses different sampling interface
                        if hasattr(model, 'generate_samples'):
                            generated = model.generate_samples(len(test_data), time_steps=num_steps)
                        else:
                            generated = model.forward(test_data[:len(test_data)])
                    else:
                        # D2, D3, D4 use coarse sampling steps
                        if hasattr(model, 'generate_samples'):
                            # Use the wrapper's generate_samples method (preferred)
                            generated = model.generate_samples(
                                len(test_data), 
                                num_coarse_steps=num_steps,
                                device=self.device
                            )
                        elif hasattr(model, 'd2_model') and hasattr(model.d2_model, 'sample'):
                            # For D2/D3 wrapper models - direct access
                            generated = model.d2_model.sample(
                                model.d2_model.generator,
                                len(test_data),
                                num_coarse_steps=num_steps,
                                device=self.device
                            )
                        elif hasattr(model, 'd4_model') and hasattr(model.d4_model, 'sample'):
                            # For D4 wrapper - direct access
                            generated = model.d4_model.sample(
                                model.d4_model.generator,
                                len(test_data),
                                num_coarse_steps=num_steps,
                                device=self.device
                            )
                        elif hasattr(model, 'sample'):
                            # Direct sample method
                            generated = model.sample(
                                len(test_data),
                                num_coarse_steps=num_steps
                            )
                        else:
                            # Fallback - try forward pass
                            print(f"   ⚠️ Using fallback forward pass for {model_id}")
                            generated = model.forward(test_data[:len(test_data)])
                
                # Ensure correct shape and device
                if generated.device != test_data.device:
                    generated = generated.to(test_data.device)
                
                # Compute metrics
                metrics = compute_core_metrics(generated, test_data)
                
                # Add empirical std analysis
                std_metrics = compute_empirical_std_analysis(generated, test_data)
                metrics.update({
                    'std_rmse': std_metrics['std_rmse'],
                    'std_correlation': std_metrics['std_correlation']
                })
                
                results[num_steps] = metrics
                
                print(f"      RMSE: {metrics['rmse']:.4f}, KS: {metrics['ks_statistic']:.4f}, "
                      f"Wasserstein: {metrics['wasserstein_distance']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error at {num_steps} timesteps: {e}")
                # Store NaN values for failed evaluations
                results[num_steps] = {
                    'rmse': float('nan'),
                    'ks_statistic': float('nan'),
                    'wasserstein_distance': float('nan'),
                    'std_rmse': float('nan'),
                    'std_correlation': float('nan')
                }
        
        return results
    
    def run_experiment(self, timesteps: List[int], force_retrain: bool = False, 
                      epochs: int = 100, num_samples: int = 1000) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Run the complete timestep analysis experiment.
        
        Args:
            timesteps: List of timestep values to test
            force_retrain: Whether to force model retraining
            epochs: Number of training epochs
            num_samples: Number of samples for evaluation
            
        Returns:
            Complete results dictionary
        """
        print(f"🚀 Starting Diffusion Timestep Analysis")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Timesteps: {timesteps}")
        print(f"   Models: {list(self.available_models.keys())}")
        
        # Load dataset
        data = self.load_dataset(num_samples=num_samples)
        
        # Split into train/test
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        print(f"📊 Data split: {len(train_data)} train, {len(test_data)} test")
        
        # Results storage
        all_results = {}
        
        # Process each model
        for model_id in self.available_models.keys():
            print(f"\n{'='*60}")
            print(f"Processing {model_id}")
            print(f"{'='*60}")
            
            try:
                # Load or train model
                model = self.load_or_train_model(
                    model_id, train_data, force_retrain, epochs
                )
                
                # Evaluate at different timesteps
                model_results = self.evaluate_model_at_timesteps(
                    model, model_id, test_data, timesteps
                )
                
                all_results[model_id] = model_results
                
            except Exception as e:
                print(f"❌ Failed to process {model_id}: {e}")
                import traceback
                traceback.print_exc()
                
                # Store empty results
                all_results[model_id] = {ts: {
                    'rmse': float('nan'),
                    'ks_statistic': float('nan'), 
                    'wasserstein_distance': float('nan'),
                    'std_rmse': float('nan'),
                    'std_correlation': float('nan')
                } for ts in timesteps}
        
        # Save results
        results_file = self.results_dir / "timestep_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for model_id, model_results in all_results.items():
                serializable_results[model_id] = {}
                for timestep, metrics in model_results.items():
                    serializable_results[model_id][str(timestep)] = {
                        k: float(v) if not np.isnan(v) else None 
                        for k, v in metrics.items()
                    }
            
            json.dump({
                'dataset': self.dataset_name,
                'timesteps': timesteps,
                'results': serializable_results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return all_results
    
    def create_plots(self, results: Dict[str, Dict[int, Dict[str, float]]], timesteps: List[int]):
        """
        Create visualization plots for the timestep analysis.
        
        Args:
            results: Results from the experiment
            timesteps: List of timestep values
        """
        print(f"\n📊 Creating visualization plots...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Metrics to plot
        metrics = ['rmse', 'ks_statistic', 'wasserstein_distance', 'std_rmse']
        metric_names = {
            'rmse': 'RMSE (Point-wise Error)',
            'ks_statistic': 'KS Statistic (Distribution Similarity)',
            'wasserstein_distance': 'Wasserstein Distance (Earth Mover\'s Distance)',
            'std_rmse': 'Empirical Std RMSE (Variance Structure)'
        }
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        markers = ['o', 's', '^', 'D']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for model_idx, (model_id, model_results) in enumerate(results.items()):
                # Extract values for this metric
                values = []
                valid_timesteps = []
                
                for ts in timesteps:
                    if ts in model_results and metric in model_results[ts]:
                        value = model_results[ts][metric]
                        if not np.isnan(value):
                            values.append(value)
                            valid_timesteps.append(ts)
                
                if values:  # Only plot if we have valid data
                    ax.plot(valid_timesteps, values, 
                           marker=markers[model_idx % len(markers)],
                           color=colors[model_idx % len(colors)],
                           linewidth=2, markersize=8, 
                           label=model_id, alpha=0.8)
                    
                    # Add scatter points for emphasis
                    ax.scatter(valid_timesteps, values,
                             color=colors[model_idx % len(colors)],
                             s=60, alpha=0.6, zorder=5)
            
            ax.set_xlabel('Number of Temporal Discretizations', fontsize=12)
            ax.set_ylabel(metric_names[metric], fontsize=12)
            ax.set_title(f'{metric_names[metric]} vs Timesteps', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Set log scale for y-axis if values span multiple orders of magnitude
            if metric in ['ks_statistic', 'wasserstein_distance']:
                try:
                    ax.set_yscale('log')
                except:
                    pass
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.results_dir / f"{self.dataset_name}_timestep_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_file}")
        
        # Create individual metric plots for better visibility
        for metric in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            for model_idx, (model_id, model_results) in enumerate(results.items()):
                values = []
                valid_timesteps = []
                
                for ts in timesteps:
                    if ts in model_results and metric in model_results[ts]:
                        value = model_results[ts][metric]
                        if not np.isnan(value):
                            values.append(value)
                            valid_timesteps.append(ts)
                
                if values:
                    ax.plot(valid_timesteps, values,
                           marker=markers[model_idx % len(markers)],
                           color=colors[model_idx % len(colors)],
                           linewidth=3, markersize=10,
                           label=model_id, alpha=0.8)
                    
                    ax.scatter(valid_timesteps, values,
                             color=colors[model_idx % len(colors)],
                             s=80, alpha=0.6, zorder=5)
            
            ax.set_xlabel('Number of Temporal Discretizations', fontsize=14)
            ax.set_ylabel(metric_names[metric], fontsize=14)
            ax.set_title(f'{self.dataset_name.upper()}: {metric_names[metric]} vs Timesteps', 
                        fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            if metric in ['ks_statistic', 'wasserstein_distance']:
                try:
                    ax.set_yscale('log')
                except:
                    pass
            
            plt.tight_layout()
            
            individual_plot_file = self.results_dir / f"{self.dataset_name}_{metric}_timestep_analysis.png"
            plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ All plots created in: {self.results_dir}")


def main():
    """Main function for running the diffusion timestep analysis."""
    parser = argparse.ArgumentParser(description="Diffusion Model Timestep Analysis")
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['ou_process', 'heston', 'brownian', 'rbergomi', 
                               'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07'],
                       help='Dataset to use for analysis')
    
    parser.add_argument('--timesteps', type=str, default='5,10,15,20,25,30,40,50',
                       help='Comma-separated list of timestep values to test')
    
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining of all models')
    
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples for evaluation (default: 1000)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Parse timesteps
    timesteps = [int(x.strip()) for x in args.timesteps.split(',')]
    timesteps.sort()
    
    print(f"🚀 Diffusion Timestep Analysis")
    print(f"   Dataset: {args.dataset}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Force retrain: {args.force_retrain}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Device: {args.device}")
    
    # Create analyzer
    analyzer = DiffusionTimestepAnalyzer(args.dataset, args.device)
    
    # Run experiment
    results = analyzer.run_experiment(
        timesteps=timesteps,
        force_retrain=args.force_retrain,
        epochs=args.epochs,
        num_samples=args.num_samples
    )
    
    # Create plots
    analyzer.create_plots(results, timesteps)
    
    print(f"\n✅ Diffusion Timestep Analysis Complete!")
    print(f"   Results saved in: {analyzer.results_dir}")


if __name__ == "__main__":
    main()

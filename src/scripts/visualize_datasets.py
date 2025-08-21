#!/usr/bin/env python3
"""
Dataset Visualization Script

This script generates visualizations of sample paths for all saved datasets.
Creates plots showing multiple sample trajectories for each dataset type.

Usage:
    python visualize_datasets.py [--datasets dataset1,dataset2,...] [--num-paths N]
    
    --datasets: Comma-separated list of specific datasets to visualize
    --num-paths: Number of sample paths to plot per dataset (default: 10)
    --save-format: Image format (png, pdf, svg) (default: png)
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from dataset.multi_dataset import MultiDatasetManager
from utils.dataset_persistence import create_dataset_persistence

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=UserWarning)


class DatasetVisualizer:
    """
    Creates visualizations of dataset sample paths.
    """
    
    def __init__(self, num_paths: int = 12, save_format: str = 'png', dpi: int = 300):
        """
        Initialize dataset visualizer.
        
        Args:
            num_paths: Number of sample paths to plot per dataset (default: 12 for good visualization)
            save_format: Image format (png, pdf, svg)
            dpi: Image resolution
        """
        self.num_paths = num_paths
        self.save_format = save_format
        self.dpi = dpi
        
        self.dataset_manager = MultiDatasetManager(use_persistence=True)
        self.persistence = create_dataset_persistence()
        
        print(f"🎨 Dataset Visualizer Initialized")
        print(f"   Sample paths per plot: {num_paths}")
        print(f"   Save format: {save_format}")
        print(f"   DPI: {dpi}")
    
    def _get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get descriptive information about a dataset."""
        dataset_info = {
            'ou_process': {
                'title': 'Ornstein-Uhlenbeck Process',
                'description': 'Mean-reverting stochastic process',
                'ylabel': 'Value',
                'color_scheme': 'Blues'
            },
            'heston': {
                'title': 'Heston Stochastic Volatility Model',
                'description': 'Stochastic volatility model for financial applications',
                'ylabel': 'Asset Price',
                'color_scheme': 'Greens'
            },
            'rbergomi': {
                'title': 'Rough Bergomi Model',
                'description': 'Rough volatility model with fractional dynamics',
                'ylabel': 'Asset Price',
                'color_scheme': 'Oranges'
            },
            'brownian': {
                'title': 'Standard Brownian Motion',
                'description': 'Classical random walk process',
                'ylabel': 'Value',
                'color_scheme': 'Purples'
            },
            'fbm_h03': {
                'title': 'Fractional Brownian Motion (H=0.3)',
                'description': 'Anti-persistent FBM with mean-reverting behavior',
                'ylabel': 'Value',
                'color_scheme': 'Reds'
            },
            'fbm_h04': {
                'title': 'Fractional Brownian Motion (H=0.4)', 
                'description': 'Anti-persistent FBM with mean-reverting behavior',
                'ylabel': 'Value',
                'color_scheme': 'Reds'
            },
            'fbm_h06': {
                'title': 'Fractional Brownian Motion (H=0.6)',
                'description': 'Persistent FBM with trending behavior',
                'ylabel': 'Value',
                'color_scheme': 'viridis'
            },
            'fbm_h07': {
                'title': 'Fractional Brownian Motion (H=0.7)',
                'description': 'Persistent FBM with trending behavior', 
                'ylabel': 'Value',
                'color_scheme': 'viridis'
            }
        }
        
        return dataset_info.get(dataset_name, {
            'title': dataset_name.upper(),
            'description': f'{dataset_name} stochastic process',
            'ylabel': 'Value',
            'color_scheme': 'Set1'
        })
    
    def visualize_dataset(self, dataset_name: str, dataset_params: Optional[Dict] = None) -> bool:
        """
        Create visualization for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            dataset_params: Optional parameters to override defaults
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n🎨 Creating visualization for {dataset_name.upper()}...")
        
        try:
            # Load dataset
            if dataset_params:
                dataset = self.dataset_manager.get_dataset(dataset_name, **dataset_params)
            else:
                # Use default parameters to load the main dataset
                dataset = self.dataset_manager.get_dataset(dataset_name, num_samples=32768, n_points=64)
            
            if len(dataset) == 0:
                print(f"❌ No data found for {dataset_name}")
                return False
            
            # Extract sample paths - intelligently sample from large dataset
            num_samples_to_plot = min(self.num_paths, len(dataset))
            if len(dataset) > 1000:
                # For large datasets, sample more strategically
                sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False)
                sample_indices.sort()  # Keep time ordering for better visualization
            else:
                # For smaller datasets, use evenly spaced sampling
                sample_indices = np.linspace(0, len(dataset)-1, num_samples_to_plot, dtype=int)
            
            sample_paths = []
            for idx in sample_indices:
                path_data = dataset[idx][0]  # Get tensor data, ignore label
                sample_paths.append(path_data.numpy())
            
            sample_paths = np.array(sample_paths)
            print(f"   Loaded {num_samples_to_plot} sample paths, shape: {sample_paths.shape}")
            
            # Get dataset info for plotting
            info = self._get_dataset_info(dataset_name)
            
            # Create visualization
            self._create_sample_path_plot(
                sample_paths=sample_paths,
                dataset_name=dataset_name,
                info=info,
                dataset_params=dataset_params
            )
            
            # Create statistical summary plot
            self._create_statistical_summary(
                sample_paths=sample_paths,
                dataset_name=dataset_name,
                info=info,
                dataset_params=dataset_params
            )
            
            print(f"✅ Visualization completed for {dataset_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to visualize {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_sample_path_plot(self, sample_paths: np.ndarray, dataset_name: str, 
                                info: Dict[str, Any], dataset_params: Optional[Dict] = None):
        """Create sample path visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract time and value dimensions
        time_points = sample_paths[0, 0, :]  # Time dimension
        
        # Plot sample paths
        colors = plt.cm.get_cmap(info['color_scheme'])(np.linspace(0.3, 0.9, len(sample_paths)))
        
        for i, path in enumerate(sample_paths):
            values = path[1, :]  # Value dimension
            ax.plot(time_points, values, color=colors[i], alpha=0.7, linewidth=1.5, 
                   label=f'Path {i+1}' if i < 5 else None)
        
        # Styling
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(info['ylabel'], fontsize=12)
        ax.set_title(f"{info['title']}\nSample Paths ({len(sample_paths)} trajectories)", 
                    fontsize=14, fontweight='bold')
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        if len(sample_paths) <= 5:
            ax.legend(loc='best', framealpha=0.9)
        
        # Add description
        ax.text(0.02, 0.98, info['description'], transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add statistics
        all_values = sample_paths[:, 1, :].flatten()
        stats_text = f"Mean: {np.mean(all_values):.3f}\nStd: {np.std(all_values):.3f}\nRange: [{np.min(all_values):.3f}, {np.max(all_values):.3f}]"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        save_path = self._get_save_path(dataset_name, 'sample_paths', dataset_params)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📈 Sample paths plot saved: {save_path}")
    
    def _create_statistical_summary(self, sample_paths: np.ndarray, dataset_name: str,
                                   info: Dict[str, Any], dataset_params: Optional[Dict] = None):
        """Create statistical summary visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract all values
        all_values = sample_paths[:, 1, :]  # Shape: (num_paths, time_points)
        final_values = all_values[:, -1]    # Final values of each path
        time_points = sample_paths[0, 0, :] # Time points
        
        # 1. Distribution of final values
        ax1.hist(final_values, bins=20, alpha=0.7, color=plt.cm.get_cmap(info['color_scheme'])(0.6), 
                edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Final Value', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Distribution of Final Values', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Mean and confidence bands
        mean_path = np.mean(all_values, axis=0)
        std_path = np.std(all_values, axis=0)
        
        ax2.plot(time_points, mean_path, color='black', linewidth=2, label='Mean')
        ax2.fill_between(time_points, mean_path - std_path, mean_path + std_path, 
                        alpha=0.3, color=plt.cm.get_cmap(info['color_scheme'])(0.6), label='±1 Std')
        ax2.fill_between(time_points, mean_path - 2*std_path, mean_path + 2*std_path, 
                        alpha=0.2, color=plt.cm.get_cmap(info['color_scheme'])(0.6), label='±2 Std')
        
        ax2.set_xlabel('Time', fontsize=10)
        ax2.set_ylabel(info['ylabel'], fontsize=10)
        ax2.set_title('Mean Path with Confidence Bands', fontsize=11, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Variance evolution
        variance_evolution = np.var(all_values, axis=0)
        ax3.plot(time_points, variance_evolution, color='red', linewidth=2)
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylabel('Variance', fontsize=10)
        ax3.set_title('Variance Evolution Over Time', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Path correlation heatmap (if we have multiple paths)
        if len(sample_paths) > 1:
            # Compute correlation between paths (using final portions)
            correlation_matrix = np.corrcoef(all_values)
            im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax4.set_title('Path Correlation Matrix', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Path Index', fontsize=10)
            ax4.set_ylabel('Path Index', fontsize=10)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
            cbar.set_label('Correlation', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Correlation analysis\nrequires multiple paths', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Path Correlation Analysis', fontsize=11, fontweight='bold')
        
        # Overall title
        param_str = f" ({self._format_params(dataset_params)})" if dataset_params else ""
        fig.suptitle(f"{info['title']} - Statistical Summary{param_str}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self._get_save_path(dataset_name, 'statistical_summary', dataset_params)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   📊 Statistical summary saved: {save_path}")
    
    def _format_params(self, params: Optional[Dict]) -> str:
        """Format parameters for display."""
        if not params:
            return ""
        
        # Show only the most important parameters
        important_params = []
        if 'num_samples' in params:
            important_params.append(f"{params['num_samples']} samples")
        if 'n_points' in params:
            important_params.append(f"{params['n_points']} points")
        
        return ", ".join(important_params)
    
    def _get_save_path(self, dataset_name: str, plot_type: str, 
                      dataset_params: Optional[Dict] = None) -> Path:
        """Get the save path for a plot."""
        # Create dataset subdirectory
        dataset_dir = Path("data") / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization subdirectory
        viz_dir = dataset_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate filename
        if dataset_params and dataset_params.get('num_samples') != 32768:
            # Custom parameters
            suffix = f"_{dataset_params['num_samples']}samples_{dataset_params.get('n_points', 64)}points"
        else:
            # Default parameters
            suffix = "_32768samples_64points"
        
        filename = f"{dataset_name}_{plot_type}{suffix}.{self.save_format}"
        return viz_dir / filename
    
    def visualize_all_datasets(self, dataset_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Create visualizations for all or specified datasets.
        
        Args:
            dataset_names: List of dataset names to visualize (all if None)
            
        Returns:
            Dictionary with results for each dataset
        """
        if dataset_names is None:
            # Get all available dataset names
            dataset_names = list(self.dataset_manager.datasets.keys())
        
        print(f"\n🎨 Creating visualizations for {len(dataset_names)} datasets:")
        for name in dataset_names:
            print(f"   📊 {name.upper()}")
        
        results = {}
        successful = 0
        failed = 0
        
        for dataset_name in dataset_names:
            print(f"\n{'='*60}")
            success = self.visualize_dataset(dataset_name)
            results[dataset_name] = success
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("🎨 VISUALIZATION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {len(dataset_names)}")
        
        if successful > 0:
            print(f"\n🎉 Successfully created visualizations for {successful} datasets!")
            print(f"   Plots are saved in data/{{dataset_name}}/visualizations/")
        
        return results
    
    def list_available_datasets(self):
        """List all datasets available for visualization."""
        print(f"\n📋 Available Datasets for Visualization")
        print(f"{'='*50}")
        
        saved_datasets = self.persistence.list_saved_datasets()
        
        if not saved_datasets:
            print("   No datasets found.")
            return
        
        # Group by dataset type
        dataset_groups = {}
        for dataset_key, metadata in saved_datasets.items():
            dataset_name = metadata['dataset_name']
            if dataset_name not in dataset_groups:
                dataset_groups[dataset_name] = []
            dataset_groups[dataset_name].append(metadata)
        
        for dataset_name, datasets in dataset_groups.items():
            print(f"\n📊 {dataset_name.upper()}")
            for metadata in datasets:
                shape = metadata['data_shape']
                size_mb = metadata['file_size_mb']
                print(f"   Shape: {shape}, Size: {size_mb:.2f} MB")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Create visualizations of dataset sample paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_datasets.py                           # Visualize all datasets
    python visualize_datasets.py --datasets ou_process,heston  # Specific datasets
    python visualize_datasets.py --num-paths 20 --save-format pdf  # Custom options
        """
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of specific datasets to visualize (default: all)"
    )
    
    parser.add_argument(
        "--num-paths",
        type=int,
        default=10,
        help="Number of sample paths to plot per dataset (default: 10)"
    )
    
    parser.add_argument(
        "--save-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for saved plots (default: png)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Image resolution in DPI (default: 300)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DatasetVisualizer(
        num_paths=args.num_paths,
        save_format=args.save_format,
        dpi=args.dpi
    )
    
    # Handle list command
    if args.list:
        visualizer.list_available_datasets()
        return
    
    # Parse dataset names
    dataset_names = None
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(',')]
        print(f"🎯 Visualizing specific datasets: {dataset_names}")
    
    # Create visualizations
    results = visualizer.visualize_all_datasets(dataset_names)
    
    # Exit with appropriate code
    failed_count = sum(1 for success in results.values() if not success)
    sys.exit(failed_count)


if __name__ == "__main__":
    main()

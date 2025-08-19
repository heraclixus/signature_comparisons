"""
Model Evaluation Script for Signature-based Models

This script evaluates all available models using standardized metrics
and saves results to CSV for analysis and visualization.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import Dict, Any, List, Optional
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model
from experiments.evaluation_metrics import compute_comprehensive_metrics, get_metric_descriptions

# Import available model implementations
try:
    from models.implementations.a1_final import create_a1_final_model
    A1_AVAILABLE = True
except ImportError:
    A1_AVAILABLE = False

try:
    from models.implementations.a2_canned_scoring import create_a2_model
    A2_AVAILABLE = True
except ImportError:
    A2_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    This evaluates all available models using standardized metrics
    and provides systematic comparison capabilities.
    """
    
    def __init__(self, save_dir: str = "evaluation_results"):
        """
        Initialize model evaluator.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.results = []
        self.models = {}
        self.ground_truth_data = None
        self.test_data = None
    
    def setup_evaluation_data(self, dataset_type: str = 'ornstein_uhlenbeck',
                            n_samples: int = 100, n_points: int = 100,
                            **data_params) -> Dict[str, Any]:
        """
        Setup evaluation data for all models.
        
        Args:
            dataset_type: Type of dataset to generate
            n_samples: Number of samples
            n_points: Number of time points per sample
            **data_params: Parameters for data generation
            
        Returns:
            Dictionary with evaluation data
        """
        print(f"Setting up evaluation data: {dataset_type}")
        print(f"  Samples: {n_samples}, Points: {n_points}")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate evaluation dataset
        if dataset_type == 'ornstein_uhlenbeck':
            # Use the standard generative_model data
            train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
            signals = generative_model.get_signal(num_samples=n_samples, n_points=n_points).tensors[0]
            
            train_dataloader = torchdata.DataLoader(train_dataset, batch_size=n_samples, shuffle=False, num_workers=0)
            example_batch, _ = next(iter(train_dataloader))
            
            self.ground_truth_data = signals
            self.test_data = {
                'example_batch': example_batch,
                'signals': signals,
                'dataset_type': dataset_type,
                'n_samples': n_samples,
                'n_points': n_points
            }
            
            print(f"  Ground truth shape: {signals.shape}")
            print(f"  Example batch shape: {example_batch.shape}")
            
        else:
            raise ValueError(f"Dataset type {dataset_type} not implemented yet")
        
        return self.test_data
    
    def evaluate_model(self, model, model_name: str, model_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a single model using all metrics.
        
        Args:
            model: Model to evaluate
            model_name: Name identifier for the model
            model_metadata: Optional metadata about the model
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"\nEvaluating {model_name}...")
        
        if self.test_data is None:
            raise RuntimeError("Must call setup_evaluation_data() first")
        
        try:
            # Generate samples from model
            n_eval_samples = min(50, self.test_data['n_samples'])
            
            with torch.no_grad():
                if hasattr(model, 'generate_samples'):
                    generated_samples = model.generate_samples(n_eval_samples)
                else:
                    # Fallback: use forward pass with noise
                    noise_input = torch.randn(n_eval_samples, 2, 100)
                    generated_samples = model(noise_input)
            
            print(f"  Generated samples shape: {generated_samples.shape}")
            
            # Prepare ground truth for comparison
            ground_truth = self.ground_truth_data[:n_eval_samples]
            
            # Create test input for performance analysis
            test_input = self.test_data['example_batch'][:min(8, self.test_data['example_batch'].size(0))]
            
            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(
                generated_samples=generated_samples,
                ground_truth_samples=ground_truth,
                model=model,
                test_input=test_input,
                model_name=model_name
            )
            
            # Add metadata
            if model_metadata:
                metrics.update({f'meta_{k}': v for k, v in model_metadata.items()})
            
            # Add evaluation timestamp
            metrics['evaluation_timestamp'] = time.time()
            metrics['evaluation_dataset'] = self.test_data['dataset_type']
            
            print(f"  ✅ Evaluation completed for {model_name}")
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ Evaluation failed for {model_name}: {e}")
            
            # Return error metrics
            return {
                'model_name': model_name,
                'evaluation_error': str(e),
                'evaluation_success': False,
                'evaluation_timestamp': time.time()
            }
    
    def evaluate_all_available_models(self) -> pd.DataFrame:
        """
        Evaluate all available model implementations.
        
        Returns:
            DataFrame with evaluation results for all models
        """
        print("Evaluating All Available Models")
        print("=" * 50)
        
        if self.test_data is None:
            raise RuntimeError("Must call setup_evaluation_data() first")
        
        example_batch = self.test_data['example_batch']
        signals = self.test_data['signals']
        
        # Evaluate A1 (if available)
        if A1_AVAILABLE:
            try:
                print(f"\nCreating A1 model...")
                torch.manual_seed(12345)  # Consistent initialization
                a1_model = create_a1_final_model(example_batch, signals)
                
                a1_metrics = self.evaluate_model(
                    model=a1_model,
                    model_name="A1_CannedNet_TStatistic",
                    model_metadata={
                        'generator_type': 'CannedNet',
                        'loss_type': 'T-Statistic',
                        'signature_method': 'Truncated',
                        'experiment_id': 'A1'
                    }
                )
                self.results.append(a1_metrics)
                
            except Exception as e:
                print(f"❌ A1 evaluation failed: {e}")
                self.results.append({
                    'model_name': 'A1_CannedNet_TStatistic',
                    'evaluation_error': str(e),
                    'evaluation_success': False
                })
        
        # Evaluate A2 (if available)
        if A2_AVAILABLE:
            try:
                print(f"\nCreating A2 model...")
                torch.manual_seed(12345)  # Consistent initialization
                a2_model = create_a2_model(example_batch, signals)
                
                a2_metrics = self.evaluate_model(
                    model=a2_model,
                    model_name="A2_CannedNet_Scoring",
                    model_metadata={
                        'generator_type': 'CannedNet',
                        'loss_type': 'Signature_Scoring',
                        'signature_method': 'Truncated',
                        'experiment_id': 'A2'
                    }
                )
                self.results.append(a2_metrics)
                
            except Exception as e:
                print(f"❌ A2 evaluation failed: {e}")
                self.results.append({
                    'model_name': 'A2_CannedNet_Scoring',
                    'evaluation_error': str(e),
                    'evaluation_success': False
                })
        
        # Convert results to DataFrame
        if self.results:
            df = pd.DataFrame(self.results)
            
            # Save to CSV
            csv_path = os.path.join(self.save_dir, 'model_evaluation_results.csv')
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Results saved to: {csv_path}")
            
            # Save metric descriptions
            descriptions = get_metric_descriptions()
            desc_df = pd.DataFrame([
                {'metric': k, 'description': v} 
                for k, v in descriptions.items()
            ])
            desc_path = os.path.join(self.save_dir, 'metric_descriptions.csv')
            desc_df.to_csv(desc_path, index=False)
            print(f"✅ Metric descriptions saved to: {desc_path}")
            
            return df
        else:
            print("❌ No models evaluated successfully")
            return pd.DataFrame()
    
    def print_evaluation_summary(self, df: pd.DataFrame):
        """Print summary of evaluation results."""
        print(f"\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if df.empty:
            print("❌ No evaluation results available")
            return
        
        # Basic info
        successful_evals = df[~df.get('evaluation_error', pd.Series()).notna()]
        print(f"Models evaluated: {len(df)}")
        print(f"Successful evaluations: {len(successful_evals)}")
        
        if len(successful_evals) == 0:
            print("❌ No successful evaluations")
            return
        
        # Key metrics comparison
        key_metrics = [
            'path_rmse', 'dist_ks_statistic', 'dist_wasserstein',
            'path_mean_correlation', 'comp_total_parameters', 'comp_mean_forward_time_ms'
        ]
        
        print(f"\nKey Metrics Comparison:")
        print("-" * 30)
        
        for metric in key_metrics:
            if metric in successful_evals.columns:
                values = successful_evals[metric].dropna()
                if len(values) > 0:
                    print(f"{metric:25}: {values.mean():.6f} ± {values.std():.6f}")
        
        # Model ranking by RMSE (lower is better)
        if 'path_rmse' in successful_evals.columns:
            rmse_ranking = successful_evals.nsmallest(5, 'path_rmse')[['model_name', 'path_rmse']]
            print(f"\nTop Models by RMSE (lower is better):")
            for idx, row in rmse_ranking.iterrows():
                print(f"  {row['model_name']:25}: {row['path_rmse']:.6f}")
        
        # Model ranking by correlation (higher is better)
        if 'path_mean_correlation' in successful_evals.columns:
            corr_ranking = successful_evals.nlargest(5, 'path_mean_correlation')[['model_name', 'path_mean_correlation']]
            print(f"\nTop Models by Correlation (higher is better):")
            for idx, row in corr_ranking.iterrows():
                print(f"  {row['model_name']:25}: {row['path_mean_correlation']:.6f}")


def run_comprehensive_evaluation(dataset_type: str = 'ornstein_uhlenbeck',
                               n_samples: int = 100,
                               save_dir: str = "evaluation_results") -> pd.DataFrame:
    """
    Run comprehensive evaluation of all available models.
    
    Args:
        dataset_type: Type of dataset for evaluation
        n_samples: Number of samples for evaluation
        save_dir: Directory to save results
        
    Returns:
        DataFrame with evaluation results
    """
    print("Comprehensive Model Evaluation")
    print("=" * 60)
    print(f"Dataset: {dataset_type}")
    print(f"Samples: {n_samples}")
    print(f"Save directory: {save_dir}")
    
    # Create evaluator
    evaluator = ModelEvaluator(save_dir)
    
    # Setup evaluation data
    evaluator.setup_evaluation_data(
        dataset_type=dataset_type,
        n_samples=n_samples,
        n_points=100
    )
    
    # Evaluate all available models
    results_df = evaluator.evaluate_all_available_models()
    
    # Print summary
    evaluator.print_evaluation_summary(results_df)
    
    print(f"\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    if not results_df.empty:
        print(f"✅ Evaluation successful!")
        print(f"   Results saved to: {save_dir}/model_evaluation_results.csv")
        print(f"   Metric descriptions: {save_dir}/metric_descriptions.csv")
        print(f"   {len(results_df)} models evaluated")
        
        # Quick comparison if multiple models
        if len(results_df) > 1:
            print(f"\nQuick Comparison:")
            for idx, row in results_df.iterrows():
                model_name = row.get('model_name', 'Unknown')
                rmse = row.get('path_rmse', float('nan'))
                params = row.get('comp_total_parameters', 0)
                time_ms = row.get('comp_mean_forward_time_ms', float('nan'))
                
                print(f"  {model_name:25}: RMSE={rmse:.4f}, Params={params:,}, Time={time_ms:.2f}ms")
    else:
        print(f"❌ No models evaluated successfully")
        print(f"   Check model implementations and dependencies")
    
    return results_df


def create_evaluation_report(results_df: pd.DataFrame, save_dir: str):
    """Create a detailed evaluation report."""
    if results_df.empty:
        return
    
    print(f"\nCreating detailed evaluation report...")
    
    # Filter successful evaluations
    successful_df = results_df[~results_df.get('evaluation_error', pd.Series()).notna()]
    
    if successful_df.empty:
        print("❌ No successful evaluations for report")
        return
    
    # Create summary statistics
    summary_stats = {}
    
    # Key metrics for comparison
    key_metrics = [
        'path_rmse', 'dist_ks_statistic', 'dist_wasserstein', 'dist_energy',
        'path_mean_correlation', 'sig_signature_mse', 'comp_mean_forward_time_ms'
    ]
    
    for metric in key_metrics:
        if metric in successful_df.columns:
            values = successful_df[metric].dropna()
            if len(values) > 0:
                summary_stats[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'count': int(len(values))
                }
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats).T
    summary_path = os.path.join(save_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_path)
    
    # Create ranking tables
    rankings = {}
    
    # RMSE ranking (lower is better)
    if 'path_rmse' in successful_df.columns:
        rmse_ranking = successful_df.nsmallest(10, 'path_rmse')[['model_name', 'path_rmse', 'comp_total_parameters']]
        rankings['rmse_ranking'] = rmse_ranking
    
    # Correlation ranking (higher is better)
    if 'path_mean_correlation' in successful_df.columns:
        corr_ranking = successful_df.nlargest(10, 'path_mean_correlation')[['model_name', 'path_mean_correlation', 'comp_total_parameters']]
        rankings['correlation_ranking'] = corr_ranking
    
    # Speed ranking (lower time is better)
    if 'comp_mean_forward_time_ms' in successful_df.columns:
        speed_ranking = successful_df.nsmallest(10, 'comp_mean_forward_time_ms')[['model_name', 'comp_mean_forward_time_ms', 'comp_total_parameters']]
        rankings['speed_ranking'] = speed_ranking
    
    # Save rankings
    for ranking_name, ranking_df in rankings.items():
        ranking_path = os.path.join(save_dir, f'{ranking_name}.csv')
        ranking_df.to_csv(ranking_path, index=False)
    
    print(f"✅ Evaluation report created:")
    print(f"   Summary: {summary_path}")
    for ranking_name in rankings.keys():
        print(f"   {ranking_name}: {save_dir}/{ranking_name}.csv")


def main():
    """Main evaluation script."""
    print("Signature-based Model Evaluation System")
    print("=" * 60)
    print("This script evaluates all available models using standardized metrics")
    print("and saves results to CSV for systematic comparison.")
    
    # Check available models
    print(f"\nAvailable Models:")
    print(f"  A1 (CannedNet + T-Statistic): {'✅' if A1_AVAILABLE else '❌'}")
    print(f"  A2 (CannedNet + Scoring): {'✅' if A2_AVAILABLE else '❌'}")
    
    if not (A1_AVAILABLE or A2_AVAILABLE):
        print(f"\n❌ No models available for evaluation")
        print(f"   Implement models first using the factory pattern")
        return False
    
    # Run evaluation
    try:
        results_df = run_comprehensive_evaluation(
            dataset_type='ornstein_uhlenbeck',
            n_samples=100,
            save_dir='evaluation_results'
        )
        
        # Create detailed report
        create_evaluation_report(results_df, 'evaluation_results')
        
        print(f"\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
        print(f"   All available models evaluated and compared")
        print(f"   Results available for analysis and visualization")
        print(f"   CSV files ready for further processing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n✅ Use evaluation_results/ for analysis and comparison")
        print(f"   Load CSV files for visualization and statistical analysis")
    else:
        print(f"\n❌ Fix implementation issues and retry evaluation")
    
    exit(0 if success else 1)

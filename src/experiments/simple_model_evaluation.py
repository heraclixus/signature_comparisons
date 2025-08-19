"""
Simple Model Evaluation Script

This evaluates available models using standardized metrics
and saves results to CSV for analysis.
"""

import torch
import torch.utils.data as torchdata
import numpy as np
import pandas as pd
import time
import sys
import os
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset import generative_model

# Import available models
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

try:
    from models.implementations.a3_canned_mmd import create_a3_model
    A3_AVAILABLE = True
except ImportError:
    A3_AVAILABLE = False


def compute_evaluation_metrics(generated_samples: torch.Tensor,
                             ground_truth_samples: torch.Tensor,
                             model_name: str) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        generated_samples: Generated samples from model
        ground_truth_samples: Ground truth target samples
        model_name: Name of the model
        
    Returns:
        Dictionary with computed metrics
    """
    # Convert to numpy
    gen_np = generated_samples.detach().cpu().numpy()
    gt_np = ground_truth_samples.detach().cpu().numpy()
    
    # Handle shape mismatches
    if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
        # Ground truth has (batch, channels, time), generated has (batch, time)
        gt_np = gt_np[:, 1, :]  # Extract value dimension
    
    # Match number of samples
    min_samples = min(gen_np.shape[0], gt_np.shape[0])
    gen_np = gen_np[:min_samples]
    gt_np = gt_np[:min_samples]
    
    # Match sequence length
    if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:, :min_length]
        gt_np = gt_np[:, :min_length]
    
    metrics = {'model_name': model_name}
    
    try:
        # 1. RMSE (Primary metric)
        rmse = np.sqrt(mean_squared_error(gen_np.flatten(), gt_np.flatten()))
        metrics['rmse'] = float(rmse)
        
        # 2. Distributional metrics
        gen_flat = gen_np.flatten()
        gt_flat = gt_np.flatten()
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(gen_flat, gt_flat)
        metrics['ks_statistic'] = float(ks_statistic)
        metrics['ks_p_value'] = float(ks_p_value)
        metrics['distributions_similar'] = ks_p_value > 0.05
        
        # Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(gen_flat, gt_flat)
        metrics['wasserstein_distance'] = float(wasserstein_dist)
        
        # 3. Moment matching
        gen_mean, gen_std = np.mean(gen_flat), np.std(gen_flat)
        gt_mean, gt_std = np.mean(gt_flat), np.std(gt_flat)
        
        metrics['generated_mean'] = float(gen_mean)
        metrics['generated_std'] = float(gen_std)
        metrics['ground_truth_mean'] = float(gt_mean)
        metrics['ground_truth_std'] = float(gt_std)
        
        metrics['mean_relative_error'] = float(abs(gen_mean - gt_mean) / (abs(gt_mean) + 1e-8))
        metrics['std_relative_error'] = float(abs(gen_std - gt_std) / (abs(gt_std) + 1e-8))
        
        # Skewness and kurtosis
        gen_skew, gen_kurt = stats.skew(gen_flat), stats.kurtosis(gen_flat)
        gt_skew, gt_kurt = stats.skew(gt_flat), stats.kurtosis(gt_flat)
        
        metrics['generated_skewness'] = float(gen_skew)
        metrics['generated_kurtosis'] = float(gen_kurt)
        metrics['ground_truth_skewness'] = float(gt_skew)
        metrics['ground_truth_kurtosis'] = float(gt_kurt)
        
        # 4. Path-wise correlations
        correlations = []
        for i in range(min_samples):
            try:
                corr, _ = stats.pearsonr(gen_np[i], gt_np[i])
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                pass
        
        if correlations:
            metrics['mean_path_correlation'] = float(np.mean(correlations))
            metrics['std_path_correlation'] = float(np.std(correlations))
            metrics['min_path_correlation'] = float(np.min(correlations))
            metrics['max_path_correlation'] = float(np.max(correlations))
        else:
            metrics.update({
                'mean_path_correlation': 0.0,
                'std_path_correlation': 0.0,
                'min_path_correlation': 0.0,
                'max_path_correlation': 0.0
            })
        
        # 5. Range and variability
        metrics['generated_range'] = float(np.max(gen_np) - np.min(gen_np))
        metrics['ground_truth_range'] = float(np.max(gt_np) - np.min(gt_np))
        metrics['range_ratio'] = float(metrics['generated_range'] / (metrics['ground_truth_range'] + 1e-8))
        
        # 6. Sample quality
        metrics['generated_finite_ratio'] = float(np.sum(np.isfinite(gen_np)) / gen_np.size)
        metrics['ground_truth_finite_ratio'] = float(np.sum(np.isfinite(gt_np)) / gt_np.size)
        
        print(f"  ✅ Metrics computed for {model_name}")
        
    except Exception as e:
        print(f"  ❌ Metric computation failed for {model_name}: {e}")
        metrics['evaluation_error'] = str(e)
    
    return metrics


def evaluate_model_performance(model, model_name: str, test_input: torch.Tensor,
                              n_runs: int = 10) -> Dict[str, float]:
    """
    Evaluate computational performance of a model.
    
    Args:
        model: Model to evaluate
        model_name: Name of the model
        test_input: Test input tensor
        n_runs: Number of runs for timing
        
    Returns:
        Dictionary with performance metrics
    """
    model.eval()
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Timing analysis
    forward_times = []
    
    # Warm-up
    with torch.no_grad():
        for _ in range(3):
            _ = model(test_input)
    
    # Timed runs
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad():
            output = model(test_input)
        end_time = time.time()
        forward_times.append((end_time - start_time) * 1000)  # ms
    
    return {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'mean_forward_time_ms': float(np.mean(forward_times)),
        'std_forward_time_ms': float(np.std(forward_times)),
        'min_forward_time_ms': float(np.min(forward_times)),
        'max_forward_time_ms': float(np.max(forward_times))
    }


def run_model_evaluation():
    """Run evaluation of all available models."""
    print("Simple Model Evaluation")
    print("=" * 50)
    
    # Setup results directory
    os.makedirs('results/evaluation', exist_ok=True)
    
    # Setup evaluation data
    print("Setting up evaluation data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 64
    n_points = 100
    
    # Generate test data
    train_dataset = generative_model.get_noise(n_points=n_points, num_samples=n_samples)
    signals = generative_model.get_signal(num_samples=n_samples, n_points=n_points).tensors[0]
    
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=n_samples, shuffle=False, num_workers=0)
    example_batch, _ = next(iter(train_dataloader))
    
    print(f"Evaluation data: signals {signals.shape}, example {example_batch.shape}")
    
    # Store all results
    all_results = []
    
    # Evaluate A1 model
    if A1_AVAILABLE:
        print(f"\nEvaluating A1 Model...")
        try:
            torch.manual_seed(12345)
            a1_model = create_a1_final_model(example_batch, signals)
            
            # Generate samples
            with torch.no_grad():
                a1_samples = a1_model.generate_samples(32)
            
            print(f"A1 generated samples: {a1_samples.shape}")
            
            # Compute metrics
            a1_metrics = compute_evaluation_metrics(a1_samples, signals[:32], "A1_CannedNet_TStatistic")
            
            # Add performance metrics
            test_input = example_batch[:8]
            a1_performance = evaluate_model_performance(a1_model, "A1", test_input)
            a1_metrics.update(a1_performance)
            
            # Add model metadata
            a1_metrics.update({
                'generator_type': 'CannedNet',
                'loss_type': 'T-Statistic',
                'signature_method': 'Truncated',
                'experiment_id': 'A1',
                'implementation_status': 'validated'
            })
            
            all_results.append(a1_metrics)
            print(f"✅ A1 evaluation completed")
            
        except Exception as e:
            print(f"❌ A1 evaluation failed: {e}")
            all_results.append({
                'model_name': 'A1_CannedNet_TStatistic',
                'evaluation_error': str(e),
                'experiment_id': 'A1'
            })
    
    # Evaluate A2 model
    if A2_AVAILABLE:
        print(f"\nEvaluating A2 Model...")
        try:
            torch.manual_seed(12345)
            a2_model = create_a2_model(example_batch, signals)
            
            # Generate samples
            with torch.no_grad():
                a2_samples = a2_model.generate_samples(32)
            
            print(f"A2 generated samples: {a2_samples.shape}")
            
            # Compute metrics
            a2_metrics = compute_evaluation_metrics(a2_samples, signals[:32], "A2_CannedNet_Scoring")
            
            # Add performance metrics
            test_input = example_batch[:8]
            a2_performance = evaluate_model_performance(a2_model, "A2", test_input)
            a2_metrics.update(a2_performance)
            
            # Add model metadata
            a2_metrics.update({
                'generator_type': 'CannedNet',
                'loss_type': 'Signature_Scoring',
                'signature_method': 'Truncated',
                'experiment_id': 'A2',
                'implementation_status': 'validated'
            })
            
            all_results.append(a2_metrics)
            print(f"✅ A2 evaluation completed")
            
        except Exception as e:
            print(f"❌ A2 evaluation failed: {e}")
            all_results.append({
                'model_name': 'A2_CannedNet_Scoring',
                'evaluation_error': str(e),
                'experiment_id': 'A2'
            })
    
    # Evaluate A3 model
    if A3_AVAILABLE:
        print(f"\nEvaluating A3 Model...")
        try:
            torch.manual_seed(12345)  # Same initialization for fair comparison
            a3_model = create_a3_model(example_batch, signals)
            
            # Generate samples
            with torch.no_grad():
                a3_samples = a3_model.generate_samples(32)
            
            print(f"A3 generated samples: {a3_samples.shape}")
            
            # Compute metrics
            a3_metrics = compute_evaluation_metrics(a3_samples, signals[:32], "A3_CannedNet_MMD")
            
            # Add performance metrics
            test_input = example_batch[:8]
            a3_performance = evaluate_model_performance(a3_model, "A3", test_input)
            a3_metrics.update(a3_performance)
            
            # Add model metadata
            a3_metrics.update({
                'generator_type': 'CannedNet',
                'loss_type': 'MMD',
                'signature_method': 'Truncated',
                'experiment_id': 'A3',
                'implementation_status': 'new'
            })
            
            all_results.append(a3_metrics)
            print(f"✅ A3 evaluation completed")
            
        except Exception as e:
            print(f"❌ A3 evaluation failed: {e}")
            all_results.append({
                'model_name': 'A3_CannedNet_MMD',
                'evaluation_error': str(e),
                'experiment_id': 'A3'
            })
    
    # Create results DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        csv_path = 'results/evaluation/model_comparison_results.csv'
        results_df.to_csv(csv_path, index=False)
        
        print(f"\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Results saved to: {csv_path}")
        print(f"Models evaluated: {len(all_results)}")
        
        # Print summary
        successful_results = [r for r in all_results if 'evaluation_error' not in r]
        
        if successful_results:
            print(f"\nSuccessful evaluations: {len(successful_results)}")
            
            print(f"\nKey Metrics Summary:")
            print(f"{'Model':<25} {'RMSE':<10} {'KS-Stat':<10} {'Corr':<10} {'Params':<10} {'Time(ms)':<10}")
            print("-" * 75)
            
            for result in successful_results:
                model_name = result.get('model_name', 'Unknown')[:24]
                rmse = result.get('rmse', float('nan'))
                ks_stat = result.get('ks_statistic', float('nan'))
                corr = result.get('mean_path_correlation', float('nan'))
                params = result.get('total_parameters', 0)
                time_ms = result.get('mean_forward_time_ms', float('nan'))
                
                print(f"{model_name:<25} {rmse:<10.4f} {ks_stat:<10.4f} {corr:<10.4f} {params:<10,} {time_ms:<10.2f}")
            
            # Best model by RMSE
            best_rmse_model = min(successful_results, key=lambda x: x.get('rmse', float('inf')))
            print(f"\n🏆 Best RMSE: {best_rmse_model['model_name']} ({best_rmse_model['rmse']:.4f})")
            
            # Best model by correlation
            best_corr_model = max(successful_results, key=lambda x: x.get('mean_path_correlation', -1))
            print(f"🏆 Best Correlation: {best_corr_model['model_name']} ({best_corr_model['mean_path_correlation']:.4f})")
            
            # Performance comparison
            if len(successful_results) > 1:
                print(f"\nModel Comparison:")
                for i, result1 in enumerate(successful_results):
                    for j, result2 in enumerate(successful_results[i+1:], i+1):
                        name1 = result1['model_name']
                        name2 = result2['model_name']
                        
                        rmse1 = result1.get('rmse', float('inf'))
                        rmse2 = result2.get('rmse', float('inf'))
                        
                        if rmse1 < rmse2:
                            improvement = (rmse2 - rmse1) / rmse2 * 100
                            print(f"  {name1} vs {name2}: {improvement:.1f}% better RMSE")
                        else:
                            improvement = (rmse1 - rmse2) / rmse1 * 100
                            print(f"  {name2} vs {name1}: {improvement:.1f}% better RMSE")
        
        return results_df
    
    else:
        print(f"\n❌ No models evaluated successfully")
        return pd.DataFrame()


def create_metric_descriptions():
    """Create CSV with metric descriptions."""
    descriptions = {
        'model_name': 'Name identifier of the model',
        'rmse': 'Root Mean Square Error between generated and ground truth (lower is better)',
        'ks_statistic': 'Kolmogorov-Smirnov test statistic (lower means more similar distributions)',
        'ks_p_value': 'KS test p-value (higher means distributions are more similar)',
        'distributions_similar': 'Whether distributions are statistically similar (True is better)',
        'wasserstein_distance': 'Wasserstein distance between distributions (lower is better)',
        'generated_mean': 'Mean of generated samples',
        'generated_std': 'Standard deviation of generated samples',
        'ground_truth_mean': 'Mean of ground truth samples',
        'ground_truth_std': 'Standard deviation of ground truth samples',
        'mean_relative_error': 'Relative error in mean (lower is better)',
        'std_relative_error': 'Relative error in standard deviation (lower is better)',
        'generated_skewness': 'Skewness of generated distribution',
        'generated_kurtosis': 'Kurtosis of generated distribution',
        'ground_truth_skewness': 'Skewness of ground truth distribution',
        'ground_truth_kurtosis': 'Kurtosis of ground truth distribution',
        'mean_path_correlation': 'Mean correlation between generated and ground truth paths (higher is better)',
        'std_path_correlation': 'Standard deviation of path correlations',
        'min_path_correlation': 'Minimum path correlation',
        'max_path_correlation': 'Maximum path correlation',
        'generated_range': 'Range (max-min) of generated samples',
        'ground_truth_range': 'Range (max-min) of ground truth samples',
        'range_ratio': 'Ratio of generated to ground truth range (closer to 1 is better)',
        'generated_finite_ratio': 'Proportion of finite values in generated samples (should be 1.0)',
        'ground_truth_finite_ratio': 'Proportion of finite values in ground truth (should be 1.0)',
        'total_parameters': 'Total number of model parameters',
        'trainable_parameters': 'Number of trainable model parameters',
        'mean_forward_time_ms': 'Mean forward pass time in milliseconds',
        'std_forward_time_ms': 'Standard deviation of forward pass times',
        'generator_type': 'Type of generator architecture used',
        'loss_type': 'Type of loss function used',
        'signature_method': 'Signature computation method used',
        'experiment_id': 'Experiment identifier (A1, A2, etc.)',
        'implementation_status': 'Status of implementation (validated, experimental, etc.)'
    }
    
    desc_df = pd.DataFrame([
        {'metric': k, 'description': v, 'type': 'lower_better' if 'error' in k or 'distance' in k or 'rmse' in k or 'ks_statistic' in k else 'higher_better' if 'correlation' in k or 'p_value' in k else 'informational'}
        for k, v in descriptions.items()
    ])
    
    desc_path = 'results/evaluation/metric_descriptions.csv'
    desc_df.to_csv(desc_path, index=False)
    print(f"Metric descriptions saved to: {desc_path}")


def main():
    """Main evaluation function."""
    print("Simple Model Evaluation System")
    print("=" * 60)
    print("Evaluating all available signature-based models")
    print("Metrics: RMSE, distributional similarity, path correlations")
    print("Results saved to CSV for analysis and visualization")
    
    # Check available models
    available_models = []
    if A1_AVAILABLE:
        available_models.append("A1")
    if A2_AVAILABLE:
        available_models.append("A2")
    
    print(f"\nAvailable models: {available_models}")
    
    if not available_models:
        print(f"❌ No models available for evaluation")
        print(f"   Implement models using the factory pattern first")
        return False
    
    # Run evaluation
    try:
        results_df = run_model_evaluation()
        
        # Create metric descriptions
        create_metric_descriptions()
        
        if not results_df.empty:
            print(f"\n🎉 EVALUATION COMPLETE!")
            print(f"   {len(results_df)} models evaluated")
            print(f"   Results saved to CSV for analysis")
            print(f"   Ready for visualization and comparison")
            
            print(f"\nNext steps:")
            print(f"   1. Load results/evaluation/model_comparison_results.csv")
            print(f"   2. Create visualizations and statistical analysis")
            print(f"   3. Compare model performance across metrics")
            print(f"   4. Use results to guide further model development")
            
            return True
        else:
            print(f"\n❌ No successful evaluations")
            return False
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

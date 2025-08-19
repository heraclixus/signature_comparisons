"""
Evaluation Metrics for Signature-based Models

This module defines standardized evaluation metrics for comparing
different signature-based time series generation models.
"""

import torch
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, List, Tuple, Optional
import warnings


class DistributionalMetrics:
    """Distributional comparison metrics between generated and ground truth data."""
    
    @staticmethod
    def kolmogorov_smirnov_test(generated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Kolmogorov-Smirnov test for distribution similarity.
        
        Args:
            generated: Generated samples, shape (n_samples, n_features) or flattened
            ground_truth: Ground truth samples, same format
            
        Returns:
            Dictionary with KS statistic and p-value
        """
        gen_flat = generated.flatten()
        gt_flat = ground_truth.flatten()
        
        ks_statistic, p_value = stats.ks_2samp(gen_flat, gt_flat)
        
        return {
            'ks_statistic': float(ks_statistic),
            'ks_p_value': float(p_value),
            'ks_significant': p_value < 0.05  # Reject null hypothesis (distributions differ)
        }
    
    @staticmethod
    def wasserstein_distance(generated: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Wasserstein (Earth Mover's) distance between distributions.
        
        Args:
            generated: Generated samples
            ground_truth: Ground truth samples
            
        Returns:
            Wasserstein distance
        """
        gen_flat = generated.flatten()
        gt_flat = ground_truth.flatten()
        
        return float(stats.wasserstein_distance(gen_flat, gt_flat))
    
    @staticmethod
    def moment_matching(generated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compare statistical moments between distributions.
        
        Args:
            generated: Generated samples
            ground_truth: Ground truth samples
            
        Returns:
            Dictionary with moment differences
        """
        gen_flat = generated.flatten()
        gt_flat = ground_truth.flatten()
        
        # Compute moments
        gen_moments = {
            'mean': np.mean(gen_flat),
            'std': np.std(gen_flat),
            'skewness': stats.skew(gen_flat),
            'kurtosis': stats.kurtosis(gen_flat)
        }
        
        gt_moments = {
            'mean': np.mean(gt_flat),
            'std': np.std(gt_flat),
            'skewness': stats.skew(gt_flat),
            'kurtosis': stats.kurtosis(gt_flat)
        }
        
        # Compute relative differences
        moment_diffs = {}
        for moment_name in gen_moments.keys():
            gen_val = gen_moments[moment_name]
            gt_val = gt_moments[moment_name]
            
            if abs(gt_val) > 1e-8:
                rel_diff = abs(gen_val - gt_val) / abs(gt_val)
            else:
                rel_diff = abs(gen_val - gt_val)
            
            moment_diffs[f'{moment_name}_diff'] = float(rel_diff)
            moment_diffs[f'{moment_name}_generated'] = float(gen_val)
            moment_diffs[f'{moment_name}_ground_truth'] = float(gt_val)
        
        return moment_diffs
    
    @staticmethod
    def energy_distance(generated: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Energy distance between distributions.
        
        Args:
            generated: Generated samples
            ground_truth: Ground truth samples
            
        Returns:
            Energy distance
        """
        gen_flat = generated.flatten()
        gt_flat = ground_truth.flatten()
        
        # Subsample for computational efficiency
        max_samples = 1000
        if len(gen_flat) > max_samples:
            idx_gen = np.random.choice(len(gen_flat), max_samples, replace=False)
            gen_flat = gen_flat[idx_gen]
        
        if len(gt_flat) > max_samples:
            idx_gt = np.random.choice(len(gt_flat), max_samples, replace=False)
            gt_flat = gt_flat[idx_gt]
        
        # Energy distance computation
        # E(X,Y) = 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
        
        # Cross-distance: E[|X-Y|]
        cross_dist = np.mean(np.abs(gen_flat[:, None] - gt_flat[None, :]))
        
        # Self-distances
        gen_self_dist = np.mean(np.abs(gen_flat[:, None] - gen_flat[None, :]))
        gt_self_dist = np.mean(np.abs(gt_flat[:, None] - gt_flat[None, :]))
        
        energy_dist = 2 * cross_dist - gen_self_dist - gt_self_dist
        
        return float(energy_dist)


class PathMetrics:
    """Path-specific metrics for time series evaluation."""
    
    @staticmethod
    def rmse(generated: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Root Mean Square Error between generated and ground truth paths.
        
        Args:
            generated: Generated paths, shape (n_samples, n_timepoints)
            ground_truth: Ground truth paths, same shape
            
        Returns:
            RMSE value
        """
        if generated.shape != ground_truth.shape:
            # Try to match shapes
            min_samples = min(generated.shape[0], ground_truth.shape[0])
            generated = generated[:min_samples]
            ground_truth = ground_truth[:min_samples]
            
            if len(generated.shape) > 2:
                generated = generated.reshape(generated.shape[0], -1)
            if len(ground_truth.shape) > 2:
                ground_truth = ground_truth.reshape(ground_truth.shape[0], -1)
            
            min_features = min(generated.shape[1], ground_truth.shape[1])
            generated = generated[:, :min_features]
            ground_truth = ground_truth[:, :min_features]
        
        return float(np.sqrt(mean_squared_error(generated.flatten(), ground_truth.flatten())))
    
    @staticmethod
    def path_wise_correlation(generated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Compute path-wise correlations between generated and ground truth.
        
        Args:
            generated: Generated paths
            ground_truth: Ground truth paths
            
        Returns:
            Dictionary with correlation statistics
        """
        if generated.shape != ground_truth.shape:
            min_samples = min(generated.shape[0], ground_truth.shape[0])
            generated = generated[:min_samples]
            ground_truth = ground_truth[:min_samples]
        
        correlations = []
        
        for i in range(min(generated.shape[0], ground_truth.shape[0])):
            gen_path = generated[i].flatten()
            gt_path = ground_truth[i].flatten()
            
            if len(gen_path) == len(gt_path):
                try:
                    corr, _ = stats.pearsonr(gen_path, gt_path)
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    pass
        
        if correlations:
            return {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'min_correlation': float(np.min(correlations)),
                'max_correlation': float(np.max(correlations)),
                'n_paths': len(correlations)
            }
        else:
            return {
                'mean_correlation': 0.0,
                'std_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'n_paths': 0
            }
    
    @staticmethod
    def path_smoothness(paths: np.ndarray) -> Dict[str, float]:
        """
        Measure path smoothness using finite differences.
        
        Args:
            paths: Path data, shape (n_samples, n_timepoints)
            
        Returns:
            Dictionary with smoothness metrics
        """
        if len(paths.shape) == 2:
            # Compute first and second differences
            first_diff = np.diff(paths, axis=1)
            second_diff = np.diff(first_diff, axis=1)
            
            # Smoothness metrics
            first_diff_var = np.var(first_diff, axis=1).mean()
            second_diff_var = np.var(second_diff, axis=1).mean()
            
            return {
                'first_diff_variance': float(first_diff_var),
                'second_diff_variance': float(second_diff_var),
                'smoothness_score': float(1.0 / (1.0 + second_diff_var))  # Higher is smoother
            }
        else:
            return {'first_diff_variance': 0.0, 'second_diff_variance': 0.0, 'smoothness_score': 0.0}


class SignatureMetrics:
    """Signature-specific evaluation metrics."""
    
    @staticmethod
    def signature_distance(generated: torch.Tensor, ground_truth: torch.Tensor, 
                          signature_depth: int = 4) -> Dict[str, float]:
        """
        Compute distance between signature features.
        
        Args:
            generated: Generated paths as tensors
            ground_truth: Ground truth paths as tensors
            signature_depth: Depth for signature computation
            
        Returns:
            Dictionary with signature-based distances
        """
        try:
            from signatures import TruncatedSignature
            
            # Create signature transform
            sig_transform = TruncatedSignature(depth=signature_depth)
            
            # Ensure 3D format for signature computation
            if len(generated.shape) == 2:
                generated = generated.unsqueeze(1)
            if len(ground_truth.shape) == 2:
                ground_truth = ground_truth.unsqueeze(1)
            
            # Compute signatures
            gen_sigs = sig_transform(generated)
            gt_sigs = sig_transform(ground_truth)
            
            # Signature distances
            sig_mse = torch.nn.functional.mse_loss(gen_sigs, gt_sigs[:gen_sigs.size(0)])
            sig_l1 = torch.nn.functional.l1_loss(gen_sigs, gt_sigs[:gen_sigs.size(0)])
            
            return {
                'signature_mse': float(sig_mse.item()),
                'signature_l1': float(sig_l1.item()),
                'signature_depth': signature_depth
            }
            
        except Exception as e:
            warnings.warn(f"Signature distance computation failed: {e}")
            return {
                'signature_mse': float('inf'),
                'signature_l1': float('inf'),
                'signature_depth': signature_depth
            }


class ComputationalMetrics:
    """Computational performance metrics."""
    
    @staticmethod
    def timing_analysis(model, test_input: torch.Tensor, n_runs: int = 10) -> Dict[str, float]:
        """
        Analyze computational performance of model.
        
        Args:
            model: Model to analyze
            test_input: Test input tensor
            n_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing metrics
        """
        model.eval()
        
        # Warm-up runs
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_input)
        
        # Timed runs
        forward_times = []
        
        for _ in range(n_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(test_input)
            end_time = time.time()
            
            forward_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return {
            'mean_forward_time_ms': float(np.mean(forward_times)),
            'std_forward_time_ms': float(np.std(forward_times)),
            'min_forward_time_ms': float(np.min(forward_times)),
            'max_forward_time_ms': float(np.max(forward_times)),
            'n_runs': n_runs
        }
    
    @staticmethod
    def memory_usage(model) -> Dict[str, Any]:
        """
        Analyze memory usage of model.
        
        Args:
            model: Model to analyze
            
        Returns:
            Dictionary with memory metrics
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'estimated_memory_mb': float(param_memory_mb),
            'parameter_efficiency': float(trainable_params / total_params) if total_params > 0 else 0.0
        }


def compute_comprehensive_metrics(generated_samples: torch.Tensor,
                                ground_truth_samples: torch.Tensor,
                                model: torch.nn.Module,
                                test_input: torch.Tensor,
                                model_name: str = "Unknown") -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for a model.
    
    Args:
        generated_samples: Generated samples from model
        ground_truth_samples: Ground truth target samples
        model: The model being evaluated
        test_input: Test input for performance analysis
        model_name: Name of the model for identification
        
    Returns:
        Dictionary with all computed metrics
    """
    print(f"Computing metrics for {model_name}...")
    
    # Convert to numpy for statistical computations
    gen_np = generated_samples.detach().cpu().numpy()
    gt_np = ground_truth_samples.detach().cpu().numpy()
    
    # Ensure compatible shapes for comparison
    if gen_np.shape != gt_np.shape:
        print(f"  Shape mismatch: {gen_np.shape} vs {gt_np.shape}, attempting to match...")
        
        # Match number of samples
        min_samples = min(gen_np.shape[0], gt_np.shape[0])
        gen_np = gen_np[:min_samples]
        gt_np = gt_np[:min_samples]
        
        # Handle different dimensionalities
        if len(gen_np.shape) != len(gt_np.shape):
            if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
                # Ground truth has extra dimension, extract relevant part
                gt_np = gt_np[:, 1, :gen_np.shape[1]]  # Extract value dimension
            elif len(gen_np.shape) == 3 and len(gt_np.shape) == 2:
                # Generated has extra dimension
                gen_np = gen_np.reshape(gen_np.shape[0], -1)
        
        # Match feature dimensions
        if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
            min_features = min(gen_np.shape[1], gt_np.shape[1])
            gen_np = gen_np[:, :min_features]
            gt_np = gt_np[:, :min_features]
    
    metrics = {
        'model_name': model_name,
        'generated_shape': list(generated_samples.shape),
        'ground_truth_shape': list(ground_truth_samples.shape),
        'processed_shape_gen': list(gen_np.shape),
        'processed_shape_gt': list(gt_np.shape)
    }
    
    try:
        # 1. Distributional Metrics
        print(f"  Computing distributional metrics...")
        dist_metrics = DistributionalMetrics()
        
        # KS test
        ks_results = dist_metrics.kolmogorov_smirnov_test(gen_np, gt_np)
        metrics.update({f'dist_{k}': v for k, v in ks_results.items()})
        
        # Wasserstein distance
        wasserstein_dist = dist_metrics.wasserstein_distance(gen_np, gt_np)
        metrics['dist_wasserstein'] = wasserstein_dist
        
        # Moment matching
        moment_results = dist_metrics.moment_matching(gen_np, gt_np)
        metrics.update({f'dist_{k}': v for k, v in moment_results.items()})
        
        # Energy distance
        energy_dist = dist_metrics.energy_distance(gen_np, gt_np)
        metrics['dist_energy'] = energy_dist
        
    except Exception as e:
        print(f"  Warning: Distributional metrics failed: {e}")
        metrics.update({
            'dist_ks_statistic': float('nan'),
            'dist_wasserstein': float('nan'),
            'dist_energy': float('nan')
        })
    
    try:
        # 2. Path Metrics
        print(f"  Computing path metrics...")
        path_metrics = PathMetrics()
        
        # RMSE
        rmse = path_metrics.rmse(gen_np, gt_np)
        metrics['path_rmse'] = rmse
        
        # Path-wise correlation
        corr_results = path_metrics.path_wise_correlation(gen_np, gt_np)
        metrics.update({f'path_{k}': v for k, v in corr_results.items()})
        
        # Path smoothness
        gen_smoothness = path_metrics.path_smoothness(gen_np)
        gt_smoothness = path_metrics.path_smoothness(gt_np)
        
        metrics.update({f'smoothness_gen_{k}': v for k, v in gen_smoothness.items()})
        metrics.update({f'smoothness_gt_{k}': v for k, v in gt_smoothness.items()})
        
    except Exception as e:
        print(f"  Warning: Path metrics failed: {e}")
        metrics.update({
            'path_rmse': float('nan'),
            'path_mean_correlation': float('nan')
        })
    
    try:
        # 3. Signature Metrics
        print(f"  Computing signature metrics...")
        sig_metrics = SignatureMetrics()
        
        sig_results = sig_metrics.signature_distance(generated_samples, ground_truth_samples)
        metrics.update({f'sig_{k}': v for k, v in sig_results.items()})
        
    except Exception as e:
        print(f"  Warning: Signature metrics failed: {e}")
        metrics.update({
            'sig_signature_mse': float('nan'),
            'sig_signature_l1': float('nan')
        })
    
    try:
        # 4. Computational Metrics
        print(f"  Computing computational metrics...")
        comp_metrics = ComputationalMetrics()
        
        # Timing analysis
        timing_results = comp_metrics.timing_analysis(model, test_input)
        metrics.update({f'comp_{k}': v for k, v in timing_results.items()})
        
        # Memory usage
        memory_results = comp_metrics.memory_usage(model)
        metrics.update({f'comp_{k}': v for k, v in memory_results.items()})
        
    except Exception as e:
        print(f"  Warning: Computational metrics failed: {e}")
        metrics.update({
            'comp_mean_forward_time_ms': float('nan'),
            'comp_total_parameters': 0
        })
    
    print(f"  ✅ Metrics computed for {model_name}")
    return metrics


def get_metric_descriptions() -> Dict[str, str]:
    """Get descriptions of all metrics for documentation."""
    return {
        # Distributional metrics
        'dist_ks_statistic': 'Kolmogorov-Smirnov test statistic (lower is better)',
        'dist_ks_p_value': 'KS test p-value (higher means distributions are similar)',
        'dist_ks_significant': 'Whether distributions are significantly different (False is better)',
        'dist_wasserstein': 'Wasserstein distance between distributions (lower is better)',
        'dist_energy': 'Energy distance between distributions (lower is better)',
        'dist_mean_diff': 'Relative difference in means (lower is better)',
        'dist_std_diff': 'Relative difference in standard deviations (lower is better)',
        'dist_skewness_diff': 'Relative difference in skewness (lower is better)',
        'dist_kurtosis_diff': 'Relative difference in kurtosis (lower is better)',
        
        # Path metrics
        'path_rmse': 'Root Mean Square Error between paths (lower is better)',
        'path_mean_correlation': 'Mean path-wise correlation (higher is better)',
        'path_std_correlation': 'Std of path-wise correlations (lower is better)',
        'path_n_paths': 'Number of paths used in correlation analysis',
        
        # Signature metrics
        'sig_signature_mse': 'MSE between signature features (lower is better)',
        'sig_signature_l1': 'L1 distance between signature features (lower is better)',
        'sig_signature_depth': 'Signature depth used for computation',
        
        # Smoothness metrics
        'smoothness_gen_smoothness_score': 'Generated path smoothness (higher is better)',
        'smoothness_gt_smoothness_score': 'Ground truth path smoothness (reference)',
        
        # Computational metrics
        'comp_mean_forward_time_ms': 'Mean forward pass time in milliseconds',
        'comp_total_parameters': 'Total number of model parameters',
        'comp_trainable_parameters': 'Number of trainable parameters',
        'comp_estimated_memory_mb': 'Estimated memory usage in MB'
    }

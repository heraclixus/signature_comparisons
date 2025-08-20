"""
Enhanced Model Evaluation with Trajectory Visualization and Empirical Std Analysis

This script provides comprehensive evaluation including:
1. 20 trajectory samples per model for visualization
2. Empirical standard deviation analysis at each time step
3. Clean, sorted visualizations without unnecessary metrics
4. Comparison plots for trajectory and std analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, Any, List, Tuple
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.generative_model import get_signal
from utils.model_checkpoint import create_checkpoint_manager


def compute_core_metrics(generated: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, float]:
    """Compute core evaluation metrics."""
    gen_np = generated.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Handle shape alignment
    if len(gt_np.shape) == 3 and len(gen_np.shape) == 2:
        gt_np = gt_np[:, 1, :]  # Extract value dimension
    
    # Match dimensions
    min_samples = min(gen_np.shape[0], gt_np.shape[0])
    gen_np = gen_np[:min_samples]
    gt_np = gt_np[:min_samples]
    
    if len(gen_np.shape) == 2 and len(gt_np.shape) == 2:
        min_length = min(gen_np.shape[1], gt_np.shape[1])
        gen_np = gen_np[:, :min_length]
        gt_np = gt_np[:, :min_length]
    
    # Core metrics
    gen_flat = gen_np.flatten()
    gt_flat = gt_np.flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((gen_flat - gt_flat) ** 2))
    
    # KS test for distribution similarity
    ks_statistic, ks_p_value = stats.ks_2samp(gen_flat, gt_flat)
    
    # Wasserstein distance
    wasserstein_dist = stats.wasserstein_distance(gen_flat, gt_flat)
    
    return {
        'rmse': rmse,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'wasserstein_distance': wasserstein_dist,
        'distributions_similar': ks_p_value > 0.05
    }


def compute_empirical_std_analysis(trajectories: torch.Tensor, ground_truth: torch.Tensor) -> Dict[str, Any]:
    """
    Compute empirical standard deviation analysis at each time step.
    
    Args:
        trajectories: Generated trajectories, shape (n_traj, time_steps)
        ground_truth: Ground truth data, shape (batch, channels, time_steps)
        
    Returns:
        Dictionary with std analysis results
    """
    traj_np = trajectories.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()
    
    # Extract ground truth values (remove time dimension)
    if len(gt_np.shape) == 3:
        gt_values = gt_np[:, 1, :]  # Shape: (batch, time_steps)
    else:
        gt_values = gt_np
    
    # Compute empirical std at each time step
    gen_std_per_timestep = np.std(traj_np, axis=0)  # Shape: (time_steps,)
    gt_std_per_timestep = np.std(gt_values, axis=0)  # Shape: (time_steps,)
    
    # Align lengths
    min_length = min(len(gen_std_per_timestep), len(gt_std_per_timestep))
    gen_std_per_timestep = gen_std_per_timestep[:min_length]
    gt_std_per_timestep = gt_std_per_timestep[:min_length]
    
    # Compute std matching metrics
    std_rmse = np.sqrt(np.mean((gen_std_per_timestep - gt_std_per_timestep) ** 2))
    std_correlation = np.corrcoef(gen_std_per_timestep, gt_std_per_timestep)[0, 1]
    std_mean_error = np.mean(np.abs(gen_std_per_timestep - gt_std_per_timestep))
    
    return {
        'empirical_std_generated': gen_std_per_timestep,
        'empirical_std_ground_truth': gt_std_per_timestep,
        'std_rmse': std_rmse,
        'std_correlation': std_correlation,
        'std_mean_absolute_error': std_mean_error,
        'time_steps': np.arange(min_length)
    }


def evaluate_models_for_dataset(dataset_name: str):
    """Enhanced evaluation of all trained models for a specific dataset."""
    print(f"\n{'='*70}")
    print(f"🎯 Enhanced Model Evaluation for {dataset_name.upper()} Dataset")
    print(f"{'='*70}")
    
    # Setup dataset-specific checkpoint manager
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}')
    
    # Get list of trained models for this dataset
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print(f"❌ No trained models found for {dataset_name}")
        return None, None
    
    print(f"Found {len(available_models)} trained models for {dataset_name}:")
    checkpoint_manager.print_available_models()
    
    # Setup evaluation data (use appropriate dataset)
    print(f"\nSetting up evaluation data for {dataset_name}...")
    if dataset_name == 'ou_process':
        # Use original OU process data
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
    else:
        # For other datasets, we should use the appropriate dataset
        # For now, use OU process as baseline (can be enhanced later)
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
        print(f"   Note: Using OU process data as evaluation baseline for {dataset_name}")
    
    print(f"Evaluation data: {eval_data.shape}")
    
    # Evaluate each model
    results = []
    trajectory_data = {}  # Store trajectory data for visualization
    
    for model_id in available_models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_id}")
        print(f"{'='*50}")
        
        try:
            # Load model
            model = checkpoint_manager.load_model(model_id)
            if model is None:
                print(f"❌ Failed to load {model_id}")
                continue
            
            checkpoint_info = checkpoint_manager.get_checkpoint_info(model_id)
            print(f"✅ Model {model_id} loaded successfully")
            print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
            print(f"Training: Epoch {checkpoint_info['epoch']}, Loss {checkpoint_info['loss']:.6f}")
            
            # Generate standard evaluation samples
            model.eval()
            with torch.no_grad():
                samples = model.generate_samples(64)
                print(f"Generated samples: {samples.shape}")
                
                # Generate 20 trajectories for visualization
                trajectories = model.generate_samples(20)
                print(f"Generated trajectories for visualization: {trajectories.shape}")
            
            # Compute core metrics
            metrics = compute_core_metrics(samples, eval_data)
            
            # Compute empirical std analysis
            std_analysis = compute_empirical_std_analysis(trajectories, eval_data)
            
            # Combine results
            result = {
                'model_id': model_id,
                'training_epoch': checkpoint_info['epoch'],
                'training_loss': checkpoint_info['loss'],
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'model_class': type(model).__name__,
                **metrics,
                'std_rmse': std_analysis['std_rmse'],
                'std_correlation': std_analysis['std_correlation'],
                'std_mean_absolute_error': std_analysis['std_mean_absolute_error']
            }
            
            results.append(result)
            
            # Store trajectory data for visualization
            trajectory_data[model_id] = {
                'trajectories': trajectories.detach().cpu().numpy(),
                'empirical_std_generated': std_analysis['empirical_std_generated'],
                'empirical_std_ground_truth': std_analysis['empirical_std_ground_truth'],
                'time_steps': std_analysis['time_steps']
            }
            
            print(f"✅ {model_id} evaluation completed")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
            print(f"   Std RMSE: {std_analysis['std_rmse']:.4f}")
            print(f"   Std Correlation: {std_analysis['std_correlation']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            continue
    
    if not results:
        print("❌ No models successfully evaluated")
        return
    
    # Save results to dataset-specific directory
    results_df = pd.DataFrame(results)
    save_dir = f"results/{dataset_name}/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'enhanced_models_evaluation.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"ENHANCED EVALUATION COMPLETE FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Models evaluated: {len(results)}")
    
    # Create enhanced visualizations
    create_enhanced_visualizations(results_df, trajectory_data, save_dir, dataset_name)
    
    return results_df, trajectory_data


def evaluate_all_models_enhanced():
    """Enhanced evaluation of all trained models across all datasets, including adversarial variants."""
    print("🎯 Enhanced Model Evaluation with Trajectory Analysis")
    print("=" * 70)
    print("Evaluating models across all available datasets (including adversarial variants)")
    
    # Define available datasets
    base_datasets = ['ou_process', 'heston', 'rbergomi', 'brownian', 'fbm_h03', 'fbm_h04', 'fbm_h06', 'fbm_h07']
    
    all_results = {}
    
    for dataset_name in base_datasets:
        # Evaluate both non-adversarial and adversarial models for each dataset
        dataset_results = {}
        
        # 1. Evaluate non-adversarial models
        dataset_dir = f'results/{dataset_name}'
        if os.path.exists(dataset_dir):
            print(f"\n📊 Evaluating non-adversarial models for {dataset_name}...")
            results_df, trajectory_data = evaluate_models_for_dataset(dataset_name)
            if results_df is not None:
                dataset_results['non_adversarial'] = {
                    'results_df': results_df,
                    'trajectory_data': trajectory_data
                }
        
        # 2. Evaluate adversarial models
        adv_dataset_dir = f'results/{dataset_name}_adversarial'
        if os.path.exists(adv_dataset_dir):
            print(f"\n⚔️ Evaluating adversarial models for {dataset_name}...")
            adv_results_df, adv_trajectory_data = evaluate_adversarial_models_for_dataset(dataset_name)
            if adv_results_df is not None:
                dataset_results['adversarial'] = {
                    'results_df': adv_results_df,
                    'trajectory_data': adv_trajectory_data
                }
        
        # 3. Evaluate latent SDE models
        latent_sde_dataset_dir = f'results/{dataset_name}_latent_sde'
        if os.path.exists(latent_sde_dataset_dir):
            print(f"\n🧠 Evaluating latent SDE models for {dataset_name}...")
            latent_sde_results_df, latent_sde_trajectory_data = evaluate_latent_sde_models_for_dataset(dataset_name)
            if latent_sde_results_df is not None:
                dataset_results['latent_sde'] = {
                    'results_df': latent_sde_results_df,
                    'trajectory_data': latent_sde_trajectory_data
                }
        
        # Store results if we have any data for this dataset
        if dataset_results:
            all_results[dataset_name] = dataset_results
    
    if not all_results:
        print("\n❌ No datasets with trained models found")
        return None, None
    
    # Create comprehensive comparison analysis
    create_adversarial_comparison_analysis(all_results)
    
    print(f"\n{'='*70}")
    print("🎉 MULTI-DATASET EVALUATION COMPLETE (Including Adversarial)")
    print(f"{'='*70}")
    print(f"Datasets evaluated: {list(all_results.keys())}")
    for dataset_name in all_results.keys():
        dataset_data = all_results[dataset_name]
        non_adv_count = len(dataset_data.get('non_adversarial', {}).get('results_df', []))
        adv_count = len(dataset_data.get('adversarial', {}).get('results_df', []))
        latent_sde_count = len(dataset_data.get('latent_sde', {}).get('results_df', []))
        print(f"   {dataset_name}: {non_adv_count} non-adversarial, {adv_count} adversarial, {latent_sde_count} latent SDE models")
    
    return all_results


def evaluate_adversarial_models_for_dataset(dataset_name: str):
    """Evaluate adversarial models for a specific dataset."""
    print(f"\n{'='*70}")
    print(f"⚔️ Enhanced Adversarial Model Evaluation for {dataset_name.upper()} Dataset")
    print(f"{'='*70}")
    
    # Setup adversarial checkpoint manager
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}_adversarial')
    
    # Get list of trained adversarial models
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print(f"❌ No trained adversarial models found for {dataset_name}")
        return None, None
    
    print(f"Found {len(available_models)} trained adversarial models for {dataset_name}:")
    checkpoint_manager.print_available_models()
    
    # Setup evaluation data (same as non-adversarial)
    print(f"\nSetting up evaluation data for {dataset_name}_adversarial...")
    if dataset_name == 'ou_process':
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
    else:
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
        print(f"   Note: Using OU process data as evaluation baseline for {dataset_name}")
    
    print(f"Evaluation data: {eval_data.shape}")
    
    # Import adversarial training functions
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from adversarial_training import create_adversarial_model
    
    # Evaluate each adversarial model
    results = []
    trajectory_data = {}
    
    for model_id in available_models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_id}")
        print(f"{'='*50}")
        
        try:
            # Extract base model ID
            base_model_id = model_id.replace('_ADV', '')
            
            # Recreate adversarial model (since they can't be loaded directly)
            print(f"Recreating adversarial model {model_id} from base {base_model_id}...")
            
            # Setup data for model creation
            example_batch = torch.randn(8, 2, 100)
            signals = eval_data
            
            # Create adversarial model
            adversarial_model = create_adversarial_model(
                base_model_id=base_model_id,
                example_batch=example_batch,
                real_data=signals,
                adversarial=True,
                memory_efficient=True
            )
            
            # Load the trained weights
            model_dir = f'results/{dataset_name}_adversarial/trained_models/{model_id}'
            model_path = os.path.join(model_dir, 'model.pth')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                adversarial_model.load_state_dict(state_dict)
                print(f"✅ Loaded trained weights for {model_id}")
            else:
                print(f"⚠️ No trained weights found for {model_id}, using initialized model")
            
            checkpoint_info = checkpoint_manager.get_checkpoint_info(model_id)
            print(f"✅ Model {model_id} loaded successfully")
            print(f"Model loaded: {sum(p.numel() for p in adversarial_model.parameters()):,} parameters")
            print(f"Training: Epoch {checkpoint_info['epoch']}, Loss {checkpoint_info['loss']:.6f}")
            
            # Generate evaluation samples
            adversarial_model.eval()
            with torch.no_grad():
                samples = adversarial_model.generate_samples(64)
                print(f"Generated samples: {samples.shape}")
                
                # Generate 20 trajectories for visualization
                trajectories = adversarial_model.generate_samples(20)
                print(f"Generated trajectories for visualization: {trajectories.shape}")
            
            # Compute core metrics
            metrics = compute_core_metrics(samples, eval_data)
            
            # Compute empirical std analysis
            std_analysis = compute_empirical_std_analysis(trajectories, eval_data)
            
            # Combine results
            result = {
                'model_id': model_id,
                'training_epoch': checkpoint_info['epoch'],
                'training_loss': checkpoint_info['loss'],
                'total_parameters': sum(p.numel() for p in adversarial_model.parameters()),
                'model_class': type(adversarial_model).__name__,
                **metrics,
                'std_rmse': std_analysis['std_rmse'],
                'std_correlation': std_analysis['std_correlation'],
                'std_mean_absolute_error': std_analysis['std_mean_absolute_error']
            }
            
            results.append(result)
            
            # Store trajectory data for visualization
            trajectory_data[model_id] = {
                'trajectories': trajectories.detach().cpu().numpy(),
                'empirical_std_generated': std_analysis['empirical_std_generated'],
                'empirical_std_ground_truth': std_analysis['empirical_std_ground_truth'],
                'time_steps': std_analysis['time_steps']
            }
            
            print(f"✅ {model_id} evaluation completed")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
            print(f"   Std RMSE: {std_analysis['std_rmse']:.4f}")
            print(f"   Std Correlation: {std_analysis['std_correlation']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("❌ No adversarial models successfully evaluated")
        return None, None
    
    # Save results to adversarial evaluation directory
    results_df = pd.DataFrame(results)
    save_dir = f"results/{dataset_name}_adversarial/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'enhanced_models_evaluation.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"ENHANCED EVALUATION COMPLETE FOR {dataset_name.upper()}_ADVERSARIAL")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Adversarial models evaluated: {len(results)}")
    
    # Create enhanced visualizations for adversarial models
    create_enhanced_visualizations(results_df, trajectory_data, save_dir, f"{dataset_name}_adversarial")
    
    return results_df, trajectory_data


def evaluate_latent_sde_models_for_dataset(dataset_name: str):
    """Evaluate latent SDE models for a specific dataset."""
    print(f"\n{'='*70}")
    print(f"🧠 Enhanced Latent SDE Model Evaluation for {dataset_name.upper()} Dataset")
    print(f"{'='*70}")
    
    # Setup latent SDE checkpoint manager
    checkpoint_manager = create_checkpoint_manager(f'results/{dataset_name}_latent_sde')
    
    # Get list of trained latent SDE models
    available_models = checkpoint_manager.list_available_models()
    
    if not available_models:
        print(f"❌ No trained latent SDE models found for {dataset_name}")
        return None, None
    
    print(f"Found {len(available_models)} trained latent SDE models for {dataset_name}:")
    checkpoint_manager.print_available_models()
    
    # Setup evaluation data (same as other models)
    print(f"\nSetting up evaluation data for {dataset_name}_latent_sde...")
    if dataset_name == 'ou_process':
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
    else:
        dataset = get_signal(num_samples=64)
        eval_data = torch.stack([dataset[i][0] for i in range(32)])
        print(f"   Note: Using OU process data as evaluation baseline for {dataset_name}")
    
    print(f"Evaluation data: {eval_data.shape}")
    
    # Import latent SDE functions
    from models.latent_sde.implementations.v1_latent_sde import create_v1_model
    from models.sdematching.implementations.v2_sde_matching import create_v2_model
    
    # Evaluate each latent SDE model
    results = []
    trajectory_data = {}
    
    for model_id in available_models:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_id}")
        print(f"{'='*50}")
        
        try:
            # Recreate latent SDE model
            print(f"Recreating latent SDE model {model_id}...")
            
            # Create example batch for model initialization
            example_batch = eval_data[:8]
            
            # Create latent SDE model
            if model_id == "V1":
                latent_sde_model = create_v1_model(
                    example_batch=example_batch,
                    real_data=eval_data,
                    theta=2.0,      # OU mean reversion rate
                    mu=0.0,         # OU long-term mean
                    sigma=0.5,      # OU volatility
                    hidden_size=64  # Neural network size
                )
            elif model_id == "V2":
                latent_sde_model = create_v2_model(
                    example_batch=example_batch,
                    real_data=eval_data,
                    data_size=1,        # Observable dimension
                    latent_size=4,      # Latent dimension
                    hidden_size=64,     # Hidden layer size
                    noise_std=0.1       # Observation noise
                )
            else:
                print(f"❌ Unknown latent SDE model: {model_id}")
                continue
            
            # Load the trained weights
            model_dir = f'results/{dataset_name}_latent_sde/trained_models/{model_id}'
            model_path = os.path.join(model_dir, 'model.pth')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                latent_sde_model.load_state_dict(state_dict)
                print(f"✅ Loaded trained weights for {model_id}")
            else:
                print(f"⚠️ No trained weights found for {model_id}, using initialized model")
            
            checkpoint_info = checkpoint_manager.get_checkpoint_info(model_id)
            print(f"✅ Model {model_id} loaded successfully")
            print(f"Model loaded: {sum(p.numel() for p in latent_sde_model.parameters()):,} parameters")
            print(f"Training: Epoch {checkpoint_info['epoch']}, Loss {checkpoint_info['loss']:.6f}")
            
            # Generate evaluation samples
            latent_sde_model.eval()
            with torch.no_grad():
                samples = latent_sde_model.generate_samples(64)
                print(f"Generated samples: {samples.shape}")
                
                # Generate 20 trajectories for visualization
                trajectories = latent_sde_model.generate_samples(20)
                print(f"Generated trajectories for visualization: {trajectories.shape}")
            
            # Compute core metrics
            metrics = compute_core_metrics(samples, eval_data)
            
            # Compute empirical std analysis
            # Extract value dimension for latent SDE models (they generate (batch, 2, time_steps))
            if trajectories.dim() == 3 and trajectories.shape[1] == 2:
                trajectories_for_std = trajectories[:, 1, :]  # Extract value dimension
            else:
                trajectories_for_std = trajectories
            
            std_analysis = compute_empirical_std_analysis(trajectories_for_std, eval_data)
            
            # Combine results
            result = {
                'model_id': model_id,
                'training_epoch': checkpoint_info['epoch'],
                'training_loss': checkpoint_info['loss'],
                'total_parameters': sum(p.numel() for p in latent_sde_model.parameters()),
                'model_class': type(latent_sde_model).__name__,
                **metrics,
                'std_rmse': std_analysis['std_rmse'],
                'std_correlation': std_analysis['std_correlation'],
                'std_mean_absolute_error': std_analysis['std_mean_absolute_error']
            }
            
            results.append(result)
            
            # Store trajectory data for visualization (use same processed trajectories)
            trajectory_data[model_id] = {
                'trajectories': trajectories_for_std.detach().cpu().numpy(),
                'empirical_std_generated': std_analysis['empirical_std_generated'],
                'empirical_std_ground_truth': std_analysis['empirical_std_ground_truth'],
                'time_steps': std_analysis['time_steps']
            }
            
            print(f"✅ {model_id} evaluation completed")
            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   KS Statistic: {metrics['ks_statistic']:.4f}")
            print(f"   Std RMSE: {std_analysis['std_rmse']:.4f}")
            print(f"   Std Correlation: {std_analysis['std_correlation']:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {model_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("❌ No latent SDE models successfully evaluated")
        return None, None
    
    # Save results to latent SDE evaluation directory
    results_df = pd.DataFrame(results)
    save_dir = f"results/{dataset_name}_latent_sde/evaluation"
    os.makedirs(save_dir, exist_ok=True)
    
    results_path = os.path.join(save_dir, 'enhanced_models_evaluation.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"ENHANCED EVALUATION COMPLETE FOR {dataset_name.upper()}_LATENT_SDE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    print(f"Latent SDE models evaluated: {len(results)}")
    
    # Create enhanced visualizations for latent SDE models
    create_enhanced_visualizations(results_df, trajectory_data, save_dir, f"{dataset_name}_latent_sde")
    
    return results_df, trajectory_data


def create_adversarial_comparison_analysis(all_results: Dict):
    """Create comprehensive adversarial vs non-adversarial comparison analysis."""
    print("\n🎯 Creating Adversarial vs Non-Adversarial Comparison Analysis...")
    
    # Combine all results across datasets
    combined_non_adv = []
    combined_adv = []
    
    for dataset_name, dataset_data in all_results.items():
        # Non-adversarial results
        if 'non_adversarial' in dataset_data:
            non_adv_df = dataset_data['non_adversarial']['results_df'].copy()
            non_adv_df['dataset'] = dataset_name
            non_adv_df['training_type'] = 'non_adversarial'
            combined_non_adv.append(non_adv_df)
        
        # Adversarial results
        if 'adversarial' in dataset_data:
            adv_df = dataset_data['adversarial']['results_df'].copy()
            adv_df['dataset'] = dataset_name
            adv_df['training_type'] = 'adversarial'
            # Clean up model IDs (remove _ADV suffix for comparison)
            adv_df['base_model_id'] = adv_df['model_id'].str.replace('_ADV', '')
            combined_adv.append(adv_df)
    
    if not combined_non_adv and not combined_adv:
        print("❌ No results to compare")
        return
    
    # Create comparison directory
    comparison_dir = 'results/adversarial_comparison'
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Combine all results
    all_combined = []
    if combined_non_adv:
        non_adv_combined = pd.concat(combined_non_adv, ignore_index=True)
        all_combined.append(non_adv_combined)
    if combined_adv:
        adv_combined = pd.concat(combined_adv, ignore_index=True)
        all_combined.append(adv_combined)
    
    if all_combined:
        full_combined = pd.concat(all_combined, ignore_index=True)
        
        # Create visualizations
        create_adversarial_vs_non_adversarial_plots(non_adv_combined if combined_non_adv else pd.DataFrame(), 
                                                   adv_combined if combined_adv else pd.DataFrame(), 
                                                   full_combined, comparison_dir)
        
        # Save combined results
        full_combined.to_csv(os.path.join(comparison_dir, 'all_models_combined_results.csv'), index=False)
        if combined_non_adv:
            non_adv_combined.to_csv(os.path.join(comparison_dir, 'non_adversarial_results.csv'), index=False)
        if combined_adv:
            adv_combined.to_csv(os.path.join(comparison_dir, 'adversarial_results.csv'), index=False)
        
        print(f"✅ Adversarial comparison analysis saved to: {comparison_dir}/")


def create_adversarial_vs_non_adversarial_plots(non_adv_df: pd.DataFrame, adv_df: pd.DataFrame, 
                                               combined_df: pd.DataFrame, save_dir: str):
    """Create adversarial vs non-adversarial comparison plots."""
    print("🎨 Creating adversarial vs non-adversarial comparison plots...")
    
    # Find models that have both adversarial and non-adversarial variants
    if not adv_df.empty:
        adv_base_models = set(adv_df['base_model_id'].unique())
    else:
        adv_base_models = set()
    
    if not non_adv_df.empty:
        non_adv_models = set(non_adv_df['model_id'].unique())
    else:
        non_adv_models = set()
    
    # Models with both variants
    paired_models = adv_base_models.intersection(non_adv_models)
    
    if not paired_models:
        print("⚠️ No models found with both adversarial and non-adversarial variants")
        return
    
    print(f"Found {len(paired_models)} models with both variants: {sorted(paired_models)}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Adversarial vs Non-Adversarial Training Comparison', fontsize=18, fontweight='bold')
    
    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    create_paired_comparison_plot(ax1, non_adv_df, adv_df, paired_models, 'rmse', 
                                 'RMSE Comparison', 'RMSE (Lower = Better)')
    
    # 2. KS Statistic Comparison  
    ax2 = axes[0, 1]
    create_paired_comparison_plot(ax2, non_adv_df, adv_df, paired_models, 'ks_statistic',
                                 'KS Statistic Comparison', 'KS Statistic (Lower = Better)')
    
    # 3. Wasserstein Distance Comparison
    ax3 = axes[1, 0]
    create_paired_comparison_plot(ax3, non_adv_df, adv_df, paired_models, 'wasserstein_distance',
                                 'Wasserstein Distance Comparison', 'Wasserstein Distance (Lower = Better)')
    
    # 4. Std RMSE Comparison
    ax4 = axes[1, 1]
    create_paired_comparison_plot(ax4, non_adv_df, adv_df, paired_models, 'std_rmse',
                                 'Empirical Std RMSE Comparison', 'Std RMSE (Lower = Better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adversarial_vs_non_adversarial_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✅ Adversarial vs non-adversarial comparison saved to: {save_dir}/adversarial_vs_non_adversarial_comparison.png")


def create_paired_comparison_plot(ax, non_adv_df: pd.DataFrame, adv_df: pd.DataFrame, 
                                 paired_models: set, metric: str, title: str, ylabel: str):
    """Create a paired comparison plot for a specific metric."""
    models = sorted(paired_models)
    non_adv_values = []
    adv_values = []
    
    for model in models:
        # Get non-adversarial value (average across datasets)
        non_adv_data = non_adv_df[non_adv_df['model_id'] == model]
        non_adv_val = non_adv_data[metric].mean() if len(non_adv_data) > 0 else np.nan
        non_adv_values.append(non_adv_val)
        
        # Get adversarial value (average across datasets)
        adv_data = adv_df[adv_df['base_model_id'] == model]
        adv_val = adv_data[metric].mean() if len(adv_data) > 0 else np.nan
        adv_values.append(adv_val)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, non_adv_values, width, label='Non-Adversarial', 
                   color='skyblue', alpha=0.8, edgecolor='navy')
    bars2 = ax.bar(x + width/2, adv_values, width, label='Adversarial', 
                   color='coral', alpha=0.8, edgecolor='darkred')
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars, values in [(bars1, non_adv_values), (bars2, adv_values)]:
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(non_adv_values + adv_values) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)


def create_enhanced_visualizations(results_df: pd.DataFrame, trajectory_data: Dict, save_dir: str, dataset_name: str = None):
    """Create enhanced visualizations with trajectories and empirical std analysis."""
    print(f"\nCreating enhanced visualizations for {dataset_name or 'dataset'}...")
    
    # Define distributional metrics to visualize
    distributional_metrics = [
        {
            'metric': 'rmse',
            'title': 'RMSE by Model (Lower = Better)',
            'ylabel': 'RMSE',
            'color': 'skyblue',
            'sort_ascending': True
        },
        {
            'metric': 'ks_statistic', 
            'title': 'KS Statistic by Model (Lower = Better)',
            'ylabel': 'KS Statistic',
            'color': 'lightcoral',
            'sort_ascending': True
        },
        {
            'metric': 'wasserstein_distance',
            'title': 'Wasserstein Distance by Model (Lower = Better)',
            'ylabel': 'Wasserstein Distance', 
            'color': 'lightgreen',
            'sort_ascending': True
        },
        {
            'metric': 'std_rmse',
            'title': 'Empirical Std RMSE by Model (Lower = Better)',
            'ylabel': 'Std RMSE',
            'color': 'gold',
            'sort_ascending': True
        }
    ]
    
    # Create figure with 4 subplots - one for each distributional metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title = f'Enhanced Model Evaluation: {dataset_name.upper() if dataset_name else "Dataset"} - Individual Distributional Metrics'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create separate bar plot for each distributional metric
    for i, metric_config in enumerate(distributional_metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        metric = metric_config['metric']
        
        # Sort models by this specific metric
        sorted_results = results_df.sort_values(metric, ascending=metric_config['sort_ascending']).reset_index(drop=True)
        models = sorted_results['model_id'].tolist()
        values = sorted_results[metric].tolist()
        
        # Create bar plot
        bars = ax.bar(range(len(models)), values, 
                     color=metric_config['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(metric_config['title'], fontweight='bold', fontsize=12)
        ax.set_ylabel(metric_config['ylabel'], fontsize=11)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        max_val = max(values) if values else 1
        for j, v in enumerate(values):
            ax.text(j, v + max_val * 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add ranking information as subtitle
        ax.text(0.5, -0.15, f'Ranked by {metric_config["ylabel"]} (Best → Worst)', 
               transform=ax.transAxes, ha='center', va='top', 
               fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'enhanced_model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Enhanced comparison visualization saved to: {os.path.join(save_dir, 'enhanced_model_comparison.png')}")
    
    # Create individual metric plots for better visibility
    create_individual_metric_plots(results_df, save_dir, dataset_name)
    
    # Create trajectory visualization
    create_trajectory_visualization(trajectory_data, save_dir, dataset_name)
    
    # Create empirical std comparison
    create_empirical_std_visualization(trajectory_data, save_dir, dataset_name)


def create_individual_metric_plots(results_df: pd.DataFrame, save_dir: str, dataset_name: str = None):
    """Create individual bar plots for each distributional metric with larger, cleaner visualizations."""
    print(f"Creating individual metric plots for {dataset_name or 'dataset'}...")
    
    # Define distributional metrics with enhanced styling
    distributional_metrics = [
        {
            'metric': 'rmse',
            'title': 'RMSE Performance Ranking',
            'ylabel': 'RMSE (Root Mean Square Error)',
            'color': '#4A90E2',  # Professional blue
            'sort_ascending': True,
            'description': 'Point-wise trajectory matching accuracy'
        },
        {
            'metric': 'ks_statistic', 
            'title': 'KS Statistic Distribution Quality',
            'ylabel': 'KS Statistic',
            'color': '#F5A623',  # Professional orange
            'sort_ascending': True,
            'description': 'Statistical distribution similarity (lower = better match)'
        },
        {
            'metric': 'wasserstein_distance',
            'title': 'Wasserstein Distance Distribution Quality',
            'ylabel': 'Wasserstein Distance', 
            'color': '#7ED321',  # Professional green
            'sort_ascending': True,
            'description': 'Earth Mover\'s Distance between distributions'
        },
        {
            'metric': 'std_rmse',
            'title': 'Empirical Standard Deviation Matching',
            'ylabel': 'Std RMSE',
            'color': '#D0021B',  # Professional red
            'sort_ascending': True,
            'description': 'Variance structure matching over time'
        }
    ]
    
    # Create individual plots for each metric
    for metric_config in distributional_metrics:
        metric = metric_config['metric']
        
        # Sort models by this specific metric
        sorted_results = results_df.sort_values(metric, ascending=metric_config['sort_ascending']).reset_index(drop=True)
        models = sorted_results['model_id'].tolist()
        values = sorted_results[metric].tolist()
        
        # Create individual plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create gradient colors for better visual ranking
        n_models = len(models)
        colors = [metric_config['color']] * n_models
        alphas = np.linspace(0.9, 0.4, n_models)  # Best models more opaque
        
        bars = ax.bar(range(len(models)), values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Apply gradient alpha
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)
        
        # Styling
        ax.set_title(f'{metric_config["title"]}\n{dataset_name.upper() if dataset_name else "Dataset"}', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_ylabel(metric_config['ylabel'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Models (Ranked Best → Worst)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        max_val = max(values) if values else 1
        for j, v in enumerate(values):
            # Color-code the text: green for best, red for worst
            if j == 0:  # Best model
                text_color = 'darkgreen'
                text_weight = 'bold'
            elif j == len(values) - 1:  # Worst model
                text_color = 'darkred' 
                text_weight = 'bold'
            else:
                text_color = 'black'
                text_weight = 'normal'
                
            ax.text(j, v + max_val * 0.02, f'{v:.4f}', 
                   ha='center', va='bottom', fontsize=11, 
                   fontweight=text_weight, color=text_color)
        
        # Add description and ranking info
        description_text = f'{metric_config["description"]}\nRanked by {metric_config["ylabel"]} (Lower = Better Performance)'
        ax.text(0.02, 0.98, description_text, transform=ax.transAxes, 
               fontsize=10, va='top', ha='left', style='italic', 
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
        
        # Add best/worst annotations
        if len(values) > 1:
            # Best model annotation
            ax.annotate(f'★ BEST\n{models[0]}', 
                       xy=(0, values[0]), xytext=(0, values[0] + max_val * 0.15),
                       ha='center', fontsize=10, fontweight='bold', color='darkgreen',
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
            
            # Worst model annotation  
            ax.annotate(f'▼ WORST\n{models[-1]}', 
                       xy=(len(models)-1, values[-1]), xytext=(len(models)-1, values[-1] + max_val * 0.15),
                       ha='center', fontsize=10, fontweight='bold', color='darkred',
                       arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f'{metric}_ranking_{dataset_name or "dataset"}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ {metric_config['title']} plot saved to: {filename}")


def create_trajectory_visualization(trajectory_data: Dict, save_dir: str, dataset_name: str = None):
    """Create ultra-clear trajectory visualization with equal sample counts."""
    print(f"Creating ultra-clear trajectory visualization for {dataset_name or 'dataset'}...")
    
    # Get ground truth for comparison (exactly 20 samples to match generated)
    dataset = get_signal(num_samples=64)
    ground_truth = torch.stack([dataset[i][0] for i in range(20)])  # Exactly 20 samples
    gt_values = ground_truth[:, 1, :].numpy()  # Extract value dimension
    
    print(f"Ground truth samples: {gt_values.shape[0]} (matching generated sample count)")
    
    # Create figure with subplots for each model
    n_models = len(trajectory_data)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    title = f'Ultra-Clear Trajectory Analysis: {dataset_name.upper() if dataset_name else "Dataset"} - Generated vs Ground Truth (20 samples each)'
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(trajectory_data.keys())
    time_steps = np.linspace(0, 1, gt_values.shape[1])
    
    for i, model_id in enumerate(model_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Create statistical envelopes for better understanding
        gt_mean = np.mean(gt_values, axis=0)
        gt_std = np.std(gt_values, axis=0)
        
        # Plot ground truth envelope
        ax.fill_between(time_steps, gt_mean - gt_std, gt_mean + gt_std, 
                       color='red', alpha=0.15, label='GT ±1σ')
        
        # Plot ground truth mean (very visible)
        ax.plot(time_steps, gt_mean, color='darkred', linewidth=4, 
               label='GT Mean', alpha=0.9, linestyle='-')
        
        # Plot a few individual ground truth trajectories (clearly visible)
        for j in range(min(3, gt_values.shape[0])):
            ax.plot(time_steps, gt_values[j], color='red', alpha=0.7, 
                   linewidth=1.2, linestyle='--')
        
        # Plot generated trajectories
        trajectories = trajectory_data[model_id]['trajectories']
        time_steps_gen = np.linspace(0, 1, trajectories.shape[1])
        
        # Generated envelope
        gen_mean = np.mean(trajectories, axis=0)
        gen_std = np.std(trajectories, axis=0)
        ax.fill_between(time_steps_gen, gen_mean - gen_std, gen_mean + gen_std,
                       color='blue', alpha=0.1, label='Gen ±1σ')
        
        # Generated mean
        ax.plot(time_steps_gen, gen_mean, color='darkblue', linewidth=2.5,
               label='Gen Mean', alpha=0.8)
        
        # Individual generated trajectories (background)
        for j in range(trajectories.shape[0]):
            ax.plot(time_steps_gen, trajectories[j], color='lightblue', alpha=0.3, linewidth=0.6)
        
        ax.set_title(f'{model_id}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add comprehensive legend for first subplot
        if i == 0:
            ax.plot([], [], color='darkred', linewidth=4, alpha=0.9, label='GT Mean')
            ax.plot([], [], color='red', linewidth=1.2, alpha=0.7, linestyle='--', label='GT Samples')
            ax.plot([], [], color='darkblue', linewidth=2.5, alpha=0.8, label='Gen Mean')
            ax.plot([], [], color='lightblue', linewidth=0.6, alpha=0.3, label='Gen Samples')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Hide unused subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ultra_clear_trajectory_visualization.png'), dpi=300, bbox_inches='tight')
    print(f"Ultra-clear trajectory visualization saved to: {os.path.join(save_dir, 'ultra_clear_trajectory_visualization.png')}")
    print(f"   - Equal sample counts: 20 ground truth, 20 generated per model")
    print(f"   - Ground truth clearly visible with dark red colors")
    print(f"   - Statistical envelopes for better interpretation")


def create_empirical_std_visualization(trajectory_data: Dict, save_dir: str, dataset_name: str = None):
    """Create empirical standard deviation comparison visualization."""
    print(f"Creating empirical std comparison for {dataset_name or 'dataset'}...")
    
    # Sort models by std RMSE (best first)
    std_rmse_scores = {model_id: np.sqrt(np.mean((data['empirical_std_generated'] - data['empirical_std_ground_truth']) ** 2))
                      for model_id, data in trajectory_data.items()}
    sorted_models = sorted(std_rmse_scores.keys(), key=lambda x: std_rmse_scores[x])
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    title = f'Empirical Standard Deviation Analysis: {dataset_name.upper() if dataset_name else "Dataset"}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Std evolution over time
    ax1 = axes[0]
    
    # Plot ground truth std (thick black line)
    gt_std = trajectory_data[list(trajectory_data.keys())[0]]['empirical_std_ground_truth']
    time_steps = trajectory_data[list(trajectory_data.keys())[0]]['time_steps']
    ax1.plot(time_steps, gt_std, color='black', linewidth=3, label='Ground Truth', alpha=0.8)
    
    # Plot each model's std
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_models)))
    for i, model_id in enumerate(sorted_models):
        data = trajectory_data[model_id]
        ax1.plot(data['time_steps'], data['empirical_std_generated'], 
                color=colors[i], linewidth=2, label=f'{model_id}', alpha=0.7)
    
    ax1.set_title('Empirical Standard Deviation Evolution Over Time', fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Empirical Std')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Std RMSE comparison (sorted)
    ax2 = axes[1]
    std_rmse_values = [std_rmse_scores[model_id] for model_id in sorted_models]
    bars = ax2.bar(range(len(sorted_models)), std_rmse_values, color='orange', alpha=0.7)
    ax2.set_title('Empirical Std RMSE by Model (Sorted by Performance)', fontweight='bold')
    ax2.set_ylabel('Std RMSE (Lower = Better)')
    ax2.set_xticks(range(len(sorted_models)))
    ax2.set_xticklabels(sorted_models, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(std_rmse_values):
        ax2.text(i, v + max(std_rmse_values) * 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'empirical_std_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Empirical std analysis saved to: {os.path.join(save_dir, 'empirical_std_analysis.png')}")


def main():
    """Main evaluation function."""
    # Run enhanced evaluation across all datasets
    all_results = evaluate_all_models_enhanced()
    
    if all_results is not None:
        print(f"\n🎉 ENHANCED EVALUATION COMPLETE!")
        print(f"   All models evaluated across all datasets")
        print(f"   Dataset-specific results saved to individual directories")
        print(f"   Clean visualizations created for each dataset")
        print(f"   20 trajectories per model for comprehensive analysis")
        print(f"   Empirical std matching analysis included")
        print(f"\n📁 Results saved to:")
        for dataset_name in all_results.keys():
            print(f"   results/{dataset_name}/evaluation/")
    else:
        print(f"\n❌ No trained models found in any dataset")
        print(f"   Train models first using train_and_save_models.py")


if __name__ == "__main__":
    main()

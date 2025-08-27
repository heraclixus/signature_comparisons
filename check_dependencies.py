#!/usr/bin/env python3
"""
Dependency Checker for Signature Comparisons Project

This script checks for all required dependencies and provides clear installation instructions
for any missing packages. Run this before attempting to use the project.
"""

import sys
import importlib
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DependencyInfo:
    name: str
    import_name: str
    required_for: str
    install_cmd: str
    critical: bool = False

# Define all project dependencies with clear information
DEPENDENCIES = [
    # Critical dependencies (project won't work without these)
    DependencyInfo("torch", "torch", "Core PyTorch functionality", "pip install torch", critical=True),
    DependencyInfo("numpy", "numpy", "Numerical computations", "pip install numpy", critical=True),
    DependencyInfo("torchsde", "torchsde", "Latent SDE models (C1-C6)", "pip install torchsde", critical=True),
    DependencyInfo("torchcde", "torchcde", "Neural CDE functionality", "pip install torchcde", critical=True),
    DependencyInfo("torchtyping", "torchtyping", "Type checking for tensors", "pip install torchtyping", critical=True),
    
    # Model-specific dependencies
    DependencyInfo("signatory", "signatory", "Signature computations", "pip install signatory", critical=False),
    DependencyInfo("iisignature", "iisignature", "Alternative signature computations", "pip install iisignature", critical=False),
    
    # Optional but recommended
    DependencyInfo("matplotlib", "matplotlib", "Plotting and visualization", "pip install matplotlib", critical=False),
    DependencyInfo("seaborn", "seaborn", "Statistical plotting", "pip install seaborn", critical=False),
    DependencyInfo("pandas", "pandas", "Data manipulation", "pip install pandas", critical=False),
    DependencyInfo("scikit-learn", "sklearn", "Machine learning utilities", "pip install scikit-learn", critical=False),
    DependencyInfo("tqdm", "tqdm", "Progress bars", "pip install tqdm", critical=False),
    
    # Specialized packages
    DependencyInfo("fbm", "fbm", "Fractional Brownian Motion generation", "pip install fbm", critical=False),
    DependencyInfo("stochastic", "stochastic", "Stochastic process generation", "pip install stochastic", critical=False),
    DependencyInfo("jax", "jax", "JAX-based computations", "pip install jax jaxlib", critical=False),
]

def check_dependency(dep: DependencyInfo) -> Tuple[bool, Optional[str]]:
    """Check if a dependency is available and return status with error message if any."""
    try:
        importlib.import_module(dep.import_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def check_conda_environment() -> Tuple[bool, str]:
    """Check if we're in the correct conda environment."""
    try:
        result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if '*' in line and 'sig19' in line:
                    return True, "✅ Running in sig19 conda environment"
            return False, "❌ Not in sig19 environment. Run: conda activate sig19"
        else:
            return False, "❌ Conda not available or not working"
    except FileNotFoundError:
        return False, "❌ Conda not found. Please install conda/miniconda"

def main():
    print("🔍 Signature Comparisons Project - Dependency Checker")
    print("=" * 60)
    
    # Check conda environment first
    env_ok, env_msg = check_conda_environment()
    print(f"\n📦 Environment Status:")
    print(f"   {env_msg}")
    
    if not env_ok:
        print(f"\n💡 Quick Fix:")
        print(f"   conda activate sig19")
        print(f"   # If environment doesn't exist:")
        print(f"   conda env create -f env.yaml")
        print(f"   conda activate sig19")
    
    print(f"\n🔧 Dependency Status:")
    
    missing_critical = []
    missing_optional = []
    available_count = 0
    
    for dep in DEPENDENCIES:
        available, error = check_dependency(dep)
        
        if available:
            print(f"   ✅ {dep.name:<15} - {dep.required_for}")
            available_count += 1
        else:
            status = "❌ CRITICAL" if dep.critical else "⚠️  OPTIONAL"
            print(f"   {status:<12} {dep.name:<15} - {dep.required_for}")
            print(f"      └─ Error: {error}")
            print(f"      └─ Install: {dep.install_cmd}")
            
            if dep.critical:
                missing_critical.append(dep)
            else:
                missing_optional.append(dep)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Available: {available_count}/{len(DEPENDENCIES)} dependencies")
    print(f"   Critical missing: {len(missing_critical)}")
    print(f"   Optional missing: {len(missing_optional)}")
    
    # Recommendations
    if missing_critical:
        print(f"\n🚨 CRITICAL ISSUES - Project will not work:")
        for dep in missing_critical:
            print(f"   • {dep.name}: {dep.install_cmd}")
        
        print(f"\n💡 Recommended fix:")
        print(f"   conda activate sig19")
        print(f"   # If that doesn't work, recreate environment:")
        print(f"   conda env create -f env.yaml --force")
        
        return 1  # Exit with error code
    
    elif missing_optional:
        print(f"\n⚠️  Optional dependencies missing (some features may be limited):")
        for dep in missing_optional:
            print(f"   • {dep.name}: {dep.install_cmd}")
    
    else:
        print(f"\n🎉 All dependencies available! Project should work correctly.")
    
    # Test all model imports
    print(f"\n🧪 Testing all model imports...")
    
    models_to_test = [
        ("A1", "src.models.implementations.a1_final", "create_a1_final_model"),
        ("A2", "src.models.implementations.a2_canned_scoring", "create_a2_model"),
        ("A3", "src.models.implementations.a3_canned_mmd", "create_a3_model"),
        ("A4", "src.models.implementations.a4_canned_logsig", "create_a4_model"),
        ("B1", "src.models.implementations.b1_nsde_scoring", "create_b1_model"),
        ("B2", "src.models.implementations.b2_nsde_mmd_pde", "create_b2_model"),
        ("B3", "src.models.implementations.b3_nsde_tstatistic", "create_b3_model"),
        ("B4", "src.models.implementations.b4_nsde_mmd", "create_b4_model"),
        ("B5", "src.models.implementations.b5_nsde_scoring", "create_b5_model"),
        ("C1", "src.models.implementations.hybrid_latent_sde.c1_latent_sde_tstat", "create_c1_model"),
        ("C2", "src.models.implementations.hybrid_latent_sde.c2_latent_sde_scoring", "create_c2_model"),
        ("C3", "src.models.implementations.hybrid_latent_sde.c3_latent_sde_mmd", "create_c3_model"),
        ("C4", "src.models.implementations.hybrid_latent_sde.c4_sde_matching_tstat", "create_c4_model"),
        ("C5", "src.models.implementations.hybrid_latent_sde.c5_sde_matching_scoring", "create_c5_model"),
        ("C6", "src.models.implementations.hybrid_latent_sde.c6_sde_matching_mmd", "create_c6_model"),
        ("D1", "src.models.implementations.d1_diffusion", "create_d1_model"),
        ("D2", "src.models.implementations.d2_distributional_diffusion", "create_model"),
        ("D3", "src.models.implementations.d3_distributional_pde", "create_model"),
        ("D4", "src.models.implementations.d4_distributional_truncated", "create_model"),
        ("V1", "src.models.latent_sde.implementations.v1_latent_sde", "create_v1_model"),
        ("V2", "src.models.sdematching.implementations.v2_sde_matching", "create_v2_model"),
    ]
    
    available_models = []
    failed_models = []
    
    for model_id, module_path, function_name in models_to_test:
        try:
            print(f"   Testing {model_id:<3} model import...", end=" ")
            module = importlib.import_module(module_path)
            create_fn = getattr(module, function_name)
            print("✅")
            available_models.append(model_id)
        except Exception as e:
            print("❌")
            print(f"      └─ Error: {e}")
            failed_models.append((model_id, str(e)))
    
    # Test signature computations
    try:
        print(f"   Testing signature computations...", end=" ")
        from src.signatures import TruncatedSignature
        print("✅")
    except Exception as e:
        print("❌")
        print(f"      └─ Error: {e}")
    
    # Summary of model availability
    print(f"\n📋 Model Availability Summary:")
    print(f"   ✅ Available models ({len(available_models)}): {', '.join(available_models)}")
    
    if failed_models:
        print(f"   ❌ Failed models ({len(failed_models)}):")
        for model_id, error in failed_models:
            print(f"      • {model_id}: {error}")
    
    # Test training script compatibility
    print(f"\n🔧 Testing training script model detection...")
    try:
        # Simulate the training script's model detection logic
        sys.path.append('src')
        
        # Import the exact same way as train_and_save_models.py
        model_configs = {}
        
        # Test D4 specifically since that was the reported issue
        try:
            from models.implementations.d4_distributional_truncated import create_model as create_d4_model
            D4_AVAILABLE = True
            model_configs["D4"] = (create_d4_model, "Distributional Diffusion + Truncated Signature Kernels", D4_AVAILABLE)
            print(f"   ✅ D4 detected by training script logic")
        except ImportError as e:
            print(f"   ❌ D4 failed in training script context: {e}")
        
        # Show what the training script would see
        available_in_training = [k for k, (_, _, available) in model_configs.items() if available]
        print(f"   📊 Models available to training script: {available_in_training}")
        
    except Exception as e:
        print(f"   ❌ Training script compatibility test failed: {e}")
    
    print(f"\n✨ Model availability check complete!")
    
    # Return error code if critical models failed
    critical_models = ["A1", "A2", "B1", "D1", "D2"]  # Core models that should always work
    failed_critical = [m for m, _ in failed_models if m in critical_models]
    
    if failed_critical:
        print(f"\n🚨 CRITICAL: Core models failed: {failed_critical}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

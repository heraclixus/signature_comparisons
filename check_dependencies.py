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
    
    # Test critical imports
    print(f"\n🧪 Testing critical imports...")
    try:
        print(f"   Testing D1 model import...", end=" ")
        from src.models.implementations.d1_diffusion import create_d1_model
        print("✅")
    except Exception as e:
        print(f"❌")
        print(f"      Error: {e}")
        print(f"      This suggests missing dependencies or import issues.")
        return 1
    
    try:
        print(f"   Testing signature computations...", end=" ")
        from src.signatures import TruncatedSignature
        print("✅")
    except Exception as e:
        print(f"❌")
        print(f"      Error: {e}")
    
    print(f"\n✨ Dependency check complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Debug Training Models Script

This script helps debug why certain models (like D3, D4) are not appearing 
in the available models list in train_and_save_models.py
"""

import sys
import os
import importlib
from typing import Dict, List, Tuple

# Add src to path
sys.path.append('src')

def test_individual_imports():
    """Test each model import individually to see which ones fail."""
    print("🔍 Testing Individual Model Imports")
    print("=" * 50)
    
    # Define all models that should be available
    model_imports = [
        ("A1", "models.implementations.a1_final", "create_a1_final_model"),
        ("A2", "models.implementations.a2_canned_scoring", "create_a2_model"),
        ("A3", "models.implementations.a3_canned_mmd", "create_a3_model"),
        ("A4", "models.implementations.a4_canned_logsig", "create_a4_model"),
        ("B1", "models.implementations.b1_nsde_scoring", "create_b1_model"),
        ("B2", "models.implementations.b2_nsde_mmd_pde", "create_b2_model"),
        ("B3", "models.implementations.b3_nsde_tstatistic", "create_b3_model"),
        ("B4", "models.implementations.b4_nsde_mmd", "create_b4_model"),
        ("B5", "models.implementations.b5_nsde_scoring", "create_b5_model"),
        ("C1", "models.implementations.hybrid_latent_sde.c1_latent_sde_tstat", "create_c1_model"),
        ("C2", "models.implementations.hybrid_latent_sde.c2_latent_sde_scoring", "create_c2_model"),
        ("C3", "models.implementations.hybrid_latent_sde.c3_latent_sde_mmd", "create_c3_model"),
        ("C4", "models.implementations.hybrid_latent_sde.c4_sde_matching_tstat", "create_c4_model"),
        ("C5", "models.implementations.hybrid_latent_sde.c5_sde_matching_scoring", "create_c5_model"),
        ("C6", "models.implementations.hybrid_latent_sde.c6_sde_matching_mmd", "create_c6_model"),
        ("D1", "models.implementations.d1_diffusion", "create_d1_model"),
        ("D2", "models.implementations.d2_distributional_diffusion", "create_model"),
        ("D3", "models.implementations.d3_distributional_pde", "create_model"),
        ("D4", "models.implementations.d4_distributional_truncated", "create_model"),
        ("V1", "models.latent_sde.implementations.v1_latent_sde", "create_v1_model"),
        ("V2", "models.sdematching.implementations.v2_sde_matching", "create_v2_model"),
    ]
    
    available_models = []
    failed_models = []
    
    for model_id, module_path, function_name in model_imports:
        try:
            print(f"   Testing {model_id:<3} import...", end=" ")
            
            # Import module
            module = importlib.import_module(module_path)
            
            # Get function
            if function_name == "create_model":
                # For D2, D3, D4 which use generic create_model
                create_fn = getattr(module, "create_model")
            else:
                create_fn = getattr(module, function_name)
            
            print("✅")
            available_models.append((model_id, create_fn, module_path))
            
        except Exception as e:
            print("❌")
            print(f"      └─ Error: {e}")
            failed_models.append((model_id, str(e)))
    
    print(f"\n📊 Import Results:")
    print(f"   ✅ Available: {len(available_models)} models")
    print(f"   ❌ Failed: {len(failed_models)} models")
    
    if available_models:
        print(f"\n✅ Available Models:")
        for model_id, _, module_path in available_models:
            print(f"   {model_id}: {module_path}")
    
    if failed_models:
        print(f"\n❌ Failed Models:")
        for model_id, error in failed_models:
            print(f"   {model_id}: {error}")
    
    return available_models, failed_models

def simulate_training_script_imports():
    """Simulate the exact import sequence from train_and_save_models.py"""
    print(f"\n🎭 Simulating Training Script Import Sequence")
    print("=" * 50)
    
    # Simulate the exact imports from train_and_save_models.py
    availability_flags = {}
    
    # A1
    try:
        from models.implementations.a1_final import create_a1_final_model
        availability_flags['A1'] = True
        print("✅ A1_AVAILABLE = True")
    except ImportError:
        availability_flags['A1'] = False
        print("❌ A1_AVAILABLE = False")
    
    # A2
    try:
        from models.implementations.a2_canned_scoring import create_a2_model
        availability_flags['A2'] = True
        print("✅ A2_AVAILABLE = True")
    except ImportError:
        availability_flags['A2'] = False
        print("❌ A2_AVAILABLE = False")
    
    # A3
    try:
        from models.implementations.a3_canned_mmd import create_a3_model
        availability_flags['A3'] = True
        print("✅ A3_AVAILABLE = True")
    except ImportError:
        availability_flags['A3'] = False
        print("❌ A3_AVAILABLE = False")
    
    # A4
    try:
        from models.implementations.a4_canned_logsig import create_a4_model
        availability_flags['A4'] = True
        print("✅ A4_AVAILABLE = True")
    except ImportError:
        availability_flags['A4'] = False
        print("❌ A4_AVAILABLE = False")
    
    # B1
    try:
        from models.implementations.b1_nsde_scoring import create_b1_model
        availability_flags['B1'] = True
        print("✅ B1_AVAILABLE = True")
    except ImportError:
        availability_flags['B1'] = False
        print("❌ B1_AVAILABLE = False")
    
    # B2
    try:
        from models.implementations.b2_nsde_mmd_pde import create_b2_model
        availability_flags['B2'] = True
        print("✅ B2_AVAILABLE = True")
    except ImportError:
        availability_flags['B2'] = False
        print("❌ B2_AVAILABLE = False")
    
    # B3
    try:
        from models.implementations.b3_nsde_tstatistic import create_b3_model
        availability_flags['B3'] = True
        print("✅ B3_AVAILABLE = True")
    except ImportError:
        availability_flags['B3'] = False
        print("❌ B3_AVAILABLE = False")
    
    # B4
    try:
        from models.implementations.b4_nsde_mmd import create_b4_model
        availability_flags['B4'] = True
        print("✅ B4_AVAILABLE = True")
    except ImportError:
        availability_flags['B4'] = False
        print("❌ B4_AVAILABLE = False")
    
    # B5
    try:
        from models.implementations.b5_nsde_scoring import create_b5_model
        availability_flags['B5'] = True
        print("✅ B5_AVAILABLE = True")
    except ImportError:
        availability_flags['B5'] = False
        print("❌ B5_AVAILABLE = False")
    
    # C1
    try:
        from models.implementations.hybrid_latent_sde.c1_latent_sde_tstat import create_c1_model
        availability_flags['C1'] = True
        print("✅ C1_AVAILABLE = True")
    except ImportError:
        availability_flags['C1'] = False
        print("❌ C1_AVAILABLE = False")
    
    # C2
    try:
        from models.implementations.hybrid_latent_sde.c2_latent_sde_scoring import create_c2_model
        availability_flags['C2'] = True
        print("✅ C2_AVAILABLE = True")
    except ImportError:
        availability_flags['C2'] = False
        print("❌ C2_AVAILABLE = False")
    
    # C3
    try:
        from models.implementations.hybrid_latent_sde.c3_latent_sde_mmd import create_c3_model
        availability_flags['C3'] = True
        print("✅ C3_AVAILABLE = True")
    except ImportError:
        availability_flags['C3'] = False
        print("❌ C3_AVAILABLE = False")
    
    # C4
    try:
        from models.implementations.hybrid_latent_sde.c4_sde_matching_tstat import create_c4_model
        availability_flags['C4'] = True
        print("✅ C4_AVAILABLE = True")
    except ImportError:
        availability_flags['C4'] = False
        print("❌ C4_AVAILABLE = False")
    
    # C5
    try:
        from models.implementations.hybrid_latent_sde.c5_sde_matching_scoring import create_c5_model
        availability_flags['C5'] = True
        print("✅ C5_AVAILABLE = True")
    except ImportError:
        availability_flags['C5'] = False
        print("❌ C5_AVAILABLE = False")
    
    # C6
    try:
        from models.implementations.hybrid_latent_sde.c6_sde_matching_mmd import create_c6_model
        availability_flags['C6'] = True
        print("✅ C6_AVAILABLE = True")
    except ImportError:
        availability_flags['C6'] = False
        print("❌ C6_AVAILABLE = False")
    
    # D1
    try:
        from models.implementations.d1_diffusion import create_d1_model
        availability_flags['D1'] = True
        print("✅ D1_AVAILABLE = True")
    except ImportError as e:
        availability_flags['D1'] = False
        print(f"❌ D1_AVAILABLE = False: {e}")
    
    # D2
    try:
        from models.implementations.d2_distributional_diffusion import create_model as create_d2_model
        availability_flags['D2'] = True
        print("✅ D2_AVAILABLE = True")
    except ImportError as e:
        availability_flags['D2'] = False
        print(f"❌ D2_AVAILABLE = False: {e}")
    
    # D3 - This is the problematic one!
    try:
        from models.implementations.d3_distributional_pde import create_model as create_d3_model
        availability_flags['D3'] = True
        print("✅ D3_AVAILABLE = True")
    except ImportError as e:
        availability_flags['D3'] = False
        print(f"❌ D3_AVAILABLE = False: {e}")
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
    
    # D4 - This might also be problematic!
    try:
        from models.implementations.d4_distributional_truncated import create_model as create_d4_model
        availability_flags['D4'] = True
        print("✅ D4_AVAILABLE = True")
    except ImportError as e:
        availability_flags['D4'] = False
        print(f"❌ D4_AVAILABLE = False: {e}")
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
    
    # V1
    try:
        from models.latent_sde.implementations.v1_latent_sde import create_v1_model
        availability_flags['V1'] = True
        print("✅ V1_AVAILABLE = True")
    except ImportError:
        availability_flags['V1'] = False
        print("❌ V1_AVAILABLE = False")
    
    # V2
    try:
        from models.sdematching.implementations.v2_sde_matching import create_v2_model
        availability_flags['V2'] = True
        print("✅ V2_AVAILABLE = True")
    except ImportError:
        availability_flags['V2'] = False
        print("❌ V2_AVAILABLE = False")
    
    return availability_flags

def check_model_configs_dict():
    """Check what the model_configs dictionary would contain in train_single_model."""
    print(f"\n📋 Checking Model Configs Dictionary")
    print("=" * 50)
    
    # Simulate the availability flags from training script
    availability_flags = simulate_training_script_imports()
    
    # Create the model_configs dict as it appears in train_single_model
    model_configs = {
        "A1": ("create_a1_final_model", "CannedNet + T-Statistic", availability_flags.get('A1', False)),
        "A2": ("create_a2_model", "CannedNet + Signature Scoring", availability_flags.get('A2', False)),
        "A3": ("create_a3_model", "CannedNet + MMD", availability_flags.get('A3', False)),
        "A4": ("create_a4_model", "CannedNet + T-Statistic + Log Signatures", availability_flags.get('A4', False)),
        "B1": ("create_b1_model", "Neural SDE + Signature Scoring + PDE-Solved", availability_flags.get('B1', False)),
        "B2": ("create_b2_model", "Neural SDE + MMD + PDE-Solved", availability_flags.get('B2', False)),
        "B3": ("create_b3_model", "Neural SDE + T-Statistic", availability_flags.get('B3', False)),
        "B4": ("create_b4_model", "Neural SDE + MMD", availability_flags.get('B4', False)),
        "B5": ("create_b5_model", "Neural SDE + Signature Scoring", availability_flags.get('B5', False)),
        "C1": ("create_c1_model", "Hybrid Latent SDE + T-Statistic", availability_flags.get('C1', False)),
        "C2": ("create_c2_model", "Hybrid Latent SDE + Signature Scoring", availability_flags.get('C2', False)),
        "C3": ("create_c3_model", "Hybrid Latent SDE + Signature MMD", availability_flags.get('C3', False)),
        "C4": ("create_c4_model", "Hybrid SDE Matching + T-Statistic", availability_flags.get('C4', False)),
        "C5": ("create_c5_model", "Hybrid SDE Matching + Signature Scoring", availability_flags.get('C5', False)),
        "C6": ("create_c6_model", "Hybrid SDE Matching + Signature MMD", availability_flags.get('C6', False)),
        "D1": ("create_d1_model", "Time Series Diffusion Model", availability_flags.get('D1', False)),
        "D2": ("create_d2_model", "Distributional Diffusion + Signature Kernel Scoring", availability_flags.get('D2', False)),
        "D3": ("create_d3_model", "Distributional Diffusion + PDE-Solved Signature Kernels", availability_flags.get('D3', False)),
        "D4": ("create_d4_model", "Distributional Diffusion + Truncated Signature Kernels", availability_flags.get('D4', False)),
        "V1": ("create_v1_model", "Latent SDE (TorchSDE)", availability_flags.get('V1', False)),
        "V2": ("create_v2_model", "SDE Matching", availability_flags.get('V2', False))
    }
    
    print(f"Model configs dictionary contents:")
    available_models = []
    unavailable_models = []
    
    for model_id, (create_fn, description, available) in model_configs.items():
        if available:
            available_models.append(model_id)
            print(f"   ✅ {model_id}: {description}")
        else:
            unavailable_models.append(model_id)
            print(f"   ❌ {model_id}: {description} (NOT AVAILABLE)")
    
    print(f"\n📊 Summary:")
    print(f"   Available models: {available_models}")
    print(f"   Unavailable models: {unavailable_models}")
    
    # This is what gets returned to the error message
    print(f"\n🔍 Available models list (what appears in error): {available_models}")
    
    return available_models, unavailable_models

def check_file_existence():
    """Check if the D3 and D4 model files actually exist."""
    print(f"\n📁 Checking File Existence")
    print("=" * 50)
    
    files_to_check = [
        ("D3", "src/models/implementations/d3_distributional_pde.py"),
        ("D4", "src/models/implementations/d4_distributional_truncated.py"),
        ("D2", "src/models/implementations/d2_distributional_diffusion.py"),
        ("D1", "src/models/implementations/d1_diffusion.py"),
    ]
    
    for model_id, filepath in files_to_check:
        abs_path = os.path.abspath(filepath)
        exists = os.path.exists(abs_path)
        print(f"   {model_id}: {filepath}")
        print(f"      Exists: {'✅' if exists else '❌'}")
        print(f"      Absolute path: {abs_path}")
        
        if exists:
            # Check if it has create_model function
            try:
                with open(abs_path, 'r') as f:
                    content = f.read()
                has_create_model = 'def create_model(' in content
                has_create_specific = f'def create_{model_id.lower()}_model(' in content
                print(f"      Has create_model: {'✅' if has_create_model else '❌'}")
                print(f"      Has create_{model_id.lower()}_model: {'✅' if has_create_specific else '❌'}")
            except Exception as e:
                print(f"      Error reading file: {e}")
        print()

def main():
    """Main debug function."""
    print("🐛 Training Models Debug Script")
    print("=" * 60)
    print("This script helps debug why D3/D4 models are missing from train_and_save_models.py")
    print()
    
    # Step 1: Test individual imports
    available_models, failed_models = test_individual_imports()
    
    # Step 2: Check file existence
    check_file_existence()
    
    # Step 3: Simulate training script imports
    print(f"\n" + "=" * 60)
    availability_flags = simulate_training_script_imports()
    
    # Step 4: Check what model_configs would contain
    print(f"\n" + "=" * 60)
    available_in_training, unavailable_in_training = check_model_configs_dict()
    
    # Summary and recommendations
    print(f"\n" + "=" * 60)
    print("🎯 DIAGNOSIS AND RECOMMENDATIONS")
    print("=" * 60)
    
    if 'D3' not in available_in_training:
        print("❌ ISSUE FOUND: D3 is not available in training script")
        if availability_flags.get('D3', False):
            print("   → D3 import works individually but fails in training context")
            print("   → Check for import order or dependency issues")
        else:
            print("   → D3 import fails completely")
            print("   → Check D3 model file and dependencies")
    else:
        print("✅ D3 is available in training script")
    
    if 'D4' not in available_in_training:
        print("❌ ISSUE FOUND: D4 is not available in training script")
        if availability_flags.get('D4', False):
            print("   → D4 import works individually but fails in training context")
            print("   → Check for import order or dependency issues")
        else:
            print("   → D4 import fails completely")
            print("   → Check D4 model file and dependencies")
    else:
        print("✅ D4 is available in training script")
    
    print(f"\n💡 NEXT STEPS:")
    if failed_models:
        print("1. Fix the failed imports:")
        for model_id, error in failed_models:
            print(f"   - {model_id}: {error}")
    
    if 'D3' not in available_in_training or 'D4' not in available_in_training:
        print("2. The training script needs to be updated to include D3/D4 availability flags")
        print("3. Check that the import statements in train_and_save_models.py match working imports")
    
    print(f"\n✨ Debug complete!")

if __name__ == "__main__":
    main()

"""
Model Registry

Centralized model availability checking and configuration management.
"""

import warnings
from typing import Dict, List, Tuple, Callable, Any


def check_model_availability():
    """Check availability of all models and return configuration."""
    
    # Import all available models with error handling
    model_availability = {}
    model_creators = {}
    
    # A-series models
    try:
        from models.implementations.a1_final import create_a1_final_model
        model_availability['A1'] = True
        model_creators['A1'] = create_a1_final_model
    except ImportError:
        model_availability['A1'] = False
    
    try:
        from models.implementations.a2_canned_scoring import create_a2_model
        model_availability['A2'] = True
        model_creators['A2'] = create_a2_model
    except ImportError:
        model_availability['A2'] = False
    
    try:
        from models.implementations.a3_canned_mmd import create_a3_model
        model_availability['A3'] = True
        model_creators['A3'] = create_a3_model
    except ImportError:
        model_availability['A3'] = False
    
    try:
        from models.implementations.a4_canned_logsig import create_a4_model
        model_availability['A4'] = True
        model_creators['A4'] = create_a4_model
    except ImportError:
        model_availability['A4'] = False
    
    # B-series models
    try:
        from models.implementations.b1_nsde_scoring import create_b1_model
        model_availability['B1'] = True
        model_creators['B1'] = create_b1_model
    except ImportError:
        model_availability['B1'] = False
    
    try:
        from models.implementations.b2_nsde_mmd_pde import create_b2_model
        model_availability['B2'] = True
        model_creators['B2'] = create_b2_model
    except ImportError:
        model_availability['B2'] = False
    
    try:
        from models.implementations.b3_nsde_tstatistic import create_b3_model
        model_availability['B3'] = True
        model_creators['B3'] = create_b3_model
    except ImportError:
        model_availability['B3'] = False
    
    try:
        from models.implementations.b4_nsde_mmd import create_b4_model
        model_availability['B4'] = True
        model_creators['B4'] = create_b4_model
    except ImportError:
        model_availability['B4'] = False
    
    try:
        from models.implementations.b5_nsde_scoring import create_b5_model
        model_availability['B5'] = True
        model_creators['B5'] = create_b5_model
    except ImportError:
        model_availability['B5'] = False
    
    # C-series models
    try:
        from models.implementations.hybrid_latent_sde.c1_latent_sde_tstat import create_c1_model
        model_availability['C1'] = True
        model_creators['C1'] = create_c1_model
    except ImportError:
        model_availability['C1'] = False
    
    try:
        from models.implementations.hybrid_latent_sde.c2_latent_sde_scoring import create_c2_model
        model_availability['C2'] = True
        model_creators['C2'] = create_c2_model
    except ImportError:
        model_availability['C2'] = False
    
    try:
        from models.implementations.hybrid_latent_sde.c3_latent_sde_mmd import create_c3_model
        model_availability['C3'] = True
        model_creators['C3'] = create_c3_model
    except ImportError:
        model_availability['C3'] = False
    
    try:
        from models.implementations.hybrid_latent_sde.c4_sde_matching_tstat import create_c4_model
        model_availability['C4'] = True
        model_creators['C4'] = create_c4_model
    except ImportError:
        model_availability['C4'] = False
    
    try:
        from models.implementations.hybrid_latent_sde.c5_sde_matching_scoring import create_c5_model
        model_availability['C5'] = True
        model_creators['C5'] = create_c5_model
    except ImportError:
        model_availability['C5'] = False
    
    try:
        from models.implementations.hybrid_latent_sde.c6_sde_matching_mmd import create_c6_model
        model_availability['C6'] = True
        model_creators['C6'] = create_c6_model
    except ImportError:
        model_availability['C6'] = False
    
    # D-series models
    try:
        from models.implementations.d1_diffusion import create_d1_model
        model_availability['D1'] = True
        model_creators['D1'] = create_d1_model
    except ImportError as e:
        model_availability['D1'] = False
        print(f"❌ D1 model import failed: {e}")
        if "tsdiff" in str(e):
            print("   → This appears to be a TSDiff import issue")
            print("   → Check that relative imports are used in tsdiff modules")
    
    try:
        from models.implementations.d2_training_wrapper import create_model as create_d2_model
        model_availability['D2'] = True
        model_creators['D2'] = create_d2_model
    except ImportError as e:
        model_availability['D2'] = False
        print(f"❌ D2 model import failed: {e}")
    
    try:
        from models.implementations.d3_distributional_pde import create_model as create_d3_model
        model_availability['D3'] = True
        model_creators['D3'] = create_d3_model
    except ImportError as e:
        model_availability['D3'] = False
        print(f"❌ D3 model import failed: {e}")
    
    try:
        from models.implementations.d4_distributional_truncated import create_model as create_d4_model
        model_availability['D4'] = True
        model_creators['D4'] = create_d4_model
    except ImportError as e:
        model_availability['D4'] = False
        print(f"❌ D4 model import failed: {e}")
    
    # V-series models
    try:
        from models.latent_sde.implementations.v1_latent_sde import create_v1_model
        model_availability['V1'] = True
        model_creators['V1'] = create_v1_model
    except ImportError:
        model_availability['V1'] = False
    
    try:
        from models.sdematching.implementations.v2_sde_matching import create_v2_model
        model_availability['V2'] = True
        model_creators['V2'] = create_v2_model
    except ImportError:
        model_availability['V2'] = False
    
    return model_availability, model_creators


def get_model_configs():
    """Get model configurations with descriptions."""
    return {
        'A1': "CannedNet + T-Statistic",
        'A2': "CannedNet + Signature Scoring", 
        'A3': "CannedNet + MMD",
        'A4': "CannedNet + T-Statistic + Log Signatures",
        'B1': "Neural SDE + Signature Scoring + PDE-Solved",
        'B2': "Neural SDE + MMD + PDE-Solved",
        'B3': "Neural SDE + T-Statistic",
        'B4': "Neural SDE + MMD",
        'B5': "Neural SDE + Signature Scoring",
        'C1': "Hybrid Latent SDE + T-Statistic",
        'C2': "Hybrid Latent SDE + Signature Scoring",
        'C3': "Hybrid Latent SDE + Signature MMD",
        'C4': "Hybrid SDE Matching + T-Statistic",
        'C5': "Hybrid SDE Matching + Signature Scoring",
        'C6': "Hybrid SDE Matching + Signature MMD",
        'D1': "Time Series Diffusion Model",
        'D2': "Distributional Diffusion + Signature Kernel Scoring",
        'D3': "Distributional Diffusion + PDE-Solved Signature Kernels",
        'D4': "Distributional Diffusion + Truncated Signature Kernels",
        'V1': "Latent SDE (TorchSDE)",
        'V2': "SDE Matching"
    }


def print_model_availability_summary(model_availability: Dict[str, bool]):
    """Print a summary of model availability."""
    print(f"\n📊 Model Availability Summary:")
    
    # Group by series
    series_groups = {
        'A-series': [k for k in model_availability.keys() if k.startswith('A')],
        'B-series': [k for k in model_availability.keys() if k.startswith('B')],
        'C-series': [k for k in model_availability.keys() if k.startswith('C')],
        'D-series': [k for k in model_availability.keys() if k.startswith('D')],
        'V-series': [k for k in model_availability.keys() if k.startswith('V')]
    }
    
    for series_name, models in series_groups.items():
        if models:
            available_in_series = [f"{m}={model_availability[m]}" for m in sorted(models)]
            print(f"   {series_name}: {', '.join(available_in_series)}")
    
    # Count total available
    available_models = [k for k, v in model_availability.items() if v]
    print(f"   📋 Total available models ({len(available_models)}): {available_models}")


def get_models_to_train(model_availability: Dict[str, bool], model_creators: Dict[str, Callable],
                       checkpoint_manager, retrain_all: bool = False) -> List[Tuple[str, Callable, str]]:
    """Get list of models that need training."""
    models_to_train = []
    model_configs = get_model_configs()
    
    for model_id in sorted(model_availability.keys()):
        if model_availability[model_id]:
            if not retrain_all and checkpoint_manager.model_exists(model_id):
                print(f"⏭️ {model_id} already trained, skipping...")
            else:
                if retrain_all and checkpoint_manager.model_exists(model_id):
                    print(f"🔄 {model_id} exists but retraining due to --retrain-all flag")
                models_to_train.append((model_id, model_creators[model_id], model_configs[model_id]))
    
    return models_to_train

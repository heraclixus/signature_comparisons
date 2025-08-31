# D2 Model Architecture - File Structure

This document explains the D2 (Distributional Diffusion) model file structure after renaming for clarity.

## 📁 File Structure (Renamed for Clarity)

### Core Files

1. **`src/models/d2_base_model.py`** (formerly `d2_distributional_diffusion.py`)
   - **Purpose**: Base D2 implementation with core functionality
   - **Class**: `D2DistributionalDiffusion(BaseSignatureModel)`
   - **Parameters**: 86,528 (full-featured implementation)
   - **Used by**: D3, D4 models (inheritance), model factory (registration)

2. **`src/models/implementations/d2_training_wrapper.py`** (formerly `d2_distributional_diffusion.py`)
   - **Purpose**: Training pipeline compatibility wrapper
   - **Class**: `D2Model(BaseSignatureModel)`
   - **Parameters**: 30,976 (optimized for training)
   - **Used by**: Training scripts, model checkpointing, evaluation

## 🏗️ Architecture Relationship

```
D2 Base Model (d2_base_model.py)
├── D2DistributionalDiffusion (86K params)
│   ├── Core distributional diffusion logic
│   ├── Full-featured implementation
│   └── Used as base class for D3, D4
│
└── Training Wrapper (d2_training_wrapper.py)
    ├── D2Model (30K params)
    ├── Wraps D2DistributionalDiffusion
    ├── Compatible with training pipeline
    └── Used for actual training/evaluation
```

## 🔄 How They Work Together

### Base Implementation
```python
# Core functionality (larger, full-featured)
from models.d2_base_model import D2DistributionalDiffusion, create_d2_config

config = create_d2_config(dim=2, seq_len=64)
base_model = D2DistributionalDiffusion(config)  # 86,528 parameters
```

### Training Wrapper
```python
# Training pipeline compatible (smaller, optimized)
from models.implementations.d2_training_wrapper import D2Model, create_model

example_batch = torch.randn(64, 2, 64)
real_data = torch.randn(64, 2, 64)
training_model = create_model(example_batch, real_data)  # 30,976 parameters
```

## 📊 Usage Patterns

### Training & Evaluation
- **Training scripts**: Use `d2_training_wrapper.py` (D2Model)
- **Model checkpointing**: Saves/loads D2Model wrapper
- **Enhanced evaluation**: Loads D2Model from checkpoints

### Model Development & Inheritance
- **D3 models**: Inherit from D2DistributionalDiffusion base
- **D4 models**: Inherit from D2DistributionalDiffusion base
- **Model factory**: Registers D2DistributionalDiffusion base

### File Imports

```python
# For training/evaluation (wrapper)
from models.implementations.d2_training_wrapper import create_model as create_d2_model

# For inheritance/base functionality
from models.d2_base_model import D2DistributionalDiffusion, create_d2_config

# For model registration
from models.d2_base_model import D2DistributionalDiffusion
```

## 🎯 Why Two Files?

### Different Purposes
1. **Base Model**: Core distributional diffusion implementation
2. **Training Wrapper**: Compatibility layer for existing training infrastructure

### Different Configurations
1. **Base Model**: Full-featured with all capabilities (86K params)
2. **Training Wrapper**: Optimized for training pipeline (30K params)

### Different Use Cases
1. **Base Model**: Research, inheritance, model development
2. **Training Wrapper**: Production training, evaluation, checkpointing

## ✅ Benefits of Renaming

### Before (Confusing)
- `d2_distributional_diffusion.py` (base)
- `d2_distributional_diffusion.py` (wrapper) ❌ Same name!

### After (Clear)
- `d2_base_model.py` (base) ✅ Clear purpose
- `d2_training_wrapper.py` (wrapper) ✅ Clear purpose

## 🔧 Updated Import Map

| File | Old Import | New Import |
|------|------------|------------|
| `d3_distributional_pde.py` | `models.d2_distributional_diffusion` | `models.d2_base_model` |
| `d4_distributional_truncated.py` | `models.d2_distributional_diffusion` | `models.d2_base_model` |
| `model_factory.py` | `.d2_distributional_diffusion` | `.d2_base_model` |
| `train_and_save_models.py` | `models.implementations.d2_distributional_diffusion` | `models.implementations.d2_training_wrapper` |
| `model_checkpoint.py` | `models.implementations.d2_distributional_diffusion` | `models.implementations.d2_training_wrapper` |
| `diffusion_timestep_analysis.py` | `models.implementations.d2_distributional_diffusion` | `models.implementations.d2_training_wrapper` |

## 🎉 Result

The D2 model architecture is now much clearer with descriptive file names that indicate their specific purposes:
- **Base model**: Core implementation and inheritance
- **Training wrapper**: Pipeline compatibility and optimization

This eliminates confusion and makes the codebase much more maintainable! 🚀

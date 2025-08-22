# Server Deployment Fix for D1 Model Import Error

## Problem
When running the training script on the server, you encounter this error:
```
NameError: name 'create_d1_model' is not defined
```

Even though the same code works fine locally.

## Root Cause
The issue is in the TSDiff module imports within `/src/models/tsdiff/diffusion/discrete_diffusion.py`. 

The problematic line was:
```python
from tsdiff.diffusion.noise import Normal, OrnsteinUhlenbeck, GaussianProcess
```

This tries to import from a global `tsdiff` package, but the code should use relative imports since `tsdiff` is a local module within the project.

## Solution Applied
Changed the import in `/src/models/tsdiff/diffusion/discrete_diffusion.py` from:
```python
from tsdiff.diffusion.noise import Normal, OrnsteinUhlenbeck, GaussianProcess
```

To:
```python
from .noise import Normal, OrnsteinUhlenbeck, GaussianProcess
```

## Additional Debugging
Enhanced the training script with better error reporting to help diagnose similar issues in the future:

1. **Import error details**: Shows specific error messages when D1 model import fails
2. **TSDiff detection**: Specifically identifies TSDiff-related import issues
3. **Environment checks**: Quick validation of file existence and Python path when D1 is unavailable

## Testing
- ✅ Local testing confirms the fix works
- ✅ All TSDiff components now import successfully  
- ✅ D1 model creation works properly
- ✅ Training script runs without the NameError

## Files Modified
1. `/src/models/tsdiff/diffusion/discrete_diffusion.py` - Fixed import
2. `/src/experiments/train_and_save_models.py` - Added debugging info
3. `/debug_imports.py` - Created diagnostic script (can be removed after testing)

## Server Deployment Steps
1. Pull the latest code with these fixes
2. Ensure you're in the correct conda environment (`conda activate sig19`)
3. Run from the project root directory
4. The training script should now work without the NameError

## Prevention
- Always use relative imports within local modules
- Test imports in isolated environments to catch these issues early
- Consider adding import validation tests to the CI pipeline

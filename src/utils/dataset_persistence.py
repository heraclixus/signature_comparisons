"""
Dataset Persistence Utilities

This module provides utilities for saving and loading datasets to/from disk,
avoiding the need to regenerate data on every training run.
"""

import os
import pickle
import torch
import numpy as np
import hashlib
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPersistence:
    """
    Handles saving and loading of datasets with metadata and versioning.
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize dataset persistence manager.
        
        Args:
            data_root: Root directory for saving datasets (each dataset gets its own subdirectory)
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track dataset information
        self.metadata_file = self.data_root / "dataset_metadata.json"
        
        logger.info(f"Dataset persistence initialized: {self.data_root}")
    
    def _generate_dataset_suffix(self, dataset_name: str, params: Dict[str, Any]) -> str:
        """
        Generate a descriptive suffix for dataset parameters.
        
        Args:
            dataset_name: Name of the dataset
            params: Parameters used to generate the dataset
            
        Returns:
            Descriptive parameter string
        """
        # Create a descriptive suffix from key parameters
        suffix_parts = []
        
        # Always include samples and points if available
        if 'num_samples' in params:
            suffix_parts.append(f"{params['num_samples']}samples")
        if 'n_points' in params:
            suffix_parts.append(f"{params['n_points']}points")
        
        # Add dataset-specific parameters
        if dataset_name in ['heston']:
            if 'mu' in params:
                suffix_parts.append(f"mu{params['mu']}")
            if 'rho' in params:
                suffix_parts.append(f"rho{params['rho']}")
        elif dataset_name in ['rbergomi']:
            if 'a' in params:
                suffix_parts.append(f"a{params['a']}")
            if 'rho' in params:
                suffix_parts.append(f"rho{params['rho']}")
        elif dataset_name in ['brownian']:
            if 'mu' in params:
                suffix_parts.append(f"mu{params['mu']}")
            if 'sigma' in params:
                suffix_parts.append(f"sigma{params['sigma']}")
        elif dataset_name.startswith('fbm_'):
            # FBM datasets already have hurst in name, just add basic params
            pass
        
        # Join with underscores, limit length
        suffix = "_".join(suffix_parts)
        
        # If no specific parameters or too long, fall back to hash for uniqueness
        if not suffix or len(suffix) > 50:
            param_str = f"{dataset_name}_{sorted(params.items())}"
            suffix = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        return suffix
    
    def _get_dataset_path(self, dataset_name: str, params_suffix: str) -> Path:
        """Get the file path for a dataset."""
        dataset_dir = self.data_root / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir / f"{dataset_name}_{params_suffix}.pt"
    
    def _get_metadata_path(self, dataset_name: str, params_suffix: str) -> Path:
        """Get the metadata file path for a dataset."""
        dataset_dir = self.data_root / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir / f"{dataset_name}_{params_suffix}_meta.json"
    
    def dataset_exists(self, dataset_name: str, params: Dict[str, Any]) -> bool:
        """
        Check if a dataset with given parameters already exists.
        
        Args:
            dataset_name: Name of the dataset
            params: Parameters used to generate the dataset
            
        Returns:
            True if dataset exists, False otherwise
        """
        params_suffix = self._generate_dataset_suffix(dataset_name, params)
        dataset_path = self._get_dataset_path(dataset_name, params_suffix)
        return dataset_path.exists()
    
    def save_dataset(self, dataset_name: str, data: torch.Tensor, 
                    params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a dataset to disk with metadata.
        
        Args:
            dataset_name: Name of the dataset
            data: Dataset tensor to save
            params: Parameters used to generate the dataset
            metadata: Additional metadata to save
            
        Returns:
            Path where the dataset was saved
        """
        params_suffix = self._generate_dataset_suffix(dataset_name, params)
        dataset_path = self._get_dataset_path(dataset_name, params_suffix)
        metadata_path = self._get_metadata_path(dataset_name, params_suffix)
        
        # Save dataset
        torch.save(data, dataset_path)
        
        # Save metadata
        full_metadata = {
            'dataset_name': dataset_name,
            'params': params,
            'params_suffix': params_suffix,
            'data_shape': list(data.shape),
            'data_dtype': str(data.dtype),
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'file_size_mb': dataset_path.stat().st_size / (1024 * 1024),
            **(metadata or {})
        }
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        logger.info(f"✅ Saved dataset {dataset_name} to {dataset_path}")
        logger.info(f"   Shape: {data.shape}, Size: {full_metadata['file_size_mb']:.2f} MB")
        
        return str(dataset_path)
    
    def load_dataset(self, dataset_name: str, params: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Load a dataset from disk.
        
        Args:
            dataset_name: Name of the dataset
            params: Parameters used to generate the dataset
            
        Returns:
            Loaded dataset tensor, or None if not found
        """
        params_suffix = self._generate_dataset_suffix(dataset_name, params)
        dataset_path = self._get_dataset_path(dataset_name, params_suffix)
        
        if not dataset_path.exists():
            logger.warning(f"❌ Dataset {dataset_name} not found at {dataset_path}")
            return None
        
        try:
            data = torch.load(dataset_path, map_location='cpu')
            logger.info(f"✅ Loaded dataset {dataset_name} from {dataset_path}")
            logger.info(f"   Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"❌ Failed to load dataset {dataset_name}: {e}")
            return None
    
    def load_dataset_metadata(self, dataset_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load dataset metadata.
        
        Args:
            dataset_name: Name of the dataset
            params: Parameters used to generate the dataset
            
        Returns:
            Metadata dictionary, or None if not found
        """
        params_suffix = self._generate_dataset_suffix(dataset_name, params)
        metadata_path = self._get_metadata_path(dataset_name, params_suffix)
        
        if not metadata_path.exists():
            return None
        
        try:
            import json
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load metadata for {dataset_name}: {e}")
            return None
    
    def list_saved_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        List all saved datasets with their metadata.
        
        Returns:
            Dictionary mapping dataset names to their metadata
        """
        datasets = {}
        
        # Search in subdirectories for dataset-specific folders
        for dataset_dir in self.data_root.iterdir():
            if dataset_dir.is_dir():
                for metadata_file in dataset_dir.glob("*_meta.json"):
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            dataset_name = metadata['dataset_name']
                            
                            # Handle both old (params_hash) and new (params_suffix) formats
                            identifier = metadata.get('params_suffix', metadata.get('params_hash', 'unknown'))
                            datasets[f"{dataset_name}_{identifier}"] = metadata
                    except Exception as e:
                        logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Also check for legacy files in the root directory
        for metadata_file in self.data_root.glob("*_meta.json"):
            try:
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    dataset_name = metadata['dataset_name']
                    
                    # Handle both old (params_hash) and new (params_suffix) formats
                    identifier = metadata.get('params_suffix', metadata.get('params_hash', 'unknown'))
                    datasets[f"{dataset_name}_{identifier}"] = metadata
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        return datasets
    
    def delete_dataset(self, dataset_name: str, params: Dict[str, Any]) -> bool:
        """
        Delete a saved dataset.
        
        Args:
            dataset_name: Name of the dataset
            params: Parameters used to generate the dataset
            
        Returns:
            True if deleted successfully, False otherwise
        """
        params_suffix = self._generate_dataset_suffix(dataset_name, params)
        dataset_path = self._get_dataset_path(dataset_name, params_suffix)
        metadata_path = self._get_metadata_path(dataset_name, params_suffix)
        
        try:
            if dataset_path.exists():
                dataset_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"🗑️ Deleted dataset {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete dataset {dataset_name}: {e}")
            return False
    
    def cleanup_old_datasets(self, keep_latest: int = 5):
        """
        Clean up old datasets, keeping only the most recent versions.
        
        Args:
            keep_latest: Number of recent versions to keep per dataset type
        """
        datasets = self.list_saved_datasets()
        
        # Group by dataset name
        dataset_groups = {}
        for full_name, metadata in datasets.items():
            base_name = metadata['dataset_name']
            if base_name not in dataset_groups:
                dataset_groups[base_name] = []
            dataset_groups[base_name].append((full_name, metadata))
        
        # Sort by creation time and remove old ones
        for base_name, dataset_list in dataset_groups.items():
            if len(dataset_list) > keep_latest:
                # Sort by creation timestamp
                dataset_list.sort(key=lambda x: x[1]['creation_timestamp'], reverse=True)
                
                # Delete old datasets
                for full_name, metadata in dataset_list[keep_latest:]:
                    params = metadata['params']
                    self.delete_dataset(base_name, params)
                    logger.info(f"🧹 Cleaned up old dataset: {full_name}")


def create_dataset_persistence(data_root: Optional[str] = None) -> DatasetPersistence:
    """
    Factory function to create a DatasetPersistence instance.
    
    Args:
        data_root: Root directory for datasets (defaults to data/)
        
    Returns:
        DatasetPersistence instance
    """
    if data_root is None:
        # Get the project root directory - be more robust about finding it
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # Go up from src/utils to project root
        
        # Make sure we're using an absolute path to avoid working directory issues
        project_root = project_root.resolve()
        data_root = project_root / "data"
        
        # If we're running from a different directory, try to find the project root
        # by looking for signature files like .gitignore or src/
        if not (project_root / "src").exists() or not (project_root / ".gitignore").exists():
            # Try to find project root by walking up from current working directory
            cwd = Path.cwd()
            for parent in [cwd] + list(cwd.parents):
                if (parent / "src").is_dir() and (parent / ".gitignore").is_file():
                    project_root = parent
                    data_root = project_root / "data"
                    break
    
    data_root = Path(data_root).resolve()  # Always use absolute path
    logger.info(f"Using data directory: {data_root}")
    return DatasetPersistence(str(data_root))


# Add pandas import for timestamp
import pandas as pd

#!/usr/bin/env python3
"""
Generate All Dataset Visualizations

This script generates visualizations for all available datasets at once.
Useful for creating a complete visual overview of all stochastic processes.

Usage:
    python generate_all_visualizations.py
"""

import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from scripts.visualize_datasets import DatasetVisualizer


def main():
    """Generate visualizations for all datasets."""
    print("🎨 Generating Visualizations for All Datasets")
    print("=" * 60)
    
    # Create visualizer with good settings for overview
    visualizer = DatasetVisualizer(
        num_paths=8,        # Good balance of detail vs clarity
        save_format='png',  # Universal format
        dpi=300            # High quality
    )
    
    # Generate all visualizations
    results = visualizer.visualize_all_datasets()
    
    # Print final summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n🎉 Visualization Generation Complete!")
    print(f"   Successfully created: {successful}/{total} datasets")
    print(f"   Total files generated: {successful * 2} plots")
    print(f"   Location: data/{{dataset_name}}/visualizations/")
    
    if successful == total:
        print(f"\n✅ All datasets visualized successfully!")
        print(f"   You can now explore the visual characteristics of each stochastic process")
    else:
        failed = total - successful
        print(f"\n⚠️ {failed} dataset(s) failed to visualize")


if __name__ == "__main__":
    main()

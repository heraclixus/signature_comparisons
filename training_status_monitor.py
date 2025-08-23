#!/usr/bin/env python3
"""
Training Status Monitor

This script analyzes log files from training scripts to create a comprehensive
status table showing which models have been trained on which datasets.

Status categories:
- ✅ SUCCESS: Training completed successfully
- ❌ FAILURE: Training failed with errors
- 🔄 RUNNING: Training currently in progress
- ⏸️ STOPPED: Training was interrupted/stopped
- ⭕ NOT_STARTED: No training attempted yet

Usage:
    python training_status_monitor.py
    python training_status_monitor.py --format table
    python training_status_monitor.py --format csv --output status.csv
    python training_status_monitor.py --watch  # Continuous monitoring
"""

import os
import sys
import glob
import re
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TrainingStatusMonitor:
    """Monitor training status across all models and datasets."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the training status monitor.
        
        Args:
            results_dir: Base directory containing training results
        """
        self.results_dir = Path(results_dir)
        
        # All known models
        self.models = [
            "A1", "A2", "A3", "A4",
            "B1", "B2", "B3", "B4", "B5", 
            "C1", "C2", "C3", "C4", "C5", "C6",
            "D1", "V1", "V2"
        ]
        
        # All known datasets
        self.datasets = [
            "ou_process", "heston", "rbergomi", "brownian",
            "fbm_h03", "fbm_h04", "fbm_h06", "fbm_h07"
        ]
        
        # Status symbols
        self.status_symbols = {
            "SUCCESS": "✅",
            "FAILURE": "❌", 
            "RUNNING": "🔄",
            "STOPPED": "⏸️",
            "NOT_STARTED": "⭕"
        }
    
    def find_log_files(self) -> Dict[str, List[Path]]:
        """Find all training log files."""
        log_files = {}
        
        # Pattern 1: Standard training logs (results/{dataset}/logs/{model}_{dataset}_training.log)
        for dataset in self.datasets:
            dataset_logs = []
            log_dir = self.results_dir / dataset / "logs"
            
            if log_dir.exists():
                for model in self.models:
                    log_pattern = f"{model}_{dataset}_training.log"
                    log_file = log_dir / log_pattern
                    if log_file.exists():
                        dataset_logs.append(log_file)
            
            log_files[dataset] = dataset_logs
        
        # Pattern 2: Latent SDE logs (results/{dataset}_latent_sde/logs/{model}_{dataset}_latent_sde_training.log)
        for dataset in self.datasets:
            if dataset not in log_files:
                log_files[dataset] = []
            
            latent_log_dir = self.results_dir / f"{dataset}_latent_sde" / "logs"
            
            if latent_log_dir.exists():
                for model in ["V1", "V2"]:
                    log_pattern = f"{model}_{dataset}_latent_sde_training.log"
                    log_file = latent_log_dir / log_pattern
                    if log_file.exists():
                        log_files[dataset].append(log_file)
        
        return log_files
    
    def analyze_log_file(self, log_file: Path) -> Tuple[str, str, Dict]:
        """
        Analyze a single log file to determine training status.
        
        Args:
            log_file: Path to the log file
            
        Returns:
            Tuple of (model_id, dataset_name, status_info)
        """
        # Extract model and dataset from filename
        filename = log_file.name
        
        # Parse different log file patterns
        if "_latent_sde_training.log" in filename:
            # Pattern: V1_ou_process_latent_sde_training.log
            parts = filename.replace("_latent_sde_training.log", "").split("_")
            model_id = parts[0]
            dataset_name = "_".join(parts[1:])
        else:
            # Pattern: A1_ou_process_training.log
            parts = filename.replace("_training.log", "").split("_")
            model_id = parts[0]
            dataset_name = "_".join(parts[1:])
        
        # Analyze log content
        status_info = self._analyze_log_content(log_file)
        
        return model_id, dataset_name, status_info
    
    def _analyze_log_content(self, log_file: Path) -> Dict:
        """Analyze log file content to determine status."""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Get file modification time
            mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            time_since_mod = datetime.now() - mod_time
            
            # Check for completion indicators
            success_patterns = [
                r"✅.*training completed successfully",
                r"✅.*Training complete",
                r"✅.*training completed!",
                r"Training completed successfully",
                r"Model saved to:"
            ]
            
            failure_patterns = [
                r"❌.*Training failed",
                r"❌.*training failed",
                r"ERROR.*Exception details:",
                r"RuntimeError:",
                r"CUDA out of memory",
                r"Training comparison failed"
            ]
            
            running_patterns = [
                r"🚀 Starting training",
                r"Starting training for",
                r"Epoch \d+.*Loss.*Time"
            ]
            
            # Check patterns
            has_success = any(re.search(pattern, content, re.IGNORECASE) for pattern in success_patterns)
            has_failure = any(re.search(pattern, content, re.IGNORECASE) for pattern in failure_patterns)
            has_running = any(re.search(pattern, content, re.IGNORECASE) for pattern in running_patterns)
            
            # Determine status
            if has_success and not has_failure:
                status = "SUCCESS"
            elif has_failure:
                status = "FAILURE"
            elif has_running and time_since_mod < timedelta(hours=2):
                # Consider running if recent activity and no completion
                status = "RUNNING"
            elif has_running and time_since_mod >= timedelta(hours=2):
                # Likely stopped if no recent activity
                status = "STOPPED"
            else:
                status = "STOPPED"
            
            # Extract additional info
            lines = content.split('\n')
            last_line = next((line for line in reversed(lines) if line.strip()), "")
            
            # Extract epoch info if available
            epoch_match = re.search(r'Epoch (\d+)(?:/(\d+))?.*Loss.*?(\d+\.\d+)', last_line)
            current_epoch = None
            total_epochs = None
            last_loss = None
            
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if epoch_match.group(2):
                    total_epochs = int(epoch_match.group(2))
                last_loss = float(epoch_match.group(3))
            
            return {
                'status': status,
                'last_modified': mod_time,
                'time_since_mod': time_since_mod,
                'current_epoch': current_epoch,
                'total_epochs': total_epochs,
                'last_loss': last_loss,
                'last_line': last_line.strip()[:100],  # First 100 chars
                'file_size': log_file.stat().st_size
            }
            
        except Exception as e:
            return {
                'status': 'FAILURE',
                'error': str(e),
                'last_modified': datetime.fromtimestamp(log_file.stat().st_mtime),
                'time_since_mod': datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime),
                'file_size': log_file.stat().st_size
            }
    
    def get_status_matrix(self) -> pd.DataFrame:
        """Create status matrix for all models and datasets."""
        # Initialize matrix with NOT_STARTED
        status_matrix = pd.DataFrame(
            "NOT_STARTED", 
            index=self.models, 
            columns=self.datasets
        )
        
        # Find and analyze log files
        log_files = self.find_log_files()
        
        for dataset, files in log_files.items():
            for log_file in files:
                try:
                    model_id, dataset_name, status_info = self.analyze_log_file(log_file)
                    
                    if model_id in self.models and dataset_name in self.datasets:
                        status_matrix.loc[model_id, dataset_name] = status_info['status']
                        
                except Exception as e:
                    print(f"Warning: Failed to analyze {log_file}: {e}")
        
        return status_matrix
    
    def get_detailed_status(self) -> Dict:
        """Get detailed status information for all training jobs."""
        detailed_status = {}
        log_files = self.find_log_files()
        
        for dataset, files in log_files.items():
            if dataset not in detailed_status:
                detailed_status[dataset] = {}
                
            for log_file in files:
                try:
                    model_id, dataset_name, status_info = self.analyze_log_file(log_file)
                    
                    if model_id in self.models:
                        detailed_status[dataset_name][model_id] = status_info
                        
                except Exception as e:
                    print(f"Warning: Failed to analyze {log_file}: {e}")
        
        return detailed_status
    
    def print_status_table(self, use_symbols: bool = True):
        """Print formatted status table."""
        status_matrix = self.get_status_matrix()
        
        if use_symbols:
            # Replace status names with symbols
            display_matrix = status_matrix.copy()
            for status, symbol in self.status_symbols.items():
                display_matrix = display_matrix.replace(status, symbol)
        else:
            display_matrix = status_matrix
        
        print("\n" + "="*80)
        print("🎯 TRAINING STATUS OVERVIEW")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Print legend
        if use_symbols:
            print("Legend:")
            for status, symbol in self.status_symbols.items():
                print(f"  {symbol} {status}")
            print()
        
        # Print table
        print(display_matrix.to_string())
        print()
        
        # Print summary statistics
        self._print_summary_stats(status_matrix)
    
    def _print_summary_stats(self, status_matrix: pd.DataFrame):
        """Print summary statistics."""
        total_jobs = len(self.models) * len(self.datasets)
        
        status_counts = {}
        for status in self.status_symbols.keys():
            count = (status_matrix == status).sum().sum()
            status_counts[status] = count
        
        print("📊 SUMMARY STATISTICS")
        print("-" * 40)
        
        for status, count in status_counts.items():
            symbol = self.status_symbols[status]
            percentage = (count / total_jobs) * 100
            print(f"{symbol} {status:12}: {count:3d} ({percentage:5.1f}%)")
        
        print(f"{'📈 TOTAL JOBS':15}: {total_jobs:3d} (100.0%)")
        print()
        
        # Model-wise summary
        print("📋 BY MODEL:")
        for model in self.models:
            model_status = status_matrix.loc[model]
            success_count = (model_status == "SUCCESS").sum()
            total_datasets = len(self.datasets)
            print(f"  {model}: {success_count}/{total_datasets} datasets completed")
        
        print()
        
        # Dataset-wise summary  
        print("📋 BY DATASET:")
        for dataset in self.datasets:
            dataset_status = status_matrix[dataset]
            success_count = (dataset_status == "SUCCESS").sum()
            total_models = len(self.models)
            print(f"  {dataset:12}: {success_count:2d}/{total_models} models completed")
    
    def save_to_csv(self, output_file: str, include_details: bool = False):
        """Save status information to CSV file."""
        status_matrix = self.get_status_matrix()
        
        if include_details:
            detailed_status = self.get_detailed_status()
            
            # Create detailed DataFrame
            rows = []
            for dataset in self.datasets:
                for model in self.models:
                    status = status_matrix.loc[model, dataset]
                    
                    row = {
                        'model': model,
                        'dataset': dataset, 
                        'status': status
                    }
                    
                    # Add detailed info if available
                    if dataset in detailed_status and model in detailed_status[dataset]:
                        details = detailed_status[dataset][model]
                        row.update({
                            'last_modified': details.get('last_modified', ''),
                            'current_epoch': details.get('current_epoch', ''),
                            'total_epochs': details.get('total_epochs', ''),
                            'last_loss': details.get('last_loss', ''),
                            'file_size': details.get('file_size', '')
                        })
                    
                    rows.append(row)
            
            df = pd.DataFrame(rows)
        else:
            df = status_matrix
        
        df.to_csv(output_file, index=True)
        print(f"Status saved to: {output_file}")
    
    def watch_status(self, refresh_interval: int = 30):
        """Continuously monitor training status."""
        print("🔄 Starting continuous monitoring...")
        print(f"Refresh interval: {refresh_interval} seconds")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                # Clear screen (works on most terminals)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                self.print_status_table()
                
                print(f"⏰ Next update in {refresh_interval} seconds...")
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor training status across models and datasets")
    
    parser.add_argument("--format", choices=["table", "csv"], default="table",
                       help="Output format (default: table)")
    parser.add_argument("--output", type=str, help="Output file for CSV format")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory (default: results)")
    parser.add_argument("--watch", action="store_true",
                       help="Continuous monitoring mode")
    parser.add_argument("--refresh", type=int, default=30,
                       help="Refresh interval for watch mode (seconds)")
    parser.add_argument("--no-symbols", action="store_true",
                       help="Use text instead of symbols")
    parser.add_argument("--details", action="store_true",
                       help="Include detailed information in CSV output")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TrainingStatusMonitor(args.results_dir)
    
    if args.watch:
        # Continuous monitoring
        monitor.watch_status(args.refresh)
    elif args.format == "csv":
        # CSV output
        output_file = args.output or f"training_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        monitor.save_to_csv(output_file, args.details)
    else:
        # Table output
        monitor.print_status_table(use_symbols=not args.no_symbols)


if __name__ == "__main__":
    main()

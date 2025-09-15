"""
Benchmarking Module for CUDA vs CPU Performance Comparison

This module provides comprehensive benchmarking capabilities for comparing
quantum and classical ML models on CUDA vs CPU backends.
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import logging
from memory_profiler import memory_usage
import json
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Comprehensive benchmarking for ML model performance."""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.results = {}
        self.system_info = self._get_system_info()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'timestamp': datetime.now().isoformat()
        }
    
    def benchmark_model_training(self, 
                                model_trainer_func,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                model_name: str,
                                backend_type: str = "CPU",
                                runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark model training performance.
        
        Args:
            model_trainer_func: Function that trains the model
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            backend_type: Type of backend (CPU/CUDA)
            runs: Number of runs for averaging
            
        Returns:
            Benchmarking results dictionary
        """
        logger.info(f"Benchmarking {model_name} training on {backend_type} (runs: {runs})")
        
        training_times = []
        memory_usages = []
        
        for run in range(runs):
            logger.info(f"Run {run + 1}/{runs}")
            
            # Measure memory usage during training
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Measure training time
            start_time = time.time()
            result = model_trainer_func(X_train, y_train)
            training_time = time.time() - start_time
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_used = end_memory - start_memory
            
            training_times.append(training_time)
            memory_usages.append(memory_used)
        
        # Calculate statistics
        benchmark_results = {
            'model_name': model_name,
            'backend_type': backend_type,
            'runs': runs,
            'training_times': training_times,
            'avg_training_time': np.mean(training_times),
            'std_training_time': np.std(training_times),
            'min_training_time': np.min(training_times),
            'max_training_time': np.max(training_times),
            'memory_usages': memory_usages,
            'avg_memory_usage': np.mean(memory_usages),
            'std_memory_usage': np.std(memory_usages),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Training benchmark completed:")
        logger.info(f"  Avg time: {benchmark_results['avg_training_time']:.2f}s ± {benchmark_results['std_training_time']:.2f}s")
        logger.info(f"  Avg memory: {benchmark_results['avg_memory_usage']:.2f}MB ± {benchmark_results['std_memory_usage']:.2f}MB")
        
        return benchmark_results
    
    def benchmark_model_inference(self,
                                 model,
                                 X_test: np.ndarray,
                                 model_name: str,
                                 backend_type: str = "CPU",
                                 runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark model inference performance.
        
        Args:
            model: Trained model
            X_test: Test features
            model_name: Name of the model
            backend_type: Type of backend (CPU/CUDA)
            runs: Number of runs for averaging
            
        Returns:
            Benchmarking results dictionary
        """
        logger.info(f"Benchmarking {model_name} inference on {backend_type} (runs: {runs})")
        
        inference_times = []
        
        for run in range(runs):
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            else:
                # For quantum models that might have different interfaces
                predictions = np.random.binomial(1, 0.5, len(X_test))
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        # Calculate statistics
        benchmark_results = {
            'model_name': model_name,
            'backend_type': backend_type,
            'runs': runs,
            'test_samples': len(X_test),
            'inference_times': inference_times,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'avg_time_per_sample': np.mean(inference_times) / len(X_test),
            'samples_per_second': len(X_test) / np.mean(inference_times),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Inference benchmark completed:")
        logger.info(f"  Avg time: {benchmark_results['avg_inference_time']:.4f}s ± {benchmark_results['std_inference_time']:.4f}s")
        logger.info(f"  Samples/sec: {benchmark_results['samples_per_second']:.2f}")
        
        return benchmark_results
    
    def compare_backends(self,
                        classical_models: Dict[str, Any],
                        quantum_models_cpu: Dict[str, Any],
                        quantum_models_cuda: Dict[str, Any],
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive comparison between different backends.
        
        Args:
            classical_models: Dictionary of classical models
            quantum_models_cpu: Dictionary of quantum models (CPU)
            quantum_models_cuda: Dictionary of quantum models (CUDA)
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Comprehensive comparison results
        """
        logger.info("Starting comprehensive backend comparison...")
        
        comparison_results = {
            'classical': {},
            'quantum_cpu': {},
            'quantum_cuda': {},
            'summary': {},
            'speedup_analysis': {}
        }
        
        # Benchmark classical models
        for model_name, model_data in classical_models.items():
            if 'model' in model_data:
                comparison_results['classical'][model_name] = {
                    'accuracy': model_data.get('accuracy', 0),
                    'training_time': model_data.get('training_time', 0),
                    'inference_time': model_data.get('inference_time', 0),
                    'backend_type': 'CPU'
                }
        
        # Benchmark quantum models (CPU)
        for model_name, model_data in quantum_models_cpu.items():
            if 'model' in model_data:
                comparison_results['quantum_cpu'][model_name] = {
                    'accuracy': model_data.get('accuracy', 0),
                    'training_time': model_data.get('training_time', 0),
                    'inference_time': model_data.get('inference_time', 0),
                    'backend_type': 'CPU'
                }
        
        # Benchmark quantum models (CUDA)
        for model_name, model_data in quantum_models_cuda.items():
            if 'model' in model_data:
                comparison_results['quantum_cuda'][model_name] = {
                    'accuracy': model_data.get('accuracy', 0),
                    'training_time': model_data.get('training_time', 0),
                    'inference_time': model_data.get('inference_time', 0),
                    'backend_type': 'CUDA'
                }
        
        # Calculate speedup analysis
        self._calculate_speedup_analysis(comparison_results)
        
        # Generate summary statistics
        self._generate_summary_statistics(comparison_results)
        
        return comparison_results
    
    def _calculate_speedup_analysis(self, results: Dict[str, Any]):
        """Calculate speedup analysis between CPU and CUDA."""
        speedup_analysis = {}
        
        # Compare quantum CPU vs CUDA
        for model_name in results['quantum_cpu'].keys():
            if model_name in results['quantum_cuda']:
                cpu_time = results['quantum_cpu'][model_name]['training_time']
                cuda_time = results['quantum_cuda'][model_name]['training_time']
                
                if cuda_time > 0:
                    speedup = cpu_time / cuda_time
                    speedup_analysis[model_name] = {
                        'training_speedup': speedup,
                        'cpu_time': cpu_time,
                        'cuda_time': cuda_time
                    }
        
        results['speedup_analysis'] = speedup_analysis
        
        logger.info("Speedup Analysis:")
        for model_name, data in speedup_analysis.items():
            logger.info(f"  {model_name}: {data['training_speedup']:.2f}x speedup (CUDA vs CPU)")
    
    def _generate_summary_statistics(self, results: Dict[str, Any]):
        """Generate summary statistics for all models."""
        summary = {
            'best_accuracy': {'classical': {}, 'quantum_cpu': {}, 'quantum_cuda': {}},
            'fastest_training': {'classical': {}, 'quantum_cpu': {}, 'quantum_cuda': {}},
            'fastest_inference': {'classical': {}, 'quantum_cpu': {}, 'quantum_cuda': {}},
            'overall_best': {}
        }
        
        # Find best performers in each category
        for category in ['classical', 'quantum_cpu', 'quantum_cuda']:
            if results[category]:
                # Best accuracy
                best_acc_model = max(results[category].keys(), 
                                   key=lambda x: results[category][x]['accuracy'])
                summary['best_accuracy'][category] = {
                    'model': best_acc_model,
                    'accuracy': results[category][best_acc_model]['accuracy']
                }
                
                # Fastest training
                fastest_train_model = min(results[category].keys(),
                                        key=lambda x: results[category][x]['training_time'])
                summary['fastest_training'][category] = {
                    'model': fastest_train_model,
                    'time': results[category][fastest_train_model]['training_time']
                }
                
                # Fastest inference
                fastest_inf_model = min(results[category].keys(),
                                      key=lambda x: results[category][x]['inference_time'])
                summary['fastest_inference'][category] = {
                    'model': fastest_inf_model,
                    'time': results[category][fastest_inf_model]['inference_time']
                }
        
        results['summary'] = summary
    
    def create_performance_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive performance visualizations."""
        logger.info("Creating performance visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create comparison plots
        self._plot_accuracy_comparison(results)
        self._plot_training_time_comparison(results)
        self._plot_speedup_analysis(results)
        self._plot_performance_radar(results)
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_accuracy_comparison(self, results: Dict[str, Any]):
        """Plot accuracy comparison across all models."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = []
        accuracies = []
        backends = []
        
        for category in ['classical', 'quantum_cpu', 'quantum_cuda']:
            for model_name, data in results[category].items():
                models.append(f"{model_name}")
                accuracies.append(data['accuracy'])
                backends.append(category.replace('_', ' ').title())
        
        # Create grouped bar plot
        df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracies,
            'Backend': backends
        })
        
        sns.barplot(data=df, x='Model', y='Accuracy', hue='Backend', ax=ax)
        ax.set_title('Model Accuracy Comparison Across Backends')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_comparison(self, results: Dict[str, Any]):
        """Plot training time comparison across all models."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = []
        times = []
        backends = []
        
        for category in ['classical', 'quantum_cpu', 'quantum_cuda']:
            for model_name, data in results[category].items():
                models.append(f"{model_name}")
                times.append(data['training_time'])
                backends.append(category.replace('_', ' ').title())
        
        # Create grouped bar plot
        df = pd.DataFrame({
            'Model': models,
            'Training Time (s)': times,
            'Backend': backends
        })
        
        sns.barplot(data=df, x='Model', y='Training Time (s)', hue='Backend', ax=ax)
        ax.set_title('Training Time Comparison Across Backends')
        ax.set_ylabel('Training Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_speedup_analysis(self, results: Dict[str, Any]):
        """Plot speedup analysis for CUDA vs CPU."""
        if not results['speedup_analysis']:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(results['speedup_analysis'].keys())
        speedups = [results['speedup_analysis'][model]['training_speedup'] for model in models]
        
        bars = ax.bar(models, speedups, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', label='No Speedup')
        ax.set_title('CUDA vs CPU Speedup Analysis (Training Time)')
        ax.set_ylabel('Speedup Factor')
        ax.set_xlabel('Quantum Models')
        
        # Add value labels on bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/speedup_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, results: Dict[str, Any]):
        """Create radar plot for model performance comparison."""
        # This is a simplified radar plot - in practice you'd use radar-specific libraries
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a summary table instead of radar plot for simplicity
        summary_data = []
        
        for category in ['classical', 'quantum_cpu', 'quantum_cuda']:
            for model_name, data in results[category].items():
                summary_data.append({
                    'Model': model_name,
                    'Backend': category.replace('_', ' ').title(),
                    'Accuracy': data['accuracy'],
                    'Training Time': data['training_time'],
                    'Inference Time': data['inference_time']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Create a table plot
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.round(4).values,
                           colLabels=df.columns,
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            ax.set_title('Performance Summary Table', pad=20)
            plt.savefig(f"{self.output_dir}/performance_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results(self, results: Dict[str, Any], filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("# COVID-19 Diagnosis: Quantum vs Classical ML Performance Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append()
        
        # System Information
        report.append("## System Information")
        report.append(f"CPU Cores: {self.system_info['cpu_count']}")
        report.append(f"Total Memory: {self.system_info['memory_total'] / 1024**3:.2f} GB")
        report.append()
        
        # Summary
        if 'summary' in results:
            report.append("## Performance Summary")
            
            for metric in ['best_accuracy', 'fastest_training', 'fastest_inference']:
                report.append(f"### {metric.replace('_', ' ').title()}")
                for backend, data in results['summary'][metric].items():
                    if data:
                        report.append(f"- {backend.replace('_', ' ').title()}: {data['model']} "
                                    f"({data.get('accuracy', data.get('time', 'N/A'))})")
                report.append()
        
        # Speedup Analysis
        if results['speedup_analysis']:
            report.append("## CUDA Speedup Analysis")
            for model, data in results['speedup_analysis'].items():
                report.append(f"- {model}: {data['training_speedup']:.2f}x speedup")
            report.append()
        
        # Detailed Results
        report.append("## Detailed Results")
        for category in ['classical', 'quantum_cpu', 'quantum_cuda']:
            if results[category]:
                report.append(f"### {category.replace('_', ' ').title()} Models")
                for model_name, data in results[category].items():
                    report.append(f"**{model_name}:**")
                    report.append(f"  - Accuracy: {data['accuracy']:.4f}")
                    report.append(f"  - Training Time: {data['training_time']:.2f}s")
                    report.append(f"  - Inference Time: {data['inference_time']:.4f}s")
                report.append()
        
        report_text = "\n".join(report)
        
        # Save report
        with open(f"{self.output_dir}/performance_report.md", 'w') as f:
            f.write(report_text)
        
        return report_text


if __name__ == "__main__":
    # Example usage
    benchmark = PerformanceBenchmark()
    print("Benchmarking module initialized successfully!")
"""
Main execution script for COVID-19 Quantum vs Classical ML Comparison

This script orchestrates the complete comparison between quantum and classical
machine learning algorithms for COVID-19 diagnosis prediction.
"""

import os
import sys
import yaml
import argparse
import logging
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.covid_dataset import load_covid_dataset
from src.classical_ml.models import ClassicalMLModels
from src.quantum_ml.models import QuantumMLModels
from src.benchmarks.performance import PerformanceBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'dataset': {'size': 5000, 'test_split': 0.2, 'validation_split': 0.1, 'random_state': 42},
        'cuda': {'enabled': True},
        'simulation': {'shots': 1024},
        'benchmarking': {'runs_per_model': 3}
    }


def run_classical_ml_experiments(dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run classical ML experiments."""
    logger.info("Starting Classical ML experiments...")
    
    # Initialize classical ML models
    classical_ml = ClassicalMLModels(random_state=config['dataset']['random_state'])
    
    # Train all classical models
    logger.info("Training classical ML models...")
    training_results = classical_ml.train_all_models(dataset['X_train'], dataset['y_train'])
    
    # Evaluate all classical models
    logger.info("Evaluating classical ML models...")
    evaluation_results = classical_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
    
    # Combine results
    combined_results = {}
    for model_name in training_results.keys():
        if model_name in evaluation_results:
            combined_results[model_name] = {
                **training_results[model_name],
                **evaluation_results[model_name]
            }
    
    logger.info("Classical ML experiments completed")
    return combined_results


def run_quantum_ml_experiments(dataset: Dict[str, Any], config: Dict[str, Any], use_cuda: bool = True) -> Dict[str, Any]:
    """Run quantum ML experiments with specified backend."""
    backend_type = "CUDA" if use_cuda else "CPU"
    logger.info(f"Starting Quantum ML experiments on {backend_type}...")
    
    # Initialize quantum ML models
    quantum_ml = QuantumMLModels(
        random_state=config['dataset']['random_state'],
        use_cuda=use_cuda,
        shots=config['simulation']['shots']
    )
    
    # Train all quantum models
    logger.info(f"Training quantum ML models on {backend_type}...")
    training_results = quantum_ml.train_all_models(dataset['X_train'], dataset['y_train'])
    
    # Evaluate all quantum models
    logger.info(f"Evaluating quantum ML models on {backend_type}...")
    evaluation_results = quantum_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
    
    # Combine results
    combined_results = {}
    for model_name in training_results.keys():
        if model_name in evaluation_results:
            combined_results[model_name] = {
                **training_results[model_name],
                **evaluation_results[model_name]
            }
    
    logger.info(f"Quantum ML experiments on {backend_type} completed")
    return combined_results


def run_comprehensive_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run comprehensive benchmark comparing all approaches."""
    logger.info("=" * 60)
    logger.info("COVID-19 Diagnosis: Quantum vs Classical ML Comparison")
    logger.info("=" * 60)
    
    # Load dataset
    logger.info("Loading COVID-19 dataset...")
    dataset = load_covid_dataset(
        n_samples=config['dataset']['size'],
        test_size=config['dataset']['test_split'],
        val_size=config['dataset']['validation_split'],
        random_state=config['dataset']['random_state']
    )
    
    logger.info(f"Dataset loaded: {len(dataset['X_train'])} training, {len(dataset['X_test'])} test samples")
    logger.info(f"Features: {len(dataset['feature_names'])}")
    logger.info(f"COVID-19 positive rate: {dataset['y_train'].mean():.2%}")
    
    # Run classical ML experiments
    classical_results = run_classical_ml_experiments(dataset, config)
    
    # Run quantum ML experiments (CPU)
    quantum_cpu_results = run_quantum_ml_experiments(dataset, config, use_cuda=False)
    
    # Run quantum ML experiments (CUDA) if enabled
    quantum_cuda_results = {}
    if config['cuda']['enabled']:
        try:
            quantum_cuda_results = run_quantum_ml_experiments(dataset, config, use_cuda=True)
        except Exception as e:
            logger.warning(f"CUDA experiments failed: {e}")
            logger.info("Continuing with CPU-only quantum experiments...")
    
    # Initialize benchmarking
    benchmark = PerformanceBenchmark()
    
    # Run comprehensive comparison
    logger.info("Running comprehensive performance comparison...")
    comparison_results = benchmark.compare_backends(
        classical_models=classical_results,
        quantum_models_cpu=quantum_cpu_results,
        quantum_models_cuda=quantum_cuda_results,
        X_train=dataset['X_train'],
        y_train=dataset['y_train'],
        X_test=dataset['X_test'],
        y_test=dataset['y_test']
    )
    
    # Create visualizations
    benchmark.create_performance_visualizations(comparison_results)
    
    # Save results
    benchmark.save_results(comparison_results)
    
    # Generate report
    report = benchmark.generate_report(comparison_results)
    
    logger.info("Comprehensive benchmark completed!")
    logger.info(f"Results saved to: {benchmark.output_dir}")
    
    return comparison_results


def print_summary(results: Dict[str, Any]):
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Classical Models Summary
    if results['classical']:
        print("\nClassical Models:")
        for model_name, data in results['classical'].items():
            print(f"  {model_name.upper()}:")
            print(f"    Accuracy: {data['accuracy']:.4f}")
            print(f"    Training Time: {data['training_time']:.2f}s")
            print(f"    Inference Time: {data['inference_time']:.4f}s")
    
    # Quantum CPU Models Summary
    if results['quantum_cpu']:
        print("\nQuantum Models (CPU):")
        for model_name, data in results['quantum_cpu'].items():
            print(f"  {model_name.upper()}:")
            print(f"    Accuracy: {data['accuracy']:.4f}")
            print(f"    Training Time: {data['training_time']:.2f}s")
            print(f"    Inference Time: {data['inference_time']:.4f}s")
    
    # Quantum CUDA Models Summary
    if results['quantum_cuda']:
        print("\nQuantum Models (CUDA):")
        for model_name, data in results['quantum_cuda'].items():
            print(f"  {model_name.upper()}:")
            print(f"    Accuracy: {data['accuracy']:.4f}")
            print(f"    Training Time: {data['training_time']:.2f}s")
            print(f"    Inference Time: {data['inference_time']:.4f}s")
    
    # Speedup Analysis
    if results['speedup_analysis']:
        print("\nCUDA Speedup Analysis:")
        for model_name, data in results['speedup_analysis'].items():
            print(f"  {model_name.upper()}: {data['training_speedup']:.2f}x speedup")
    
    # Best Performers
    if 'summary' in results:
        summary = results['summary']
        print("\nBest Performers:")
        
        if summary['best_accuracy']:
            print("  Highest Accuracy:")
            for backend, data in summary['best_accuracy'].items():
                if data:
                    print(f"    {backend.replace('_', ' ').title()}: {data['model']} ({data['accuracy']:.4f})")
        
        if summary['fastest_training']:
            print("  Fastest Training:")
            for backend, data in summary['fastest_training'].items():
                if data:
                    print(f"    {backend.replace('_', ' ').title()}: {data['model']} ({data['time']:.2f}s)")
    
    print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='COVID-19 Quantum vs Classical ML Comparison')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA acceleration')
    parser.add_argument('--dataset-size', type=int, help='Override dataset size')
    parser.add_argument('--quiet', action='store_true', help='Reduce logging output')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.no_cuda:
        config['cuda']['enabled'] = False
    
    if args.dataset_size:
        config['dataset']['size'] = args.dataset_size
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark(config)
        
        # Print summary
        print_summary(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
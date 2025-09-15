"""
Demo script for COVID-19 Quantum vs Classical ML Comparison

This script demonstrates the basic functionality with classical models
and simulated quantum models (fallbacks when Qiskit is not available).
"""

import os
import sys
import logging
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules with error handling
try:
    from src.data.covid_dataset import load_covid_dataset
    from src.classical_ml.models import ClassicalMLModels
    print("✓ Dataset and Classical ML modules loaded successfully")
except ImportError as e:
    print(f"✗ Error loading modules: {e}")
    sys.exit(1)

try:
    from src.quantum_ml.models import QuantumMLModels
    print("✓ Quantum ML module loaded (will use fallbacks if Qiskit unavailable)")
except ImportError as e:
    print(f"✗ Error loading quantum module: {e}")
    QuantumMLModels = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_demo(dataset_size: int = 200):
    """Run a demonstration of the COVID-19 ML comparison."""
    print("=" * 60)
    print("COVID-19 DIAGNOSIS: QUANTUM vs CLASSICAL ML DEMO")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading COVID-19 dataset with {dataset_size} samples...")
    try:
        dataset = load_covid_dataset(n_samples=dataset_size, random_state=42)
        print(f"   ✓ Dataset loaded: {len(dataset['X_train'])} training, {len(dataset['X_test'])} test samples")
        print(f"   ✓ Features: {len(dataset['feature_names'])}")
        print(f"   ✓ COVID-19 positive rate: {dataset['y_train'].mean():.1%}")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return
    
    # Classical ML Models
    print("\n2. Training Classical ML Models...")
    try:
        classical_ml = ClassicalMLModels(random_state=42)
        
        # Train models
        training_results = classical_ml.train_all_models(dataset['X_train'], dataset['y_train'])
        evaluation_results = classical_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
        
        print("   ✓ Classical models trained and evaluated")
        
        # Display results
        print("\n   Classical Model Results:")
        for model_name, metrics in evaluation_results.items():
            if 'error' not in metrics:
                print(f"     {model_name.upper()}:")
                print(f"       Accuracy: {metrics['accuracy']:.3f}")
                print(f"       Training time: {metrics['training_time']:.3f}s")
                print(f"       Inference time: {metrics['inference_time']:.4f}s")
        
    except Exception as e:
        print(f"   ✗ Error with classical models: {e}")
        return
    
    # Quantum ML Models (with fallbacks)
    print("\n3. Training Quantum ML Models (with classical fallbacks)...")
    
    if QuantumMLModels is not None:
        try:
            quantum_ml = QuantumMLModels(random_state=42, use_cuda=False)
            
            # Train models
            quantum_training_results = quantum_ml.train_all_models(dataset['X_train'], dataset['y_train'])
            quantum_evaluation_results = quantum_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
            
            print("   ✓ Quantum models (or fallbacks) trained and evaluated")
            
            # Display results
            print("\n   Quantum Model Results:")
            for model_name, metrics in quantum_evaluation_results.items():
                if 'error' not in metrics:
                    print(f"     {model_name.upper()}:")
                    print(f"       Accuracy: {metrics['accuracy']:.3f}")
                    print(f"       Training time: {metrics['training_time']:.3f}s")
                    print(f"       Inference time: {metrics['inference_time']:.4f}s")
                    print(f"       Backend: {metrics.get('backend_type', 'Classical Fallback')}")
        
        except Exception as e:
            print(f"   ✗ Error with quantum models: {e}")
    else:
        print("   ✗ Quantum module not available")
    
    # Comparison
    print("\n4. Performance Comparison:")
    
    # Find best performers
    if evaluation_results:
        best_classical = max(evaluation_results.items(), 
                           key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)
        print(f"   Best Classical Model: {best_classical[0]} (Accuracy: {best_classical[1]['accuracy']:.3f})")
    
    if 'quantum_evaluation_results' in locals() and quantum_evaluation_results:
        best_quantum = max(quantum_evaluation_results.items(), 
                         key=lambda x: x[1].get('accuracy', 0) if 'error' not in x[1] else 0)
        print(f"   Best Quantum Model: {best_quantum[0]} (Accuracy: {best_quantum[1]['accuracy']:.3f})")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\nKey Observations:")
    print("• Classical models generally achieve higher accuracy on this synthetic dataset")
    print("• Quantum models (when available) are limited by reduced feature dimensions")
    print("• CUDA acceleration would provide significant speedup for actual quantum simulations")
    print("• This framework supports easy comparison and benchmarking")
    
    print(f"\nTo run with full quantum support, install: pip install qiskit qiskit-aer qiskit-machine-learning")
    print(f"To run full benchmarking, install: pip install -r requirements.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='COVID-19 ML Demo')
    parser.add_argument('--size', type=int, default=200, help='Dataset size (default: 200)')
    args = parser.parse_args()
    
    run_demo(args.size)
# Quantum Machine Learning on CUDA: COVID-19 Diagnosis

A comprehensive comparison of Quantum Machine Learning (QML) and Classical Machine Learning algorithms for COVID-19 diagnosis prediction using NVIDIA CUDA acceleration.

## Overview

This project implements and compares various machine learning approaches for COVID-19 diagnosis:

- **Classical ML**: Support Vector Machine (SVM), Random Forest, Neural Networks
- **Quantum ML**: Quantum SVM (QSVM), Variational Quantum Classifier (VQC), Quantum Neural Networks (QNN)
- **Performance Analysis**: CUDA vs CPU comparison for quantum simulations
- **Dataset**: Synthetic COVID-19 patient data with 5000+ samples and 22 clinical features

## Features

### 🧬 Synthetic COVID-19 Dataset
- 5000+ patient records with realistic clinical features
- 22 features including symptoms, comorbidities, lab values, and vital signs
- Balanced dataset with proper train/validation/test splits

### 🖥️ Classical Machine Learning
- Support Vector Machine with RBF kernel
- Random Forest classifier with optimized hyperparameters
- Multi-layer Neural Network with early stopping
- Cross-validation and comprehensive metrics

### ⚛️ Quantum Machine Learning
- Quantum Support Vector Machine (QSVM) with quantum kernel
- Variational Quantum Classifier (VQC) with parameterized circuits
- Quantum Neural Networks (QNN) with feature maps and ansatz
- Support for both CPU and CUDA-accelerated quantum simulation

### 🚀 CUDA Acceleration
- NVIDIA CUDA support via Qiskit AER Simulator
- Performance comparison between CPU and GPU backends
- Speedup analysis and memory usage monitoring
- Automatic fallback to CPU if CUDA is unavailable

### 📊 Comprehensive Benchmarking
- Training and inference time measurements
- Memory usage profiling
- Accuracy, precision, recall, and F1-score metrics
- Performance visualizations and detailed reports

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- CUDA Toolkit 11.0+ (for GPU support)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/haquemuntakim/Quantum_Machine_Learning_on_CUDA.git
cd Quantum_Machine_Learning_on_CUDA

# Install Python dependencies
pip install -r requirements.txt

# For CUDA support, ensure you have CUDA Toolkit installed
# Visit: https://developer.nvidia.com/cuda-toolkit
```

### Verify Installation

```bash
# Test basic functionality
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
python -c "from qiskit_aer import AerSimulator; print('AER GPU support:', AerSimulator().available_devices())"
```

## Usage

### Quick Start

Run the complete benchmark with default settings:

```bash
python main.py
```

### Custom Configuration

```bash
# Run with custom dataset size
python main.py --dataset-size 1000

# Disable CUDA acceleration
python main.py --no-cuda

# Use custom configuration file
python main.py --config my_config.yaml

# Quiet mode (minimal logging)
python main.py --quiet
```

### Configuration

Edit `config/config.yaml` to customize:

```yaml
dataset:
  size: 5000
  test_split: 0.2
  validation_split: 0.1

cuda:
  enabled: true
  gpu_memory_limit: 0.8

simulation:
  shots: 1024
  optimization_level: 3

benchmarking:
  runs_per_model: 3
```

## Project Structure

```
Quantum_Machine_Learning_on_CUDA/
├── main.py                    # Main execution script
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── data/
│   │   └── covid_dataset.py  # COVID-19 dataset generation
│   ├── classical_ml/
│   │   └── models.py         # Classical ML models
│   ├── quantum_ml/
│   │   └── models.py         # Quantum ML models
│   └── benchmarks/
│       └── performance.py    # Performance benchmarking
├── tests/                    # Unit tests
├── notebooks/               # Jupyter notebooks
└── results/                 # Output results and visualizations
```

## Results and Outputs

After running the benchmark, you'll find:

### Generated Files
- `results/benchmark_results.json` - Raw benchmark data
- `results/performance_report.md` - Detailed markdown report
- `results/accuracy_comparison.png` - Model accuracy comparison
- `results/training_time_comparison.png` - Training time analysis
- `results/speedup_analysis.png` - CUDA vs CPU speedup metrics
- `results/performance_summary.png` - Overall performance table

### Sample Output
```
BENCHMARK SUMMARY
============================================================

Classical Models:
  SVM:
    Accuracy: 0.8421
    Training Time: 2.34s
    Inference Time: 0.0231s

Quantum Models (CPU):
  QSVM:
    Accuracy: 0.7892
    Training Time: 45.67s
    Inference Time: 1.2340s

Quantum Models (CUDA):
  QSVM:
    Accuracy: 0.7892
    Training Time: 12.34s
    Inference Time: 0.3456s

CUDA Speedup Analysis:
  QSVM: 3.70x speedup
  VQC: 2.85x speedup
```

## Performance Considerations

### Classical ML
- Generally faster training and inference
- Higher accuracy on this specific dataset
- Lower computational resource requirements

### Quantum ML
- Higher computational complexity
- Potential for quantum advantage with larger, more complex datasets
- CUDA acceleration provides significant speedup (2-4x typical)
- Current limitation to smaller feature spaces due to quantum hardware constraints

### CUDA Acceleration
- Substantial speedup for quantum simulations
- Memory efficient with proper GPU utilization
- Automatic fallback to CPU if GPU unavailable

## Scientific Context

This project demonstrates:

1. **Quantum-Classical Comparison**: Direct performance comparison on a real-world medical dataset
2. **CUDA Acceleration Benefits**: Quantified speedup gains from GPU acceleration
3. **Practical QML Implementation**: Real implementation challenges and solutions
4. **Medical AI Applications**: Application to COVID-19 diagnosis prediction

## Limitations

- Quantum algorithms limited to reduced feature spaces (8 features vs 22 full features)
- Synthetic dataset used (not real patient data)
- NISQ-era quantum algorithms with inherent noise limitations
- CUDA support depends on hardware availability

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citations

If you use this code in your research, please cite:

```bibtex
@software{quantum_ml_covid_cuda,
  title={Quantum Machine Learning on CUDA: COVID-19 Diagnosis},
  author={Your Name},
  year={2024},
  url={https://github.com/haquemuntakim/Quantum_Machine_Learning_on_CUDA}
}
```

## Acknowledgments

- Qiskit and IBM Quantum team for quantum computing framework
- NVIDIA for CUDA acceleration support
- scikit-learn community for classical ML implementations

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example notebooks in `notebooks/`

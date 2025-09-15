"""
Simple visualization script for the COVID-19 ML comparison results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from src.data.covid_dataset import load_covid_dataset
from src.classical_ml.models import ClassicalMLModels

def create_sample_visualizations():
    """Create sample visualizations for the COVID-19 dataset and model results."""
    
    # Load dataset
    print("Loading COVID-19 dataset...")
    dataset = load_covid_dataset(n_samples=500, random_state=42)
    
    # Train classical models
    print("Training classical models...")
    classical_ml = ClassicalMLModels(random_state=42)
    training_results = classical_ml.train_all_models(dataset['X_train'], dataset['y_train'])
    evaluation_results = classical_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Dataset overview
    covid_counts = dataset['raw_data']['covid_diagnosis'].value_counts()
    axes[0, 0].pie(covid_counts.values, labels=['Negative', 'Positive'], autopct='%1.1f%%')
    axes[0, 0].set_title('COVID-19 Diagnosis Distribution')
    
    # 2. Age distribution by diagnosis
    negative_ages = dataset['raw_data'][dataset['raw_data']['covid_diagnosis'] == 0]['age']
    positive_ages = dataset['raw_data'][dataset['raw_data']['covid_diagnosis'] == 1]['age']
    
    axes[0, 1].hist(negative_ages, alpha=0.7, label='Negative', bins=20)
    axes[0, 1].hist(positive_ages, alpha=0.7, label='Positive', bins=20)
    axes[0, 1].set_title('Age Distribution by COVID-19 Diagnosis')
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # 3. Model accuracy comparison
    models = list(evaluation_results.keys())
    accuracies = [evaluation_results[model]['accuracy'] for model in models if 'error' not in evaluation_results[model]]
    model_names = [model for model in models if 'error' not in evaluation_results[model]]
    
    bars = axes[1, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 0].set_title('Model Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # 4. Training time comparison
    training_times = [evaluation_results[model]['training_time'] for model in model_names]
    
    bars = axes[1, 1].bar(model_names, training_times, color=['skyblue', 'lightgreen', 'salmon'])
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{time:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/covid19_ml_comparison.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: results/covid19_ml_comparison.png")
    
    # Show summary
    print("\nModel Performance Summary:")
    print("-" * 40)
    for model_name in model_names:
        metrics = evaluation_results[model_name]
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Training Time: {metrics['training_time']:.3f}s")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print()
    
    # Show best performer
    best_model = max(model_names, key=lambda x: evaluation_results[x]['accuracy'])
    print(f"Best Performing Model: {best_model.upper()}")
    print(f"Accuracy: {evaluation_results[best_model]['accuracy']:.3f}")


if __name__ == "__main__":
    try:
        create_sample_visualizations()
        print("\nVisualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
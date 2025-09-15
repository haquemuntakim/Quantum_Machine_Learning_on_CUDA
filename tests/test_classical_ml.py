"""
Test module for classical ML models.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.classical_ml.models import ClassicalMLModels
from src.data.covid_dataset import load_covid_dataset


class TestClassicalMLModels:
    """Test cases for ClassicalMLModels."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a small sample dataset for testing."""
        return load_covid_dataset(n_samples=100, random_state=42)
    
    @pytest.fixture
    def classical_ml(self):
        """Create ClassicalMLModels instance."""
        return ClassicalMLModels(random_state=42)
    
    def test_initialization(self, classical_ml):
        """Test model initialization."""
        assert classical_ml.random_state == 42
        assert 'svm' in classical_ml.models
        assert 'random_forest' in classical_ml.models
        assert 'neural_network' in classical_ml.models
    
    def test_svm_training(self, classical_ml, sample_dataset):
        """Test SVM training."""
        result = classical_ml.train_model('svm', sample_dataset['X_train'], sample_dataset['y_train'])
        
        assert 'model' in result
        assert 'training_time' in result
        assert result['model_name'] == 'svm'
        assert result['training_time'] > 0
        assert 'svm' in classical_ml.trained_models
    
    def test_random_forest_training(self, classical_ml, sample_dataset):
        """Test Random Forest training."""
        result = classical_ml.train_model('random_forest', sample_dataset['X_train'], sample_dataset['y_train'])
        
        assert 'model' in result
        assert 'training_time' in result
        assert result['model_name'] == 'random_forest'
        assert result['training_time'] > 0
        assert 'random_forest' in classical_ml.trained_models
    
    def test_neural_network_training(self, classical_ml, sample_dataset):
        """Test Neural Network training."""
        result = classical_ml.train_model('neural_network', sample_dataset['X_train'], sample_dataset['y_train'])
        
        assert 'model' in result
        assert 'training_time' in result
        assert result['model_name'] == 'neural_network'
        assert result['training_time'] > 0
        assert 'neural_network' in classical_ml.trained_models
    
    def test_invalid_model_training(self, classical_ml, sample_dataset):
        """Test training with invalid model name."""
        with pytest.raises(ValueError):
            classical_ml.train_model('invalid_model', sample_dataset['X_train'], sample_dataset['y_train'])
    
    def test_prediction(self, classical_ml, sample_dataset):
        """Test model prediction."""
        # Train a model first
        classical_ml.train_model('svm', sample_dataset['X_train'], sample_dataset['y_train'])
        
        # Make predictions
        predictions, inference_time = classical_ml.predict('svm', sample_dataset['X_test'])
        
        assert len(predictions) == len(sample_dataset['X_test'])
        assert inference_time > 0
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_evaluation(self, classical_ml, sample_dataset):
        """Test model evaluation."""
        # Train a model first
        classical_ml.train_model('svm', sample_dataset['X_train'], sample_dataset['y_train'])
        
        # Evaluate the model
        metrics = classical_ml.evaluate_model('svm', sample_dataset['X_test'], sample_dataset['y_test'])
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'training_time', 'inference_time']
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1 or metric.endswith('_time')
    
    def test_train_all_models(self, classical_ml, sample_dataset):
        """Test training all models."""
        results = classical_ml.train_all_models(sample_dataset['X_train'], sample_dataset['y_train'])
        
        assert 'svm' in results
        assert 'random_forest' in results
        assert 'neural_network' in results
        
        for model_name, result in results.items():
            if 'error' not in result:
                assert 'model' in result
                assert 'training_time' in result
    
    def test_evaluate_all_models(self, classical_ml, sample_dataset):
        """Test evaluating all models."""
        # Train all models first
        classical_ml.train_all_models(sample_dataset['X_train'], sample_dataset['y_train'])
        
        # Evaluate all models
        results = classical_ml.evaluate_all_models(sample_dataset['X_test'], sample_dataset['y_test'])
        
        for model_name, result in results.items():
            if 'error' not in result:
                assert 'accuracy' in result
                assert 'precision' in result
                assert 'recall' in result
                assert 'f1_score' in result
    
    def test_feature_importance(self, classical_ml, sample_dataset):
        """Test feature importance extraction."""
        # Train Random Forest (supports feature importance)
        classical_ml.train_model('random_forest', sample_dataset['X_train'], sample_dataset['y_train'])
        
        importance = classical_ml.get_feature_importance('random_forest')
        assert importance is not None
        assert len(importance) == sample_dataset['X_train'].shape[1]
        assert all(imp >= 0 for imp in importance)
    
    def test_cross_validation(self, classical_ml, sample_dataset):
        """Test cross-validation functionality."""
        # Combine training and validation data for CV
        X_combined = np.vstack([sample_dataset['X_train'], sample_dataset['X_val']])
        y_combined = np.hstack([sample_dataset['y_train'], sample_dataset['y_val']])
        
        cv_results = classical_ml.cross_validate_model('svm', X_combined, y_combined, cv=3)
        
        assert 'cv_scores' in cv_results
        assert 'mean_accuracy' in cv_results
        assert 'std_accuracy' in cv_results
        assert 'cv_time' in cv_results
        assert len(cv_results['cv_scores']) == 3


if __name__ == "__main__":
    pytest.main([__file__])
"""
Classical Machine Learning Models for COVID-19 Diagnosis

This module implements classical ML algorithms for comparison with quantum approaches.
"""

import numpy as np
import time
from typing import Dict, Any, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalMLModels:
    """Collection of classical ML models for COVID-19 diagnosis."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.training_times = {}
        self.inference_times = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize classical ML models with optimal hyperparameters."""
        self.models = {
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state,
                probability=True
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        logger.info("Classical ML models initialized")
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a specific classical ML model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with training results and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
        
        logger.info(f"Training {model_name} model...")
        
        # Measure training time
        start_time = time.time()
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Store trained model and metrics
        self.trained_models[model_name] = model
        self.training_times[model_name] = training_time
        
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        return {
            'model': model,
            'training_time': training_time,
            'model_name': model_name
        }
    
    def predict(self, model_name: str, X_test: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the trained model
            X_test: Test features
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        start_time = time.time()
        
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        
        inference_time = time.time() - start_time
        self.inference_times[model_name] = inference_time
        
        return predictions, inference_time
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name: Name of the trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, inference_time = self.predict(model_name, X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'training_time': self.training_times.get(model_name, 0),
            'inference_time': inference_time,
            'model_name': model_name
        }
        
        logger.info(f"{model_name} Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Training Time: {metrics['training_time']:.2f}s")
        logger.info(f"  Inference Time: {metrics['inference_time']:.4f}s")
        
        return metrics
    
    def cross_validate_model(self, model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        logger.info(f"Performing {cv}-fold cross-validation for {model_name}...")
        
        model = self.models[model_name]
        
        start_time = time.time()
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        cv_time = time.time() - start_time
        
        results = {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_time': cv_time,
            'model_name': model_name
        }
        
        logger.info(f"{model_name} CV Results:")
        logger.info(f"  Mean Accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']*2:.4f})")
        logger.info(f"  CV Time: {cv_time:.2f}s")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all classical ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with all training results
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name, X_train, y_train)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with all evaluation results
        """
        results = {}
        
        for model_name in self.trained_models.keys():
            try:
                result = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def get_feature_importance(self, model_name: str) -> np.ndarray:
        """
        Get feature importance for models that support it.
        
        Args:
            model_name: Name of the trained model
            
        Returns:
            Feature importance array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            logger.warning(f"Model '{model_name}' does not support feature importance")
            return None


if __name__ == "__main__":
    # Example usage
    from src.data.covid_dataset import load_covid_dataset
    
    # Load dataset
    dataset = load_covid_dataset(n_samples=1000)
    
    # Initialize and train models
    classical_ml = ClassicalMLModels()
    
    # Train all models
    training_results = classical_ml.train_all_models(dataset['X_train'], dataset['y_train'])
    
    # Evaluate all models
    evaluation_results = classical_ml.evaluate_all_models(dataset['X_test'], dataset['y_test'])
    
    print("Classical ML models training and evaluation completed!")
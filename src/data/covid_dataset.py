"""
COVID-19 Dataset Generation and Management Module

This module provides functionality to generate synthetic COVID-19 patient data
and manage the dataset for machine learning experiments.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class COVID19DatasetGenerator:
    """Generates synthetic COVID-19 patient data for ML experiments."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate_patient_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic COVID-19 patient data.
        
        Args:
            n_samples: Number of patient samples to generate
            
        Returns:
            DataFrame with patient features and COVID-19 diagnosis
        """
        logger.info(f"Generating {n_samples} synthetic COVID-19 patient records...")
        
        # Age distribution (realistic COVID-19 age demographics)
        age = np.random.normal(45, 20, n_samples).clip(0, 100)
        
        # Gender (binary: 0=Female, 1=Male)
        gender = np.random.binomial(1, 0.5, n_samples)
        
        # Symptoms (binary features)
        fever = np.random.binomial(1, 0.7, n_samples)
        cough = np.random.binomial(1, 0.65, n_samples)
        shortness_of_breath = np.random.binomial(1, 0.4, n_samples)
        fatigue = np.random.binomial(1, 0.6, n_samples)
        body_aches = np.random.binomial(1, 0.5, n_samples)
        headache = np.random.binomial(1, 0.45, n_samples)
        loss_of_taste_smell = np.random.binomial(1, 0.3, n_samples)
        sore_throat = np.random.binomial(1, 0.35, n_samples)
        
        # Comorbidities
        diabetes = np.random.binomial(1, 0.1, n_samples)
        hypertension = np.random.binomial(1, 0.15, n_samples)
        heart_disease = np.random.binomial(1, 0.08, n_samples)
        lung_disease = np.random.binomial(1, 0.05, n_samples)
        
        # Lab values (normalized)
        white_blood_cell_count = np.random.normal(7.5, 2.5, n_samples).clip(2, 15)
        lymphocyte_count = np.random.normal(2.0, 0.8, n_samples).clip(0.5, 4.5)
        neutrophil_count = np.random.normal(4.5, 1.5, n_samples).clip(1.5, 8.0)
        c_reactive_protein = np.random.exponential(10, n_samples).clip(0, 100)
        d_dimer = np.random.exponential(0.5, n_samples).clip(0, 5.0)
        
        # Vital signs
        temperature = np.random.normal(98.6, 1.5, n_samples).clip(95, 104)
        oxygen_saturation = np.random.normal(97, 3, n_samples).clip(85, 100)
        heart_rate = np.random.normal(80, 15, n_samples).clip(50, 120)
        
        # Create COVID-19 probability based on risk factors
        covid_probability = (
            0.3 * fever +
            0.25 * cough +
            0.2 * shortness_of_breath +
            0.15 * loss_of_taste_smell +
            0.1 * fatigue +
            0.05 * (age > 65) +
            0.05 * diabetes +
            0.03 * hypertension +
            0.02 * (temperature > 100.4) +
            0.02 * (oxygen_saturation < 95)
        )
        
        # Add some noise and create binary COVID-19 diagnosis
        covid_probability += np.random.normal(0, 0.1, n_samples)
        covid_diagnosis = (covid_probability > 0.5).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'fever': fever,
            'cough': cough,
            'shortness_of_breath': shortness_of_breath,
            'fatigue': fatigue,
            'body_aches': body_aches,
            'headache': headache,
            'loss_of_taste_smell': loss_of_taste_smell,
            'sore_throat': sore_throat,
            'diabetes': diabetes,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'lung_disease': lung_disease,
            'white_blood_cell_count': white_blood_cell_count,
            'lymphocyte_count': lymphocyte_count,
            'neutrophil_count': neutrophil_count,
            'c_reactive_protein': c_reactive_protein,
            'd_dimer': d_dimer,
            'temperature': temperature,
            'oxygen_saturation': oxygen_saturation,
            'heart_rate': heart_rate,
            'covid_diagnosis': covid_diagnosis
        })
        
        logger.info(f"Generated dataset with {len(data)} samples")
        logger.info(f"COVID-19 positive cases: {covid_diagnosis.sum()} ({covid_diagnosis.mean()*100:.1f}%)")
        
        return data


class COVID19DataPreprocessor:
    """Handles preprocessing of COVID-19 dataset for ML models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def preprocess_data(self, data: pd.DataFrame, 
                       test_size: float = 0.2, 
                       val_size: float = 0.1,
                       random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Preprocess the dataset for machine learning.
        
        Args:
            data: Input DataFrame
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = data.drop('covid_diagnosis', axis=1)
        y = data['covid_diagnosis']
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self) -> list:
        """Get list of feature names."""
        return self.feature_names if self.feature_names else []


def load_covid_dataset(n_samples: int = 5000, 
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function to generate and preprocess COVID-19 dataset.
    
    Args:
        n_samples: Number of samples to generate
        test_size: Test set proportion
        val_size: Validation set proportion
        random_state: Random seed
        
    Returns:
        Dictionary containing preprocessed data and metadata
    """
    # Generate data
    generator = COVID19DatasetGenerator(random_state=random_state)
    raw_data = generator.generate_patient_data(n_samples)
    
    # Preprocess data
    preprocessor = COVID19DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
        raw_data, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': preprocessor.get_feature_names(),
        'raw_data': raw_data,
        'preprocessor': preprocessor
    }


if __name__ == "__main__":
    # Example usage
    dataset = load_covid_dataset(n_samples=5000)
    print("Dataset loaded successfully!")
    print(f"Training set shape: {dataset['X_train'].shape}")
    print(f"Feature names: {dataset['feature_names']}")
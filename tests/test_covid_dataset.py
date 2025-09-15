"""
Test module for COVID-19 dataset functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.covid_dataset import COVID19DatasetGenerator, COVID19DataPreprocessor, load_covid_dataset


class TestCOVID19DatasetGenerator:
    """Test cases for COVID19DatasetGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = COVID19DatasetGenerator(random_state=42)
        assert generator.random_state == 42
    
    def test_data_generation(self):
        """Test data generation functionality."""
        generator = COVID19DatasetGenerator(random_state=42)
        data = generator.generate_patient_data(n_samples=100)
        
        # Check data shape and structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'covid_diagnosis' in data.columns
        assert 'age' in data.columns
        assert 'fever' in data.columns
        
        # Check data types
        assert data['covid_diagnosis'].dtype in [int, bool]
        assert data['age'].dtype in [int, float]
        
        # Check value ranges
        assert data['age'].min() >= 0
        assert data['age'].max() <= 100
        assert data['covid_diagnosis'].isin([0, 1]).all()
    
    def test_data_generation_consistency(self):
        """Test that same random state produces consistent results."""
        generator1 = COVID19DatasetGenerator(random_state=42)
        generator2 = COVID19DatasetGenerator(random_state=42)
        
        data1 = generator1.generate_patient_data(n_samples=50)
        data2 = generator2.generate_patient_data(n_samples=50)
        
        # Should be identical with same random state
        pd.testing.assert_frame_equal(data1, data2)


class TestCOVID19DataPreprocessor:
    """Test cases for COVID19DataPreprocessor."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = COVID19DataPreprocessor()
        assert preprocessor.scaler is not None
        assert preprocessor.label_encoder is not None
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        # Generate test data
        generator = COVID19DatasetGenerator(random_state=42)
        data = generator.generate_patient_data(n_samples=100)
        
        # Preprocess data
        preprocessor = COVID19DataPreprocessor()
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(
            data, test_size=0.2, val_size=0.1, random_state=42
        )
        
        # Check shapes
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100
        
        # Check that we have the right proportions (approximately)
        assert len(X_test) == 20  # 20% of 100
        assert len(X_val) == 9   # ~10% of 100 (adjusted for remaining after test split)
        assert len(X_train) == 71  # Remaining
        
        # Check that features are scaled (mean ≈ 0, std ≈ 1)
        assert abs(X_train.mean()) < 0.5  # Should be close to 0
        assert abs(X_train.std() - 1.0) < 0.5  # Should be close to 1
        
        # Check feature names
        feature_names = preprocessor.get_feature_names()
        assert len(feature_names) > 0
        assert 'covid_diagnosis' not in feature_names  # Should be excluded


class TestLoadCovidDataset:
    """Test cases for load_covid_dataset function."""
    
    def test_load_dataset_default(self):
        """Test loading dataset with default parameters."""
        dataset = load_covid_dataset(n_samples=100, random_state=42)
        
        # Check required keys
        required_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test', 
                        'feature_names', 'raw_data', 'preprocessor']
        for key in required_keys:
            assert key in dataset
        
        # Check data types
        assert isinstance(dataset['X_train'], np.ndarray)
        assert isinstance(dataset['y_train'], pd.Series)
        assert isinstance(dataset['feature_names'], list)
        assert isinstance(dataset['raw_data'], pd.DataFrame)
    
    def test_load_dataset_custom_params(self):
        """Test loading dataset with custom parameters."""
        dataset = load_covid_dataset(
            n_samples=200, 
            test_size=0.3, 
            val_size=0.15, 
            random_state=123
        )
        
        total_samples = len(dataset['X_train']) + len(dataset['X_val']) + len(dataset['X_test'])
        assert total_samples == 200
        
        # Check approximate proportions
        assert len(dataset['X_test']) == 60  # 30% of 200
        
    def test_reproducibility(self):
        """Test that same parameters produce identical results."""
        dataset1 = load_covid_dataset(n_samples=50, random_state=42)
        dataset2 = load_covid_dataset(n_samples=50, random_state=42)
        
        # Should have identical results
        np.testing.assert_array_equal(dataset1['X_train'], dataset2['X_train'])
        pd.testing.assert_series_equal(dataset1['y_train'], dataset2['y_train'])


if __name__ == "__main__":
    pytest.main([__file__])
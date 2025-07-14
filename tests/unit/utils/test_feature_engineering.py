import pandas as pd
import numpy as np
from api.utils.feature_engineering import FeatureEngineer

def test_add_features():
    df = pd.DataFrame({
        'Temperature': [25, 26],
        'Pressure': [100, 101],
        'VibrationX': [10, 11],
        'VibrationY': [5, 6],
        'VibrationZ': [3, 4],
        'Frequency': [50, 51]
    })
    engineer = FeatureEngineer()
    
    result = engineer.add_features(df)
    
    assert 'vib_magnitude' in result.columns
    assert 'vib_combined' in result.columns
    assert 'temp_vib_ratio' in result.columns
    assert 'pressure_diff' in result.columns
    assert 'outlier_flag' in result.columns
    

    expected_magnitude = np.sqrt(10**2 + 5**2 + 3**2)
    assert np.isclose(result.loc[0, 'vib_magnitude'], expected_magnitude)
    
    expected_combined = 5 + 0.7 * 3
    assert np.isclose(result.loc[0, 'vib_combined'], expected_combined)
    
    expected_ratio = 25 / (expected_combined + 1e-5)
    assert np.isclose(result.loc[0, 'temp_vib_ratio'], expected_ratio)
    
    assert result.loc[0, 'pressure_diff'] == 0

def test_outlier_detection():
    df = pd.DataFrame({
        'Temperature': [25, 26],
        'Pressure': [160, 100],
        'VibrationX': [10, 11],
        'VibrationY': [130, 50],
        'VibrationZ': [50, 90],
        'Frequency': [50, 51]
    })
    engineer = FeatureEngineer()
    
    result = engineer.add_features(df)
    
    assert result.loc[0, 'outlier_flag'] == 1
    assert result.loc[1, 'outlier_flag'] == 0